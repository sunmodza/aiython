"""Optional, scheduler-independent mailboxes for cooperating Aiython tasks.

Tickets are serializable and can be passed to threads, tasks, interpreters or
processes. A local participant calls the broker directly; another interpreter
uses the same operations over a loopback socket. No service starts until a
group is explicitly opened.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import sys
import threading
import time
from uuid import uuid4


MAX_MESSAGE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PENDING = 100
MAX_PARTICIPANTS = 128
CURRENT_PARTICIPANT: ContextVar[Participant | None] = ContextVar(
    "aiython_participant", default=None)


def _task_identity():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@dataclass(frozen=True)
class Ticket:
    group_id: str
    participant: str
    secret: str = field(repr=False)
    host: str
    port: int
    owner_pid: int
    project_root: str


class _Mailbox:
    def __init__(self, secret):
        self.secret = secret
        self.messages = deque()
        self.attached = False
        self.sequence = 0
        self.async_waiters = []


class _GroupState:
    def __init__(self):
        self.members: dict[str, _Mailbox] = {}
        self.seen: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self.closed = False


class _Broker:
    def __init__(self):
        self.condition = threading.Condition()
        self.groups: dict[str, _GroupState] = {}

    def open(self, group_id):
        with self.condition:
            self.groups[group_id] = _GroupState()

    def invite(self, group_id, name, *, secret=None):
        if not isinstance(name, str) or not name or len(name) > 100:
            raise ValueError("Participant name must contain 1 to 100 characters")
        with self.condition:
            group = self.groups[group_id]
            if name in group.members:
                raise ValueError(f"Participant {name!r} already exists")
            if len(group.members) >= MAX_PARTICIPANTS:
                raise BufferError("Collaboration group is full")
            secret = secret or secrets.token_urlsafe(32)
            group.members[name] = _Mailbox(secret)
            return secret

    def close(self, group_id):
        with self.condition:
            group = self.groups.pop(group_id, None)
            if group:
                group.closed = True
                for mailbox in group.members.values():
                    self._wake(mailbox)
            self.condition.notify_all()

    @staticmethod
    def _wake(mailbox):
        for loop, future in mailbox.async_waiters:
            try:
                loop.call_soon_threadsafe(lambda future=future: None if future.done() else future.set_result(None))
            except RuntimeError:
                pass
        mailbox.async_waiters.clear()

    async def await_message(self, ticket, timeout):
        if type(timeout) not in (int, float) or not 0 <= timeout <= 120:
            raise ValueError("Wait timeout must be between 0 and 120 seconds")
        loop = asyncio.get_running_loop()
        with self.condition:
            group = self.groups.get(ticket.group_id)
            mailbox = group.members.get(ticket.participant) if group else None
            if mailbox is None or group.closed or not mailbox.attached or not secrets.compare_digest(ticket.secret, mailbox.secret):
                raise ValueError("Group is closed or the ticket is invalid")
            if mailbox.messages:
                return _request(ticket, "read", limit=1)
            future = loop.create_future()
            entry = (loop, future)
            mailbox.async_waiters.append(entry)
        try:
            try:
                await asyncio.wait_for(future, timeout)
            except TimeoutError:
                return []
            return _request(ticket, "read", limit=1)
        finally:
            with self.condition:
                if entry in mailbox.async_waiters:
                    mailbox.async_waiters.remove(entry)

    def perform(self, request):
        group_id = request.get("group_id")
        with self.condition:
            group = self.groups.get(group_id)
            if group is None or group.closed:
                raise ValueError("Group is closed or the ticket is invalid")
            name = request.get("participant")
            mailbox = group.members.get(name)
            if mailbox is None or not secrets.compare_digest(
                    str(request.get("secret", "")), mailbox.secret):
                raise ValueError("Participant is not invited or the ticket is invalid")
            operation = request.get("operation")
            if operation == "join":
                if mailbox.attached:
                    raise ValueError("Participant is already attached")
                mailbox.attached = True
                return {"joined": name}
            if not mailbox.attached:
                raise ValueError("Participant is not attached")
            if operation == "leave":
                mailbox.attached = False
                self._wake(mailbox)
                self.condition.notify_all()
                return {"left": name}
            if operation == "peers":
                return sorted(peer for peer, box in group.members.items()
                              if box.attached and peer != name)
            if operation == "pending":
                return len(mailbox.messages)
            if operation == "send":
                recipient = request.get("recipient")
                if not isinstance(recipient, str):
                    raise ValueError("Recipient must be a participant name")
                target = group.members.get(recipient)
                if target is None:
                    raise ValueError(f"Unknown recipient: {recipient!r}")
                reply_to = request.get("reply_to")
                if reply_to is not None and (not isinstance(reply_to, str) or len(reply_to) > 128):
                    raise ValueError("reply_to must be a message ID")
                payload = request.get("payload")
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                if len(encoded) > MAX_MESSAGE_BYTES:
                    raise ValueError("Message payload exceeds 64 KiB")
                payload = json.loads(encoded)
                message_id = request.get("message_id")
                if not isinstance(message_id, str) or not message_id or len(message_id) > 128:
                    raise ValueError("Message ID must contain 1 to 128 characters")
                key = (name, message_id)
                if key in group.seen:
                    previous = group.seen[key]
                    if (previous["recipient"] != recipient or previous["payload"] != payload
                            or previous["reply_to"] != reply_to):
                        raise ValueError("Message ID was reused with different content")
                    return previous
                if len(target.messages) >= MAX_PENDING:
                    raise BufferError("Recipient mailbox is full")
                mailbox.sequence += 1
                message = {"id": message_id, "group_id": group_id,
                           "sender": name, "recipient": recipient,
                           "sequence": mailbox.sequence, "reply_to": reply_to,
                           "payload": payload, "created_at": time.time()}
                target.messages.append(message)
                group.seen[key] = message
                if len(group.seen) > 4096:
                    group.seen.popitem(last=False)
                self.condition.notify_all()
                self._wake(target)
                return message
            if operation in ("read", "wait"):
                limit = request.get("limit", 1)
                if type(limit) is not int or not 1 <= limit <= 50:
                    raise ValueError("Read limit must be between 1 and 50")
                if operation == "wait":
                    timeout = request.get("timeout")
                    if type(timeout) not in (int, float) or not 0 <= timeout <= 120:
                        raise ValueError("Wait timeout must be between 0 and 120 seconds")
                    deadline = time.monotonic() + timeout
                    while not mailbox.messages and not group.closed and mailbox.attached:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            break
                        self.condition.wait(left)
                    if group.closed or not mailbox.attached:
                        raise ValueError("Group or participant closed while waiting")
                    return bool(mailbox.messages)
                return [mailbox.messages.popleft() for _ in range(min(limit, len(mailbox.messages)))]
            raise ValueError(f"Unknown group operation: {operation!r}")


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise ValueError("Group request exceeds the size limit")
            result = self.server.broker.perform(json.loads(raw))
            response = {"ok": True, "result": result}
        except (ValueError, TypeError, BufferError, json.JSONDecodeError) as exc:
            response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        try:
            self.wfile.write(json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, broker):
        self.broker = broker
        super().__init__(("127.0.0.1", 0), _RequestHandler)


_LOCK = threading.Lock()
_BROKER: _Broker | None = None
_SERVER: _Server | None = None
_OWNER_PID: int | None = None
_SPAWN_SPECS = {}


def _open_group(group_id):
    global _BROKER, _SERVER, _OWNER_PID
    with _LOCK:
        if _SERVER is None or _OWNER_PID != os.getpid():
            _BROKER = _Broker()
            _SERVER = _Server(_BROKER)
            _OWNER_PID = os.getpid()
            threading.Thread(target=_SERVER.serve_forever,
                             kwargs={"poll_interval": 0.05}, daemon=True,
                             name="aiython-group-broker").start()
        _BROKER.open(group_id)
        return _BROKER, _SERVER.server_address


def _close_group(group_id):
    global _BROKER, _SERVER, _OWNER_PID
    with _LOCK:
        if _BROKER is None:
            return
        _BROKER.close(group_id)
        if _BROKER.groups or _SERVER is None or _OWNER_PID != os.getpid():
            return
        _SERVER.shutdown()
        _SERVER.server_close()
        _SERVER = None
        _BROKER = None
        _OWNER_PID = None


def _enable_spawn(module):
    import importlib.util
    with _LOCK:
        key = id(module)
        record = _SPAWN_SPECS.get(key)
        if record is None:
            original = module.__spec__
            module.__spec__ = importlib.util.find_spec("aiython._worker_main")
            _SPAWN_SPECS[key] = (module, original, 1)
        else:
            _SPAWN_SPECS[key] = (record[0], record[1], record[2] + 1)


def _disable_spawn(module):
    with _LOCK:
        key = id(module)
        record = _SPAWN_SPECS[key]
        if record[2] == 1:
            module.__spec__ = record[1]
            del _SPAWN_SPECS[key]
        else:
            _SPAWN_SPECS[key] = (record[0], record[1], record[2] - 1)


def _request(ticket: Ticket, operation: str, **arguments):
    request = {"group_id": ticket.group_id, "secret": ticket.secret,
               "participant": ticket.participant, "operation": operation, **arguments}
    if os.getpid() == ticket.owner_pid and _BROKER is not None:
        return _BROKER.perform(request)
    wire = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    if len(wire) > MAX_REQUEST_BYTES:
        raise ValueError("Group request exceeds the size limit")
    with socket.create_connection((ticket.host, ticket.port), timeout=5) as connection:
        connection.settimeout(max(5, float(arguments.get("timeout", 0)) + 5))
        connection.sendall(wire + b"\n")
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_RESPONSE_BYTES + 1)
    if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
        raise ConnectionError("Group broker returned an incomplete response")
    response = json.loads(line)
    if not response.get("ok"):
        error = BufferError if response.get("error") == "BufferError" else ValueError
        raise error(response.get("message", "Group operation failed"))
    return response["result"]


async def _request_async(ticket: Ticket, operation: str, **arguments):
    request = {"group_id": ticket.group_id, "secret": ticket.secret,
               "participant": ticket.participant, "operation": operation, **arguments}
    wire = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    if len(wire) > MAX_REQUEST_BYTES:
        raise ValueError("Group request exceeds the size limit")
    reader, writer = await asyncio.open_connection(ticket.host, ticket.port,
                                                    limit=MAX_RESPONSE_BYTES + 1)
    try:
        writer.write(wire + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=max(5, float(arguments.get("timeout", 0)) + 5))
    finally:
        writer.close()
        await writer.wait_closed()
    if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
        raise ConnectionError("Group broker returned an incomplete response")
    response = json.loads(line)
    if not response.get("ok"):
        error = BufferError if response.get("error") == "BufferError" else ValueError
        raise error(response.get("message", "Group operation failed"))
    return response["result"]


class Participant:
    def __init__(self, ticket: Ticket):
        self.ticket = ticket
        self._context_token = None
        self._owner = None

    @property
    def name(self):
        return self.ticket.participant

    def __enter__(self):
        _request(self.ticket, "join")
        self._owner = (os.getpid(), threading.get_ident(), _task_identity())
        self._context_token = CURRENT_PARTICIPANT.set(self)
        return self

    def __exit__(self, *_):
        if self._context_token is not None:
            CURRENT_PARTICIPANT.reset(self._context_token)
            self._context_token = None
        self._owner = None
        _request(self.ticket, "leave")

    def _check_owner(self):
        if self._owner != (os.getpid(), threading.get_ident(), _task_identity()):
            raise RuntimeError("Participant must be used by the task that joined it")

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *error):
        self.__exit__(*error)

    def peers(self):
        self._check_owner()
        return _request(self.ticket, "peers")

    def pending(self):
        self._check_owner()
        return _request(self.ticket, "pending")

    def send(self, recipient, payload, *, reply_to=None, message_id=None):
        self._check_owner()
        return _request(self.ticket, "send", recipient=recipient, payload=payload,
                        reply_to=reply_to, message_id=message_id or uuid4().hex)

    def read(self, limit=20):
        self._check_owner()
        return _request(self.ticket, "read", limit=limit)

    def wait(self, timeout=30):
        self._check_owner()
        return self.read(1) if _request(self.ticket, "wait", timeout=timeout, limit=1) else []

    async def wait_async(self, timeout=30):
        self._check_owner()
        if os.getpid() == self.ticket.owner_pid and _BROKER is not None:
            return await _BROKER.await_message(self.ticket, timeout)
        return self.read(1) if await _request_async(self.ticket, "wait", timeout=timeout, limit=1) else []

    async def send_a2a(self, url, payload, *, timeout=120, api_key_env=None):
        self._check_owner()
        from .a2a import send_a2a
        return await send_a2a(url, payload, group_id=self.ticket.group_id,
                              sender=self.name, timeout=timeout,
                              api_key_env=api_key_env)


class Group:
    def __init__(self, project_root=None):
        if project_root is None:
            from .cli import ProjectFinder
            runtime = next((finder.runtime for finder in sys.meta_path
                            if isinstance(finder, ProjectFinder)), None)
            project_root = runtime.config.project_root if runtime is not None else Path.cwd()
        self.project_root = str(Path(project_root).resolve())
        self._ticket: Ticket | None = None
        self._owner: Participant | None = None
        self._main_module = None

    def __enter__(self):
        group_id, secret = uuid4().hex, secrets.token_urlsafe(32)
        broker, address = _open_group(group_id)
        try:
            broker.invite(group_id, "main", secret=secret)
            self._ticket = Ticket(group_id, "main", secret, address[0], address[1],
                                  os.getpid(), self.project_root)
            self._owner = Participant(self._ticket)
            self._owner.__enter__()
            # Aiython source needs its import-safe bootstrap when Python spawns
            # from this group. Outside the group, preserve normal __spec__.
            import sys
            main_module = sys.modules.get("__main__")
            entry = os.environ.get("AIYTHON_SPAWN_ENTRY")
            if main_module is not None and entry and getattr(main_module, "__file__", None) == entry:
                self._main_module = main_module
                _enable_spawn(main_module)
        except BaseException:
            if self._owner is not None and self._owner._context_token is not None:
                self._owner.__exit__(None, None, None)
            _close_group(group_id)
            raise
        return self

    def __exit__(self, *error):
        try:
            if self._owner:
                self._owner.__exit__(*error)
        finally:
            if self._main_module is not None:
                _disable_spawn(self._main_module)
            if self._ticket:
                _close_group(self._ticket.group_id)

    def invite(self, name):
        if self._ticket is None or _BROKER is None:
            raise RuntimeError("Open the group before inviting a participant")
        secret = _BROKER.invite(self._ticket.group_id, name)
        return Ticket(self._ticket.group_id, name, secret,
                      self._ticket.host, self._ticket.port, self._ticket.owner_pid,
                      self._ticket.project_root)

    def peers(self):
        return self._owner.peers()

    def pending(self):
        return self._owner.pending()

    def send(self, recipient, payload, **kwargs):
        return self._owner.send(recipient, payload, **kwargs)

    def read(self, limit=20):
        return self._owner.read(limit)

    def wait(self, timeout=30):
        return self._owner.wait(timeout)


def group(project_root=None):
    return Group(project_root)


def join(ticket: Ticket):
    return Participant(ticket)


def current():
    participant = CURRENT_PARTICIPANT.get()
    if participant is None:
        return None
    return participant if participant._owner == (
        os.getpid(), threading.get_ident(), _task_identity()) else None


class _FallbackWorker:
    """One reusable native-capable process owned by a subinterpreter."""

    def __init__(self, root):
        import subprocess
        import sys
        import tempfile

        self.directory = tempfile.TemporaryDirectory(prefix="aiython-worker-")
        self.sequence = 0
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "aiython._subprocess_worker"],
                stdin=subprocess.PIPE, cwd=root)
        except BaseException:
            self.directory.cleanup()
            raise

    def execute(self, ticket, module, function, args, kwargs):
        import pickle

        self.sequence += 1
        directory = Path(self.directory.name)
        request = directory / f"request-{self.sequence}.pickle"
        response = directory / f"response-{self.sequence}.pickle"
        with request.open("xb") as file:
            pickle.dump((ticket, module, function, args, kwargs), file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        try:
            if self.process.stdin is None:
                raise RuntimeError("Aiython worker control pipe is unavailable")
            self.process.stdin.write((json.dumps([str(request), str(response)]) + "\n").encode())
            self.process.stdin.flush()
            while not response.is_file():
                if (status := self.process.poll()) is not None:
                    raise RuntimeError(f"Aiython worker process exited with status {status}")
                time.sleep(0.001)
            with response.open("rb") as file:
                result = pickle.load(file)
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)
        if result[0] == "error":
            raise RuntimeError(f"Aiython worker failed ({result[1]}): {result[2]}")
        return result[1]

    def close(self):
        import subprocess

        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=1)
        self.directory.cleanup()


_FALLBACK_WORKERS = {}
_FALLBACK_LOCK = threading.Lock()


def _close_fallback_workers():
    with _FALLBACK_LOCK:
        workers = list(_FALLBACK_WORKERS.values())
        _FALLBACK_WORKERS.clear()
    for worker in workers:
        worker.close()


def _subprocess_entry(ticket, module, function, args, kwargs):
    """Use the native-capable process belonging to this interpreter/root."""
    root = str(Path(ticket.project_root).resolve())
    with _FALLBACK_LOCK:
        worker = _FALLBACK_WORKERS.get(root)
        if worker is None or worker.process.poll() is not None:
            if worker is not None:
                worker.close()
            worker = _FallbackWorker(root)
            _FALLBACK_WORKERS[root] = worker
        return worker.execute(ticket, module, function, args, kwargs)


import atexit
atexit.register(_close_fallback_workers)


def _is_subinterpreter():
    try:
        from concurrent import interpreters
    except ImportError:
        return False
    return interpreters.get_current() != interpreters.get_main()


def worker_entry(ticket: Ticket, module: str, function: str, *args, **kwargs):
    """Picklable entry for an importable worker function and group ticket."""
    import importlib
    import sys

    # A worker may import another project module containing an AI block after
    # its own import. Native model/config extensions are not subinterpreter
    # safe, so move the whole worker before any project code can execute.
    if _is_subinterpreter():
        return _subprocess_entry(ticket, module, function, args, kwargs)

    root = Path(ticket.project_root)
    sys.path.insert(0, str(root))
    try:
        from .cli import ProjectFinder
        from .config import resolve
        from .runtime import Runtime
        finder = ProjectFinder(Runtime(resolve(root / "__aiython_worker__.py")))
        sys.meta_path.insert(0, finder)
        try:
            imported = importlib.import_module(module)
            target = getattr(imported, function)
            return target(ticket, *args, **kwargs)
        finally:
            sys.meta_path.remove(finder)
    finally:
        sys.path.remove(str(root))
