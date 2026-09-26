import asyncio
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiython import collaboration as c


class BrokerEdgeTests(unittest.TestCase):
    def setUp(self):
        self.broker = c._Broker()
        self.broker.open('test')
        self.secret = self.broker.invite('test', 'main', secret='secret')

    def request(self, operation, **kwargs):
        return self.broker.perform({'group_id': 'test', 'participant': 'main',
                                    'secret': self.secret, 'operation': operation, **kwargs})

    def test_invitation_and_group_lifecycle_validation(self):
        for name in ('', 42, 'x' * 101):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Participant name'):
                self.broker.invite('test', name)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.broker.invite('test', 'main')
        with patch.object(c, 'MAX_PARTICIPANTS', 1), self.assertRaisesRegex(BufferError, 'full'):
            self.broker.invite('test', 'other')
        self.broker.close('absent')
        self.broker.close('test')
        with self.assertRaisesRegex(ValueError, 'Group is closed'):
            self.request('pending')

    def test_attached_state_and_message_validation(self):
        with self.assertRaisesRegex(ValueError, 'not attached'):
            self.request('pending')
        with self.assertRaisesRegex(ValueError, 'not invited'):
            self.broker.perform({'group_id': 'test', 'participant': 'main',
                                 'secret': 'wrong', 'operation': 'join'})
        self.assertEqual(self.request('join'), {'joined': 'main'})
        with self.assertRaisesRegex(ValueError, 'already attached'):
            self.request('join')
        with self.assertRaisesRegex(ValueError, 'Recipient must'):
            self.request('send', recipient=5)
        with self.assertRaisesRegex(ValueError, 'Unknown recipient'):
            self.request('send', recipient='missing')
        with self.assertRaisesRegex(ValueError, 'reply_to must'):
            self.request('send', recipient='main', reply_to=5)
        with patch.object(c, 'MAX_MESSAGE_BYTES', 1), self.assertRaisesRegex(ValueError, '64 KiB'):
            self.request('send', recipient='main', payload='long')
        for message_id in ('', 3, 'x' * 129):
            with self.subTest(message_id=message_id), self.assertRaisesRegex(ValueError, 'Message ID'):
                self.request('send', recipient='main', payload='ok', message_id=message_id)
        with self.assertRaisesRegex(ValueError, 'Unknown group operation'):
            self.request('unknown')
        self.assertEqual(self.request('leave'), {'left': 'main'})

    def test_read_wait_limits_and_seen_eviction(self):
        self.request('join')
        for limit in (0, 51, True, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'Read limit'):
                self.request('read', limit=limit)
        for timeout in (-1, 121, None, True):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'Wait timeout'):
                self.request('wait', timeout=timeout)
        self.assertFalse(self.request('wait', timeout=0))
        self.request('send', recipient='main', payload={'a': 1}, message_id='first')
        self.assertTrue(self.request('wait', timeout=0))
        self.assertEqual(self.request('read', limit=1)[0]['payload'], {'a': 1})
        self.assertEqual(self.request('read'), [])
        group = self.broker.groups['test']
        for index in range(4095):
            group.seen[('other', str(index))] = {'payload': None}
        self.request('send', recipient='main', payload=2, message_id='last')
        self.assertEqual(len(group.seen), 4096)
        self.assertNotIn(('main', 'first'), group.seen)

    def test_wait_wakes_when_participant_leaves(self):
        self.request('join')
        ready = threading.Event()
        outcome = []
        def wait():
            ready.set()
            try:
                self.request('wait', timeout=2)
            except ValueError as error:
                outcome.append(str(error))
        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(ready.wait(1))
        self.request('leave')
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, ['Group or participant closed while waiting'])

    def test_async_wait_timeout_and_ticket_validation(self):
        ticket = c.Ticket('test', 'main', self.secret, '127.0.0.1', 0,
                          os.getpid(), os.getcwd())
        async def scenario():
            with self.assertRaisesRegex(ValueError, 'Wait timeout'):
                await self.broker.await_message(ticket, -1)
            with self.assertRaisesRegex(ValueError, 'ticket is invalid'):
                await self.broker.await_message(ticket, 0)
            self.request('join')
            self.assertEqual(await self.broker.await_message(ticket, 0), [])
            self.request('send', recipient='main', payload=42, message_id='one')
            with patch.object(c, '_request', side_effect=lambda t, op, **kw: self.request(op, **kw)):
                self.assertEqual((await self.broker.await_message(ticket, 0))[0]['payload'], 42)
        asyncio.run(scenario())

    def test_wake_ignores_closed_event_loop(self):
        class ClosedLoop:
            def call_soon_threadsafe(self, callback):
                raise RuntimeError('closed')
        mailbox = c._Mailbox('secret')
        mailbox.async_waiters.append((ClosedLoop(), object()))
        c._Broker._wake(mailbox)
        self.assertEqual(mailbox.async_waiters, [])


class GroupTransportEdgeTests(unittest.TestCase):
    def test_closing_absent_broker_and_unentered_participant(self):
        with patch.object(c, '_BROKER', None):
            c._close_group('already-closed')
        ticket = c.Ticket('test', 'main', 'secret', '127.0.0.1', 1, -1, os.getcwd())
        with patch.object(c, '_request', return_value={'left': 'main'}) as request:
            c.Participant(ticket).__exit__(None, None, None)
        request.assert_called_once_with(ticket, 'leave')

    def test_remote_sync_and_async_requests_and_errors(self):
        async def async_requests(ticket):
            self.assertEqual(await c._request_async(ticket, 'pending'), 0)
            with self.assertRaisesRegex(ValueError, 'Unknown group operation'):
                await c._request_async(ticket, 'unknown')
        with c.group() as team:
            ticket = replace(team._ticket, owner_pid=-1)
            self.assertEqual(c._request(ticket, 'pending'), 0)
            with self.assertRaisesRegex(ValueError, 'Unknown group operation'):
                c._request(ticket, 'unknown')
            asyncio.run(async_requests(ticket))
            with socket.create_connection((ticket.host, ticket.port)) as connection:
                connection.sendall(b'not json\n')
                self.assertFalse(json.loads(connection.makefile('rb').readline())['ok'])

    def test_request_bounds_and_incomplete_responses(self):
        with c.group() as team:
            ticket = replace(team._ticket, owner_pid=-1)
            with patch.object(c, 'MAX_REQUEST_BYTES', 1), self.assertRaisesRegex(ValueError, 'size limit'):
                c._request(ticket, 'pending')
            async def oversized():
                with patch.object(c, 'MAX_REQUEST_BYTES', 1), self.assertRaisesRegex(ValueError, 'size limit'):
                    await c._request_async(ticket, 'pending')
            asyncio.run(oversized())
            with socket.create_connection((ticket.host, ticket.port)) as connection:
                connection.sendall(b'x' * (c.MAX_REQUEST_BYTES + 1) + b'\n')
                response = json.loads(connection.makefile('rb').readline())
            self.assertEqual(response['error'], 'ValueError')

    def test_public_participant_validation_and_helpers(self):
        unopened = c.Group()
        with self.assertRaisesRegex(RuntimeError, 'Open the group'):
            unopened.invite('worker')
        with c.group() as team:
            ticket = team.invite('worker')
            self.assertEqual(team.peers(), [])
            with c.join(ticket) as worker:
                self.assertEqual(team.peers(), ['worker'])
                self.assertEqual(worker.peers(), ['main'])
                self.assertEqual(worker.pending(), 0)
                self.assertEqual(worker.wait(0), [])
                team.send('worker', 'message')
                self.assertEqual(worker.wait(0)[0]['payload'], 'message')
                errors = []
                thread = threading.Thread(target=lambda: errors.append(self._foreign_use(worker)))
                thread.start()
                thread.join()
                self.assertEqual(errors, ['Participant must be used by the task that joined it'])
            self.assertEqual(team.wait(0), [])

    def test_remote_async_participant_wait_and_a2a(self):
        async def scenario():
            with c.group() as team:
                ticket = replace(team.invite('worker'), owner_pid=-1)
                async with c.join(ticket) as worker:
                    self.assertEqual(await worker.wait_async(0), [])
                    team.send('worker', 42)
                    self.assertEqual((await worker.wait_async(0))[0]['payload'], 42)
                    with patch('aiython.a2a.send_a2a', new_callable=AsyncMock,
                               return_value=['response']) as send:
                        self.assertEqual(await worker.send_a2a('https://example.test', {'hi': 1}),
                                         ['response'])
                    self.assertEqual(send.await_args.kwargs['group_id'], ticket.group_id)
                    self.assertEqual(send.await_args.kwargs['sender'], 'worker')
        asyncio.run(scenario())

    def test_broker_handles_malformed_request_and_disconnected_client(self):
        handler = object.__new__(c._RequestHandler)
        handler.rfile = BytesIO(b'not json\n')
        handler.wfile = SimpleNamespace(write=Mock(side_effect=BrokenPipeError()))
        handler.server = SimpleNamespace(broker=Mock())
        handler.handle()
        handler.wfile.write.assert_called_once()

    def test_group_entry_failure_and_unentered_exit(self):
        unopened = c.Group()
        unopened.__exit__(None, None, None)
        with patch.object(c._Broker, 'invite', side_effect=ValueError('invite failed')):
            with self.assertRaisesRegex(ValueError, 'invite failed'):
                with c.group():
                    pass
        with patch.object(c.Participant, '__enter__', side_effect=RuntimeError('join failed')):
            with self.assertRaisesRegex(RuntimeError, 'join failed'):
                with c.group():
                    pass
        self.assertIsNone(c._BROKER)

    def test_spawn_bootstrap_nested_group_reference_counts(self):
        module = SimpleNamespace(__spec__='original')
        with patch('importlib.util.find_spec', return_value='worker-spec'):
            c._enable_spawn(module)
            c._enable_spawn(module)
        self.assertEqual(module.__spec__, 'worker-spec')
        c._disable_spawn(module)
        self.assertEqual(module.__spec__, 'worker-spec')
        c._disable_spawn(module)
        self.assertEqual(module.__spec__, 'original')

    def test_group_entry_cleans_joined_owner_after_spawn_failure(self):
        main = sys.modules['__main__']
        original = getattr(main, '__file__', None)
        try:
            main.__file__ = 'test-entry.py'
            with patch.dict(os.environ, {'AIYTHON_SPAWN_ENTRY': 'test-entry.py'}), \
                    patch.object(c, '_enable_spawn', side_effect=RuntimeError('spawn failed')):
                with self.assertRaisesRegex(RuntimeError, 'spawn failed'):
                    with c.group():
                        pass
            self.assertIsNone(c.current())
            self.assertIsNone(c._BROKER)
        finally:
            if original is None:
                del main.__file__
            else:
                main.__file__ = original

    def test_request_and_async_request_reject_incomplete_wire_response(self):
        class Connection:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def settimeout(self, *_): pass
            def sendall(self, *_): pass
            def makefile(self, *_): return BytesIO(b'incomplete')
        ticket = c.Ticket('test', 'main', 'secret', '127.0.0.1', 1, -1, os.getcwd())
        with patch.object(socket, 'create_connection', return_value=Connection()):
            with self.assertRaisesRegex(ConnectionError, 'incomplete response'):
                c._request(ticket, 'pending')
        writer = SimpleNamespace(write=Mock(), drain=AsyncMock(), close=Mock(),
                                 wait_closed=AsyncMock())
        reader = SimpleNamespace(readline=AsyncMock(return_value=b'incomplete'))
        async def run():
            with patch.object(asyncio, 'open_connection', AsyncMock(return_value=(reader, writer))):
                with self.assertRaisesRegex(ConnectionError, 'incomplete response'):
                    await c._request_async(ticket, 'pending')
        asyncio.run(run())

    @staticmethod
    def _foreign_use(worker):
        try:
            worker.pending()
        except RuntimeError as error:
            return str(error)


class FallbackWorkerEdgeTests(unittest.TestCase):
    def test_worker_constructor_cleans_directory_if_process_fails(self):
        directory = SimpleNamespace(cleanup=Mock())
        with patch('tempfile.TemporaryDirectory', return_value=directory), \
                patch('subprocess.Popen', side_effect=OSError('spawn failed')):
            with self.assertRaisesRegex(OSError, 'spawn failed'):
                c._FallbackWorker(os.getcwd())
        directory.cleanup.assert_called_once()

    def test_worker_rejects_missing_control_pipe_and_cleans_request(self):
        with tempfile.TemporaryDirectory() as root:
            worker = object.__new__(c._FallbackWorker)
            worker.directory = SimpleNamespace(name=root)
            worker.sequence = 0
            worker.process = SimpleNamespace(stdin=None)
            ticket = c.Ticket('test', 'main', 'secret', '127.0.0.1', 1, -1, root)
            with self.assertRaisesRegex(RuntimeError, 'control pipe is unavailable'):
                worker.execute(ticket, 'module', 'function', (), {})
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_worker_close_handles_missing_pipe_broken_pipe_and_timeout(self):
        for stdin, broken, expired in ((None, False, False), (Mock(), True, True)):
            worker = object.__new__(c._FallbackWorker)
            if broken:
                stdin.close.side_effect = BrokenPipeError()
            process = SimpleNamespace(stdin=stdin, wait=Mock(), terminate=Mock())
            if expired:
                process.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), 0]
            worker.process = process
            worker.directory = SimpleNamespace(cleanup=Mock())
            worker.close()
            worker.directory.cleanup.assert_called_once()
            self.assertEqual(process.terminate.call_count, int(expired))

    def test_subinterpreter_probe_handles_older_python(self):
        real_import = __import__
        def imported(name, *args, **kwargs):
            if name == 'concurrent':
                raise ImportError('interpreters unavailable')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=imported):
            self.assertFalse(c._is_subinterpreter())


if __name__ == '__main__':
    unittest.main()
