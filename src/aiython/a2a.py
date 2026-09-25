"""Optional A2A boundary for contacting an independently hosted agent."""
from __future__ import annotations

import asyncio
import json
import os
import re
from urllib.parse import urlsplit


async def send_a2a(url: str, payload, *, group_id: str | None = None,
                   sender: str | None = None, timeout: float = 120,
                   api_key_env: str | None = None) -> list[dict]:
    """Send one task message through the official A2A SDK without retrying it.

    A2A task responses are returned verbatim as JSON-compatible dictionaries;
    the caller decides how to use status, artifacts and follow-up task IDs.
    """
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("A2A agent URL must be HTTP(S)")
    if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
        raise ValueError("A2A timeout must be between 0 and 3600 seconds")
    if api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("api_key_env must name an environment variable")
    credential = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not credential:
        raise ValueError(f"Missing A2A credential in {api_key_env}")
    if credential and urlsplit(url).scheme != "https" and urlsplit(url).hostname not in {
            "localhost", "127.0.0.1", "::1"}:
        raise ValueError("Authenticated A2A endpoints must use HTTPS outside loopback")
    try:
        from a2a.client import create_client
        from a2a.client.client import ClientConfig
        from a2a.helpers import new_text_message
        from a2a.types import Role, SendMessageRequest
        from google.protobuf.json_format import MessageToDict
        import httpx
    except ImportError as exc:
        raise RuntimeError("A2A support requires the optional 'aiython[a2a]' dependency") from exc
    envelope = {"payload": payload}
    if group_id is not None:
        envelope["group_id"] = group_id
    if sender is not None:
        envelope["sender"] = sender
    content = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
    async def exchange(client):
        async with client:
            message = new_text_message(content, role=Role.ROLE_USER)
            response = []
            async for event in client.send_message(SendMessageRequest(message=message)):
                response.append(MessageToDict(event, preserving_proto_field_name=True))
            return response
    async with asyncio.timeout(timeout):
        if credential:
            async with httpx.AsyncClient(headers={"Authorization": f"Bearer {credential}"}) as http:
                return await exchange(await create_client(url, client_config=ClientConfig(
                    streaming=False, httpx_client=http)))
        return await exchange(await create_client(url, client_config=ClientConfig(streaming=False)))
