"""Input and dependency checks for the optional A2A boundary."""

import asyncio
import builtins
import os
import unittest
from unittest.mock import patch

from aiython.a2a import send_a2a


class A2AValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_invalid_urls_and_timeouts(self):
        for url in (None, 42, "example.test", "ftp://example.test"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "HTTP"):
                await send_a2a(url, {})
        for timeout in (True, "30", 0, -1, 3601):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "timeout"):
                await send_a2a("https://example.test", {}, timeout=timeout)

    async def test_requires_valid_credential_name_and_value(self):
        for name in ("", "BAD-NAME", "1INVALID"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "api_key_env"):
                await send_a2a("https://example.test", {}, api_key_env=name)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Missing A2A credential"):
                await send_a2a("https://example.test", {}, api_key_env="AIYTHON_TEST_A2A_KEY")
        with patch.dict(os.environ, {"AIYTHON_TEST_A2A_KEY": "secret"}):
            with self.assertRaisesRegex(ValueError, "HTTPS outside loopback"):
                await send_a2a("http://example.test", {}, api_key_env="AIYTHON_TEST_A2A_KEY")

    async def test_reports_missing_optional_dependency(self):
        original_import = builtins.__import__

        def without_a2a(name, *args, **kwargs):
            if name == "a2a.client":
                raise ImportError("simulated absent optional dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_a2a):
            with self.assertRaisesRegex(RuntimeError, r"optional 'aiython\[a2a\]'"):
                await send_a2a("https://example.test", {})

    async def test_rejects_non_json_payload(self):
        with self.assertRaises(TypeError):
            await send_a2a("https://example.test", object())

    async def test_local_authenticated_endpoint_is_allowed(self):
        from unittest.mock import AsyncMock

        class EmptyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def send_message(self, _request):
                if False:
                    yield None

        factory = AsyncMock(return_value=EmptyClient())
        with patch.dict(os.environ, {"AIYTHON_TEST_A2A_KEY": "secret"}), \
                patch("a2a.client.create_client", factory):
            for host in ("localhost", "127.0.0.1", "[::1]"):
                with self.subTest(host=host):
                    self.assertEqual(await send_a2a(
                        f"http://{host}", {}, api_key_env="AIYTHON_TEST_A2A_KEY"), [])
        self.assertEqual(factory.await_count, 3)

    async def test_timeout_stops_exchange(self):
        from unittest.mock import AsyncMock

        async def slow_client(*_args, **_kwargs):
            await asyncio.sleep(1)

        with patch("a2a.client.create_client", AsyncMock(side_effect=slow_client)):
            with self.assertRaises(TimeoutError):
                await send_a2a("https://example.test", {}, timeout=0.001)
