"""Exercise deployment checks against a real ephemeral HTTP server."""

import contextlib
import http.server
import io
import json
import threading
import unittest
from unittest.mock import patch

from scripts import smoke


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        code, body, content_type = 200, b"ok", "text/plain"
        if self.path == "/json":
            body, content_type = b'{"ok": true}', "application/json"
        elif self.path == "/missing":
            code, body, content_type = 404, b'{"code": "not_found"}', "application/json"
        elif self.path == "/large":
            body = b"x" * 2_000_001
        elif self.path == "/malformed":
            body, content_type = b"broken", "application/json"
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"body": body.decode()}).encode())


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(5)

    def test_decodes_json_and_keeps_non_json_bytes(self):
        self.assertEqual(smoke.request(self.base, "/json"), {"ok": True})
        self.assertEqual(smoke.request(self.base, "/"), b"ok")

    def test_checks_http_error_status_and_body(self):
        self.assertEqual(smoke.request(self.base, "/missing", status=404)["code"], "not_found")
        with self.assertRaisesRegex(AssertionError, "expected HTTP 200, received 404"):
            smoke.request(self.base, "/missing")

    def test_sends_payload_and_rejects_malformed_or_oversized_body(self):
        self.assertEqual(smoke.request(self.base, "/", data=b"a=1"), {"body": "a=1"})
        with self.assertRaisesRegex(AssertionError, "exceeds smoke-test limit"):
            smoke.request(self.base, "/large")
        with self.assertRaises(ValueError):
            smoke.request(self.base, "/malformed")

    def test_readiness_retries_transient_failures(self):
        with patch.object(smoke, "request", side_effect=[OSError("starting"), {"status": "ready"}]):
            with patch.object(smoke.time, "sleep"):
                smoke.wait_ready(self.base, 1)

    def test_readiness_has_a_deadline(self):
        with patch.object(smoke, "request", return_value={"status": "not_ready"}):
            with patch.object(smoke.time, "monotonic", side_effect=[0, 2]):
                with self.assertRaisesRegex(AssertionError, "deadline"):
                    smoke.wait_ready(self.base, 1)

    def test_readiness_reduces_socket_timeout_as_deadline_approaches(self):
        with (
            patch.object(smoke, "request", side_effect=TimeoutError("starting")) as request,
            patch.object(smoke.time, "monotonic", side_effect=[0, 0.2, 0.6, 0.9, 1.01]),
            patch.object(smoke.time, "sleep") as sleep,
            self.assertRaisesRegex(AssertionError, "deadline"),
        ):
            smoke.wait_ready(self.base, 1)
        self.assertEqual(request.call_count, 2)
        self.assertAlmostEqual(request.call_args_list[0].kwargs["timeout"], 0.8)
        self.assertAlmostEqual(request.call_args_list[1].kwargs["timeout"], 0.1)
        sleep.assert_called_once_with(0.25)

    def test_readiness_never_starts_request_after_deadline(self):
        with (
            patch.object(smoke, "request") as request,
            patch.object(smoke.time, "monotonic", side_effect=[0, 1]),
            self.assertRaisesRegex(AssertionError, "deadline"),
        ):
            smoke.wait_ready(self.base, 1)
        request.assert_not_called()

    def test_direct_readiness_calls_reject_invalid_timeouts(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with (
                self.subTest(timeout=timeout),
                patch.object(smoke, "request") as request,
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                smoke.wait_ready(self.base, timeout)
            request.assert_not_called()

    def test_cli_rejects_invalid_urls_and_timeouts(self):
        for arguments in (["file:///tmp/app"], ["http://localhost?q=x"], ["--timeout", "nan"]):
            with self.subTest(arguments=arguments), patch("sys.argv", ["smoke", *arguments]):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    smoke.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
