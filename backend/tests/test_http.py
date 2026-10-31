import logging
import tempfile
import unittest
from pathlib import Path

from flask import Flask, jsonify

from backend.http import register_http_errors


class HttpLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.INFO)
        register_http_errors(self.app)

        @self.app.get("/ok")
        def ok():
            return jsonify(status="ok")

        @self.app.get("/crash")
        def crash():
            raise RuntimeError("private implementation details")

        self.client = self.app.test_client()

    def test_ids_are_generated_and_consistent_in_json_failures(self):
        response = self.client.get("/absent", headers={"X-Request-ID": "untrusted-client-value"})
        request_id = response.headers["X-Request-ID"]
        self.assertRegex(request_id, "^[0-9a-f]{32}$")
        self.assertEqual(response.get_json()["request_id"], request_id)
        self.assertNotEqual(request_id, self.client.get("/ok").headers["X-Request-ID"])

    def test_request_logs_omit_query_values_and_user_request_ids(self):
        with self.assertLogs(self.app.logger, level="INFO") as captured:
            self.client.get("/ok?token=private-query", headers={"X-Request-ID": "private-header"})
        output = "\n".join(captured.output)
        self.assertIn("route=/ok", output)
        self.assertIn("status=200", output)
        self.assertIn("duration_ms=", output)
        self.assertNotIn("private-query", output)
        self.assertNotIn("private-header", output)

    def test_unhandled_errors_hide_exception_details_and_retain_trace_id(self):
        with self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.get("/crash")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["code"], "internal_error")
        self.assertNotIn("private implementation", response.get_data(as_text=True))
        self.assertEqual(response.get_json()["request_id"], response.headers["X-Request-ID"])

    def test_security_headers_cover_success_errors_and_head(self):
        for path in ["/ok", "/missing"]:
            for method in ["get", "head"]:
                with self.subTest(path=path, method=method):
                    response = getattr(self.client, method)(path)
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                    self.assertEqual(response.headers["X-Frame-Options"], "DENY")
                    self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
                    self.assertIn(
                        "frame-ancestors 'none'", response.headers["Content-Security-Policy"]
                    )
                    self.assertNotIn("Strict-Transport-Security", response.headers)
                    if method == "head":
                        self.assertEqual(response.data, b"")

    def test_tls_and_development_headers_require_explicit_modes(self):
        self.app.config["HTTPS"] = True
        self.assertEqual(
            self.client.get("/ok").headers["Strict-Transport-Security"], "max-age=31536000"
        )
        self.app.debug = True
        response = self.client.get("/ok")
        self.assertNotIn("Content-Security-Policy", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_static_assets_keep_conditional_caching_and_security_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "asset.css").write_text("body { color: green; }", encoding="utf-8")
            self.app.static_folder = directory
            with self.client.get("/static/asset.css") as response:
                self.assertEqual(response.status_code, 200)
                self.assertNotEqual(response.headers["Cache-Control"], "no-store")
                etag = response.headers["ETag"]
            with self.client.get("/static/asset.css", headers={"If-None-Match": etag}) as response:
                self.assertEqual(response.status_code, 304)
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
