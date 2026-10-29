import logging
import unittest

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
