"""Exercise release checks with optimized Python and incorrect API responses."""

import contextlib
import inspect
import io
import types
import unittest
from unittest.mock import patch

from scripts import smoke


class SmokeContractTests(unittest.TestCase):
    def setUp(self):
        self.optimized = types.ModuleType("optimized_smoke")
        exec(
            compile(inspect.getsource(smoke), smoke.__file__, "exec", optimize=2),
            self.optimized.__dict__,
        )

    def responses(self):
        return {
            "/health/live": {"status": "alive"},
            "/": b"<!doctype html><title>Overview</title>",
            "/predictor": b"<!doctype html><title>Predictor</title>",
            "/history": b"<!doctype html><title>History</title>",
            "/static/js/dashboard.js": b"// dashboard",
            "/static/css/dashboard.css": b"body {}",
            "/get_data": [{"Crop": "Wheat", "District": "Amritsar"}],
            "/predict": {"predicted": 3.5, "optimal": 4.0, "gap": 0.5},
            "/invalid": {"code": "invalid_input"},
            "/not-a-fieldwise-route": {"code": "not_found"},
            "/model_info": {"features": ["Crop", "District", "Rainfall"]},
        }

    def run_smoke(self, responses):
        def request(base_url, path, **kwargs):
            key = "/invalid" if kwargs.get("status") == 400 else path
            return responses[key]

        with (
            patch.object(self.optimized, "wait_ready"),
            patch.object(self.optimized, "request", side_effect=request) as calls,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.optimized.smoke("http://127.0.0.1:10000")
        return calls

    def test_optimized_smoke_exercises_every_release_endpoint(self):
        calls = self.run_smoke(self.responses())
        self.assertEqual(calls.call_count, 11)
        self.assertEqual(calls.call_args_list[0].args[1], "/health/live")
        self.assertEqual(calls.call_args_list[-1].args[1], "/model_info")

    def test_optimization_does_not_hide_broken_release_contracts(self):
        failures = (
            ("/health/live", {"status": "down"}, "Liveness"),
            ("/history", b"proxy error", "expected HTML"),
            ("/static/js/dashboard.js", b"", "asset is empty"),
            ("/get_data", [{"Crop": "Rice", "District": "Amritsar"}], "selection"),
            ("/predict", {"predicted": float("nan"), "optimal": 4, "gap": 1}, "non-finite"),
            ("/predict", {"predicted": 5, "optimal": 4, "gap": -1}, "below"),
            ("/invalid", {"code": "internal_error"}, "Invalid input"),
            ("/not-a-fieldwise-route", {"code": "internal_error"}, "Unknown route"),
            ("/model_info", {"features": []}, "metadata"),
        )
        for path, response, message in failures:
            with self.subTest(path=path, message=message):
                responses = self.responses()
                responses[path] = response
                with self.assertRaisesRegex(AssertionError, message):
                    self.run_smoke(responses)


if __name__ == "__main__":
    unittest.main()
