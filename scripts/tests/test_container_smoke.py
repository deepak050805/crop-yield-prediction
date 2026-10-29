"""Check that image verification always cleans up its own container."""

import contextlib
import io
import unittest
from unittest.mock import patch

from scripts import container_smoke


class ContainerSmokeTests(unittest.TestCase):
    def runtime(self, health="healthy", readonly=True):
        return {
            "NetworkSettings": {"Ports": {"10000/tcp": [{"HostPort": "12345"}]}},
            "State": {"Health": {"Status": health}},
            "HostConfig": {"ReadonlyRootfs": readonly},
        }

    def test_success_removes_only_the_unique_test_container(self):
        with (
            patch.object(container_smoke, "docker", return_value="") as docker,
            patch.object(container_smoke, "inspect_running", return_value=self.runtime()),
            patch.object(container_smoke, "smoke") as smoke,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            container_smoke.container_smoke("fieldwise:test")
        smoke.assert_called_once_with("http://127.0.0.1:12345", timeout=90)
        created = docker.call_args_list[0].args
        name = created[created.index("--name") + 1]
        self.assertTrue(name.startswith("fieldwise-smoke-"))
        self.assertIn("--read-only", created)
        self.assertIn("no-new-privileges:true", created)
        self.assertEqual(docker.call_args_list[-1].args, ("rm", "--force", name))

    def test_failed_http_smoke_still_logs_and_cleans_up(self):
        with (
            patch.object(container_smoke, "docker", return_value="") as docker,
            patch.object(container_smoke, "inspect_running", return_value=self.runtime()),
            patch.object(container_smoke, "smoke", side_effect=AssertionError("prediction broken")),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(AssertionError, "prediction broken"),
        ):
            container_smoke.container_smoke("fieldwise:test")
        self.assertEqual(docker.call_args_list[-2].args[:3], ("logs", "--tail", "100"))
        self.assertEqual(docker.call_args_list[-1].args[:2], ("rm", "--force"))

    def test_failed_start_removes_the_created_container(self):
        with (
            patch.object(
                container_smoke, "docker", side_effect=["id", OSError("start failed"), "", ""]
            ) as docker,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(OSError, "start failed"),
        ):
            container_smoke.container_smoke("fieldwise:test")
        self.assertEqual(docker.call_args_list[-1].args[:2], ("rm", "--force"))

    def test_unhealthy_or_writable_containers_fail(self):
        for runtime, message in (
            (self.runtime(health="unhealthy"), "did not become healthy"),
            (self.runtime(readonly=False), "filesystem is writable"),
        ):
            with (
                self.subTest(message=message),
                patch.object(container_smoke, "docker", return_value="") as docker,
                patch.object(container_smoke, "inspect_running", return_value=runtime),
                patch.object(container_smoke, "smoke"),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(AssertionError, message),
            ):
                container_smoke.container_smoke("fieldwise:test")
            self.assertEqual(docker.call_args_list[-1].args[:2], ("rm", "--force"))


if __name__ == "__main__":
    unittest.main()
