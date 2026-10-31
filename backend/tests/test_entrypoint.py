import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from backend.config import BASE_DIR


class DirectEntrypointTests(unittest.TestCase):
    def test_script_import_avoids_stdlib_shadowing_from_another_directory(self):
        # Reproduce Python's direct-script import path in a fresh interpreter.
        # Loading under a non-main name exercises startup without a listening server.
        source = textwrap.dedent("""
            import runpy
            import sys
            from pathlib import Path
            script = Path(sys.argv[1])
            sys.path.insert(0, str(script.parent))
            namespace = runpy.run_path(str(script), run_name="entrypoint_regression")
            import http.server
            assert not Path(http.server.__file__).resolve().is_relative_to(script.parent)
            assert namespace["app"].test_client().get("/health/ready").status_code == 200
            print("entrypoint ready")
        """)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-c", source, str(BASE_DIR / "app.py")],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("entrypoint ready", result.stdout)
