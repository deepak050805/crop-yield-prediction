"""Run an image with production restrictions and remove only the test container."""

import argparse
import json
import subprocess
import time
import uuid

from scripts.smoke import smoke


def docker(*arguments, timeout=120):
    result = subprocess.run(
        ["docker", *arguments], check=True, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip()


def inspect_running(name):
    return json.loads(docker("inspect", name))[0]


def container_smoke(image):
    name = "fieldwise-smoke-" + uuid.uuid4().hex[:12]
    started = False
    try:
        docker(
            "create",
            "--name",
            name,
            "--init",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "128",
            "--health-interval",
            "1s",
            "--health-start-period",
            "5s",
            "--health-timeout",
            "3s",
            "--health-retries",
            "10",
            "--publish",
            "127.0.0.1::10000",
            image,
        )
        started = True
        docker("start", name)
        runtime = inspect_running(name)
        port = runtime["NetworkSettings"]["Ports"]["10000/tcp"][0]["HostPort"]
        smoke(f"http://127.0.0.1:{port}", timeout=90)
        deadline = time.monotonic() + 30
        while True:
            runtime = inspect_running(name)
            health = runtime["State"].get("Health", {}).get("Status")
            if health == "healthy":
                break
            if health in {None, "unhealthy"} or time.monotonic() >= deadline:
                raise AssertionError(f"Container did not become healthy: {health}")
            time.sleep(0.5)
        if not runtime["HostConfig"]["ReadonlyRootfs"]:
            raise AssertionError("Container filesystem is writable")
        docker(
            "exec",
            name,
            "python",
            "-c",
            "import os, pathlib, tempfile; "
            "assert os.getuid() != 0, 'runtime user must not be root'; "
            "assert not any(pathlib.Path('/app').rglob('.env*')), 'dotenv entered image'; "
            "assert not pathlib.Path('/app/.git').exists(), 'Git metadata entered image'; "
            "assert not pathlib.Path('/app/backend/tests').exists(), 'tests entered image'; "
            "handle = tempfile.TemporaryFile(dir='/tmp'); handle.close()",
        )
        print(
            "PASS: healthy Linux image, nonroot user, read-only root, temporary storage, context exclusions"
        )
    except Exception:
        if started:
            print(docker("logs", "--tail", "100", name))
        raise
    finally:
        if started:
            docker("rm", "--force", name, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="fieldwise:smoke")
    args = parser.parse_args()
    container_smoke(args.image)


if __name__ == "__main__":
    main()
