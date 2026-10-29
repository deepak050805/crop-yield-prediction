"""Check a running Fieldwise deployment with only Python's standard library."""

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request


def request(base_url, path, *, data=None, content_type=None, status=200):
    headers = {"Accept": "application/json", "User-Agent": "Fieldwise-Smoke/1"}
    if content_type:
        headers["Content-Type"] = content_type
    message = urllib.request.Request(base_url + path, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(message, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(2_000_001)
        if response.status != status:
            raise AssertionError(f"{path}: expected HTTP {status}, received {response.status}")
        if len(body) > 2_000_000:
            raise AssertionError(f"{path}: response exceeds smoke-test limit")
        if "application/json" in response.headers.get("Content-Type", ""):
            return json.loads(body)
        return body


def wait_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            if request(base_url, "/health/ready") == {"status": "ready"}:
                return
        except (OSError, AssertionError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise AssertionError("Deployment did not become ready before the deadline")
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))


def smoke(base_url, timeout=60):
    wait_ready(base_url, timeout)
    assert request(base_url, "/health/live") == {"status": "alive"}
    for path in ("/", "/predictor", "/history"):
        assert b"<!doctype html" in request(base_url, path).lower(), path
    for path in ("/static/js/dashboard.js", "/static/css/dashboard.css"):
        assert request(base_url, path), path
    selection = {"crop": "Wheat", "district": "Amritsar"}
    records = request(
        base_url,
        "/get_data",
        data=json.dumps(selection).encode(),
        content_type="application/json",
    )
    assert records and all(
        row["Crop"] == selection["crop"] and row["District"] == selection["district"]
        for row in records
    ), "Historical selection did not match"
    payload = {**selection, "rainfall": "735", "temperature": "27.5", "humidity": "67"}
    result = request(
        base_url,
        "/predict",
        data=urllib.parse.urlencode(payload).encode(),
        content_type="application/x-www-form-urlencoded",
    )
    assert all(math.isfinite(result[key]) for key in ("predicted", "optimal", "gap"))
    assert result["optimal"] >= result["predicted"]
    invalid = request(
        base_url,
        "/predict",
        data=b"humidity=nan",
        content_type="application/x-www-form-urlencoded",
        status=400,
    )
    assert invalid.get("code") == "invalid_input"
    assert request(base_url, "/not-a-fieldwise-route", status=404).get("code") == "not_found"
    assert request(base_url, "/model_info").get("features")
    print("PASS: readiness, liveness, pages, assets, history, prediction, errors, model report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:10000")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    parsed = urllib.parse.urlsplit(args.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        parser.error("base_url must be an HTTP(S) URL without query or fragment")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a positive finite number")
    smoke(args.base_url.rstrip("/"), args.timeout)


if __name__ == "__main__":
    main()
