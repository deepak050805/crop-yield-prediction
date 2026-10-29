"""Shared JSON failure responses for API and framework errors."""

import time
import uuid

from flask import g, jsonify, request
from werkzeug.exceptions import HTTPException

from backend.validation import ValidationError


def register_http_errors(app):
    @app.before_request
    def begin_request():
        g.request_id = uuid.uuid4().hex
        g.request_started = time.perf_counter()

    @app.after_request
    def record_request(response):
        request_id = getattr(g, "request_id", None) or uuid.uuid4().hex
        response.headers["X-Request-ID"] = request_id
        if response.status_code >= 400 and response.is_json:
            payload = response.get_json(silent=True)
            if isinstance(payload, dict):
                payload["request_id"] = request_id
                response.set_data(app.json.dumps(payload))
        duration = (time.perf_counter() - getattr(g, "request_started", time.perf_counter())) * 1000
        route = request.url_rule.rule if request.url_rule is not None else "unmatched"
        log = (
            app.logger.debug
            if request.endpoint in {"static", "health_live", "health_ready"}
            else app.logger.info
        )
        log(
            "request id=%s method=%s route=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            route,
            response.status_code,
            max(0, duration),
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.endpoint != "static" or response.status_code >= 400:
            response.headers["Cache-Control"] = "no-store"
        if not app.debug:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'self'"
            )
        if app.config.get("HTTPS", False):
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(ValidationError)
    def invalid_input(error):
        return jsonify(error=str(error), code="invalid_input", field=error.field), 400

    @app.errorhandler(HTTPException)
    def framework_error(error):
        response = error.get_response()
        # Retain protocol headers such as Allow on a 405 response.
        response.set_data(
            app.json.dumps(
                {
                    "error": error.name + ".",
                    "code": error.name.lower().replace(" ", "_"),
                }
            )
        )
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Unhandled request failure")
        return jsonify(error="The request could not be completed.", code="internal_error"), 500
