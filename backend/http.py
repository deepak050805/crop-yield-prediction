"""Shared JSON failure responses for API and framework errors."""

from flask import jsonify
from werkzeug.exceptions import HTTPException

from backend.validation import ValidationError


def register_http_errors(app):
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
