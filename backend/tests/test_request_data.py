import io
import unittest

from flask import Flask
from werkzeug.datastructures import MultiDict
from werkzeug.exceptions import UnsupportedMediaType

from backend.request_data import read_payload
from backend.validation import ValidationError


class PayloadDecodingTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)

    def test_json_requires_an_object_and_standard_finite_numbers(self):
        for body in [
            "[]",
            "null",
            "false",
            "1",
            '"text"',
            "{",
            '{"value":NaN}',
            '{"value":Infinity}',
        ]:
            with (
                self.subTest(body=body),
                self.app.test_request_context(
                    "/", method="POST", data=body, content_type="application/json"
                ),
            ):
                with self.assertRaises(ValidationError):
                    read_payload()

    def test_duplicate_json_keys_are_rejected_instead_of_last_value_winning(self):
        with self.app.test_request_context(
            "/",
            method="POST",
            data='{"crop":"Wheat","crop":"Rice"}',
            content_type="application/json",
        ):
            with self.assertRaises(ValidationError) as error:
                read_payload()
            self.assertEqual(error.exception.field, "crop")

    def test_form_values_are_scalar_and_files_are_rejected(self):
        with self.app.test_request_context(
            "/", method="POST", data=MultiDict([("crop", "Wheat"), ("crop", "Rice")])
        ):
            with self.assertRaises(ValidationError):
                read_payload(allow_form=True)
        with self.app.test_request_context(
            "/", method="POST", data={"upload": (io.BytesIO(b"content"), "data.csv")}
        ):
            with self.assertRaises(ValidationError):
                read_payload(allow_form=True)

    def test_supported_media_types_and_scalar_round_trip(self):
        for options in [{"json": {"rainfall": 700}}, {"data": {"rainfall": "700"}}]:
            with self.app.test_request_context("/", method="POST", **options):
                self.assertIn(read_payload(allow_form=True)["rainfall"], [700, "700"])
        with self.app.test_request_context(
            "/", method="POST", data="rainfall=700", content_type="text/plain"
        ):
            with self.assertRaises(UnsupportedMediaType):
                read_payload(allow_form=True)
