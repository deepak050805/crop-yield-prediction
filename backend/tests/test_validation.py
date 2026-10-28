import unittest

from backend.validation import ValidationError, parse_prediction, parse_selection


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.crops, self.districts = ["Wheat"], ["Amritsar"]
        self.payload = {
            "crop": "Wheat",
            "district": "Amritsar",
            "rainfall": "700",
            "temperature": "27",
            "humidity": "70",
        }

    def parse(self, payload):
        return parse_prediction(payload, self.crops, self.districts)

    def test_each_required_value_is_rejected_when_missing(self):
        for field in self.payload:
            with self.subTest(field=field), self.assertRaises(ValidationError) as error:
                self.parse({key: value for key, value in self.payload.items() if key != field})
            self.assertEqual(error.exception.field, field)

    def test_numeric_boundaries_are_inclusive(self):
        for field, limits in [
            ("rainfall", (0, 2000)),
            ("temperature", (-20, 60)),
            ("humidity", (0, 100)),
        ]:
            for value in limits:
                with self.subTest(field=field, value=value):
                    self.assertEqual(
                        getattr(self.parse({**self.payload, field: value}), field), value
                    )

    def test_nonfinite_boolean_and_compound_values_fail_safely(self):
        for value in [True, False, [], {}, None, "", "nan", "inf", "-inf", "1e999", "bad"]:
            for field in ["rainfall", "temperature", "humidity"]:
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    self.parse({**self.payload, field: value})

    def test_categories_require_exact_known_strings(self):
        for field in ["crop", "district"]:
            for value in [[], {}, True, None, " Wheat", "<script>", "Wheat\n"]:
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    parse_selection({**self.payload, field: value}, self.crops, self.districts)
