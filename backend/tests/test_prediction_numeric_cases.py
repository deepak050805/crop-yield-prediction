"""Exercise weather parsing with numeric formats and out-of-range inputs."""

import json
import unittest
from pathlib import Path

from backend.validation import ValidationError, parse_prediction


class PredictionNumericCasesTests(unittest.TestCase):
    def test_weather_input_cases(self):
        cases_path = Path(__file__).with_name("prediction_numeric_cases.jsonl")
        payload = {
            "crop": "Wheat",
            "district": "Amritsar",
            "rainfall": 700,
            "temperature": 27,
            "humidity": 70,
        }
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            field = case["field"]
            with self.subTest(case=case):
                data = {**payload, field: case["value"]}
                if case["accepted"]:
                    result = parse_prediction(data, ["Wheat"], ["Amritsar"])
                    self.assertAlmostEqual(getattr(result, field), case["expected"])
                else:
                    with self.assertRaises(ValidationError) as error:
                        parse_prediction(data, ["Wheat"], ["Amritsar"])
                    self.assertEqual(error.exception.field, field)
