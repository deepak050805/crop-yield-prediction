import unittest

from backend.config import load_config


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_production_safe_and_paths_are_absolute(self):
        config = load_config({})
        self.assertFalse(config["DEBUG"])
        self.assertFalse(config["TESTING"])
        self.assertEqual(config["PORT"], 10000)
        self.assertTrue(config["DATA_PATH"].is_absolute())

    def test_port_and_logging_configuration(self):
        self.assertEqual(load_config({"PORT": "8080", "FIELDWISE_LOG_LEVEL": "warning"})["LOG_LEVEL"], "WARNING")
        for port in ["bad", "0", "-1", "65536"]:
            with self.subTest(port=port), self.assertRaises(ValueError):
                load_config({"PORT": port})
        with self.assertRaises(ValueError):
            load_config({"FIELDWISE_LOG_LEVEL": "verbose"})
