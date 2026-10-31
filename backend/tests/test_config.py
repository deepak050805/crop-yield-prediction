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
        self.assertEqual(
            load_config({"PORT": "8080", "FIELDWISE_LOG_LEVEL": "warning"})["LOG_LEVEL"], "WARNING"
        )
        for port in ["bad", "0", "-1", "65536"]:
            with self.subTest(port=port), self.assertRaises(ValueError):
                load_config({"PORT": port})
        with self.assertRaises(ValueError):
            load_config({"FIELDWISE_LOG_LEVEL": "verbose"})

    def test_transport_flags_are_explicit_and_disabled_by_default(self):
        for env_name, config_name in [
            ("FIELDWISE_HTTPS", "HTTPS"),
            ("FIELDWISE_TRUST_PROXY", "TRUST_PROXY"),
        ]:
            self.assertFalse(load_config({})[config_name])
            for value, expected in [
                ("true", True),
                ("1", True),
                (" TRUE ", True),
                ("false", False),
                ("0", False),
            ]:
                with self.subTest(flag=env_name, value=value):
                    self.assertIs(load_config({env_name: value})[config_name], expected)
            for value in ["", "yes", "enabled", "2"]:
                with self.subTest(flag=env_name, value=value), self.assertRaises(ValueError):
                    load_config({env_name: value})

    def test_trusted_hosts_require_explicit_nonempty_names(self):
        self.assertIsNone(load_config({})["TRUSTED_HOSTS"])
        configured = load_config(
            {"FIELDWISE_TRUSTED_HOSTS": " example.org, .example.net,127.0.0.1 "}
        )
        self.assertEqual(configured["TRUSTED_HOSTS"], ["example.org", ".example.net", "127.0.0.1"])
        for value in ["", "example.org,", "*", "https://example.org", "two hosts", "."]:
            with self.subTest(hosts=value), self.assertRaises(ValueError):
                load_config({"FIELDWISE_TRUSTED_HOSTS": value})

    def test_request_parsing_budgets_are_bounded(self):
        config = load_config({})
        self.assertEqual(config["MAX_CONTENT_LENGTH"], 16384)
        self.assertEqual(config["MAX_FORM_MEMORY_SIZE"], 16384)
        self.assertEqual(config["MAX_FORM_PARTS"], 16)
