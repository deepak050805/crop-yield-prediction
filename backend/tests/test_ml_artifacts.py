import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.ml.artifacts import (
    ArtifactValidationError,
    load_model_bundle,
    publish_bundle,
    text_sha256,
)
from backend.ml.schema import prepare_input

BASE = Path(__file__).resolve().parents[1]


class ArtifactBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Repository artifacts are trusted test fixtures, never uploaded input.
        cls.model = pickle.loads((BASE / "models/yield_model.pkl").read_bytes())
        cls.report = json.loads((BASE / "models/model_metrics.json").read_text())
        cls.dataset = BASE / "data/real_final_dataset.csv"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dataset_copy = self.directory / "observations.csv"
        self.dataset_copy.write_bytes(self.dataset.read_bytes())
        publish_bundle(self.model, self.report, self.dataset_copy, self.directory)

    def load(self):
        return load_model_bundle(
            self.directory / "yield_model.pkl",
            self.directory / "model_metrics.json",
            self.dataset_copy,
        )

    def edit_manifest(self, callback):
        path = self.directory / "model_manifest.json"
        manifest = json.loads(path.read_text())
        callback(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def assert_rejected_before_pickle(self):
        with patch("backend.ml.artifacts.pickle.loads") as deserialize:
            with self.assertRaises(ArtifactValidationError):
                self.load()
            deserialize.assert_not_called()

    def test_trusted_release_round_trip_preserves_predictions(self):
        model, report = self.load()
        features = prepare_input(27.5, 735, 67, "Wheat", "Amritsar")
        np.testing.assert_array_equal(model.predict(features), self.model.predict(features))
        self.assertEqual(
            report["model_version"],
            json.loads((self.directory / "model_manifest.json").read_text())["model_version"],
        )

    def test_modified_model_report_or_dataset_is_rejected_before_deserialization(self):
        for name in ["yield_model.pkl", "model_metrics.json", "observations.csv"]:
            with self.subTest(name=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.assert_rejected_before_pickle()
                path.write_bytes(original)

    def test_feature_order_mismatch_is_rejected_before_deserialization(self):
        self.edit_manifest(lambda manifest: manifest["feature_schema"]["features"].reverse())
        self.assert_rejected_before_pickle()

    def test_runtime_version_mismatch_is_rejected_before_deserialization(self):
        self.edit_manifest(lambda manifest: manifest["runtime_versions"].update(sklearn="0.0"))
        self.assert_rejected_before_pickle()

    def test_missing_manifest_and_invalid_metadata_fail_closed(self):
        path = self.directory / "model_manifest.json"
        path.unlink()
        self.assert_rejected_before_pickle()
        for content in ["not json", "[]", '{"format_version":999}']:
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                self.assert_rejected_before_pickle()

    def test_publication_is_byte_reproducible_for_identical_inputs(self):
        names = ["yield_model.pkl", "model_metrics.json", "model_manifest.json"]
        original = {name: (self.directory / name).read_bytes() for name in names}
        publish_bundle(self.model, self.report, self.dataset_copy, self.directory)
        self.assertEqual(original, {name: (self.directory / name).read_bytes() for name in names})

    def test_windows_text_checkouts_preserve_release_integrity(self):
        original_model, original_report = self.load()
        for path in [self.dataset_copy, self.directory / "model_metrics.json"]:
            normalized = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(normalized.replace(b"\n", b"\r\n"))
        model, report = self.load()
        self.assertEqual(report, original_report)
        features = prepare_input(27.5, 735, 67, "Wheat", "Amritsar")
        np.testing.assert_array_equal(model.predict(features), original_model.predict(features))

    def test_nonfinite_and_duplicate_metadata_are_rejected_even_with_matching_hashes(self):
        manifest_path = self.directory / "model_manifest.json"
        original_manifest = manifest_path.read_bytes()
        for name in ["model_manifest.json", "model_metrics.json"]:
            path = self.directory / name
            original = path.read_bytes()
            for suffix in ['"extra":NaN', '"extra":Infinity', '"extra":1,"extra":2']:
                with self.subTest(name=name, suffix=suffix):
                    changed = original.rstrip()[:-1] + b"," + suffix.encode() + b"}"
                    path.write_bytes(changed)
                    if name == "model_metrics.json":
                        self.edit_manifest(
                            lambda manifest: manifest.update(metrics_sha256=text_sha256(changed))
                        )
                    self.assert_rejected_before_pickle()
                    path.write_bytes(original)
                    manifest_path.write_bytes(original_manifest)
