import unittest

from backend.config import BASE_DIR
from backend.data_store import DataStore


class HistoricalStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DataStore(BASE_DIR / "data" / "real_final_dataset.csv")

    def test_records_are_ordered_and_do_not_mutate_shared_state(self):
        records = self.store.records("Wheat", "Amritsar")
        self.assertEqual([r["Year"] for r in records], list(range(2015, 2025)))
        records[0]["Yield"] = -1
        self.assertGreaterEqual(self.store.records("Wheat", "Amritsar")[0]["Yield"], 0)
        self.assertEqual(self.store.records("Unknown", "Unknown"), [])

    def test_crop_bounds_do_not_leak_mutation(self):
        bounds = self.store.bounds("Wheat")
        bounds["Rainfall"] = (0, 0)
        self.assertGreater(self.store.bounds("Wheat")["Rainfall"][1], 0)
