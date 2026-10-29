"""Validate a source CSV without training or modifying the dataset."""

import argparse
import json
from pathlib import Path

try:
    from .dataset import load_dataset
    from .schema import NUMERIC, SCHEMA_VERSION
except ImportError:  # Support direct execution beside train.py.
    from dataset import load_dataset
    from schema import NUMERIC, SCHEMA_VERSION


def dataset_summary(path: str | Path) -> dict:
    """Report observed coverage only after every source row passes validation."""
    data = load_dataset(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(data),
        "years": sorted(int(year) for year in data["Year"].unique()),
        "crops": sorted(data["Crop"].unique().tolist()),
        "districts": sorted(data["District"].unique().tolist()),
        "records_per_year": {
            str(year): int(count) for year, count in data.groupby("Year").size().items()
        },
        "observed_ranges": {
            column: {"min": float(data[column].min()), "max": float(data[column].max())}
            for column in [*NUMERIC, "Yield"]
        },
        "yield_unit": "undocumented",
        "rainfall_accumulation_period": "undocumented",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/real_final_dataset.csv",
        help="UTF-8 source CSV; defaults to the bundled dataset.",
    )
    args = parser.parse_args(argv)
    try:
        report = dataset_summary(args.data)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
