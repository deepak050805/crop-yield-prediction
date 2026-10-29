# Dataset contract and provenance

`backend/data/real_final_dataset.csv` contains 2,000 observations: ten crops in
twenty Punjab districts over 2015–2024, with 200 records per year. Each
`District`, `Crop`, `Year` combination occurs once. These structural properties
do not establish how the observations were collected or whether they represent
independent real-world measurements.

The repository supplies no original publisher, collection method, sampling
protocol, source URL or dataset license. Yield units and the rainfall
accumulation period are also undocumented. Do not infer an annual rainfall
period, tonnes/hectare, or an official agricultural source from column names.
Confirm these details with the data owner before external redistribution or
operational use.

| Column | Meaning and validation |
| --- | --- |
| District, Crop | Nonempty printable text without surrounding whitespace |
| Year | Integer calendar year from 1 to 9999 |
| Temperature | Finite real number, −20 to 60 °C |
| Rainfall | Finite real number, 0 to 2,000 mm; period unknown |
| Humidity | Finite real number, 0 to 100% |
| Yield | Finite nonnegative real number; unit unknown |

These broad weather bounds constrain requests and data validation. The actual
bundled observations occupy much narrower ranges: temperature 24.7–30.3 °C,
rainfall 638–832 mm, humidity 60–75%, yield 1.3–6.5 in unspecified units. A
request inside the broad bounds can still be outside observed conditions.

## Validate a candidate CSV

From the project root, using the installed backend environment:

```console
python -m backend.ml.validate_data
python -m backend.ml.validate_data path/to/candidate.csv
```

The command prints JSON coverage, per-year counts and observed ranges, exits
with status 1 for invalid or unreadable data, and never rewrites its input.
Validation requires exactly the eight columns above, no missing values,
booleans, complex values, duplicate rows or conflicting observation keys.
Rows are sorted by year in an independent in-memory copy; no observation is
silently dropped, imputed, clipped, deduplicated or repaired.

## Change and release procedure

Preserve the original source CSV. Obtain provenance and units before accepting
a replacement, validate the entire candidate, then rerun temporal evaluation
and examine matched crop-mean baseline comparisons and subgroup sample counts.
Adding history can alter the final two-year holdout and the validation folds;
compare the recorded year partitions before comparing scores across releases.

Retrain into a separate output directory and review the model report, manifest
and pickle together as described in [the model card](model.md). The manifest's
dataset SHA-256 uses `utf8-lf-v1`: CRLF line endings are converted to LF before
hashing. This permits Windows checkouts while still detecting content changes.
Changing only the dataset with an old model release must fail startup integrity
checks; deploy the validated dataset and its complete matched release together.
