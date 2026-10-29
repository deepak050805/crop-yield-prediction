# Fieldwise model and evaluation

The bundled model estimates the dataset's reported yield from temperature,
rainfall, humidity, crop and district. Year and yield are excluded from the
inference features. Yield units and the source collection method are undocumented;
this release supports exploratory comparisons, not farm-level accuracy claims.

## Selection and held-out evaluation

Training validates every CSV row before fitting. The development partition contains
2015–2022 (1,600 observations); the untouched evaluation partition contains
2023–2024 (400 observations). Expanding temporal validation predicts 2020, 2021
and 2022 using only earlier years. Scaling and category encoding are refitted
inside every fold. The crop-mean baseline also uses only those earlier records;
an unseen crop receives the training partition's overall mean.

The declared candidates are Ridge regression (three regularization strengths),
Random Forest and Extra Trees (four configurations each, 200 trees per ensemble).
The first declared candidate within one standard error of the best validation
MAE is selected. The final holdout never decides hyperparameters.

The current release selects Ridge with alpha 1.0:

| Comparison | Model MAE | Crop-mean MAE | Model RMSE | Crop-mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Temporal validation | 0.2510 | 0.2515 | — | — |
| Final 2023–2024 holdout | 0.2417 | 0.2425 | 0.2812 | 0.2800 |

The MAE difference is small and RMSE favors the baseline. Overall holdout R² of
0.9566 largely reflects differences among crops; most per-crop R² values are
near zero or negative. Read subgroup sample counts and baseline comparisons in
`backend/models/model_metrics.json`. No calibrated prediction intervals or causal
effects are established by these scores.

The report evaluates the development-only fitted model. After evaluation, the
serving model is refitted on all 2,000 observations, including 2023–2024. The
holdout scores therefore describe the evaluation procedure, not an independent
test of the final refitted pickle.

## Reproduce and verify a release

Use the Python 3.12 runtime and dependency versions in `backend/requirements.txt`:

```console
python -m backend.ml.train --seed 42 --trees 200 --jobs 1 --output /tmp/fieldwise-release
python -m unittest discover -s backend/tests -p "test_ml*.py"
```

Choose a writable output directory for your operating system. Training produces
`yield_model.pkl`, `model_metrics.json` and `model_manifest.json` as one release.
Publish the three files together. The manifest is replaced last; interrupted
publication fails verification until a complete release is restored.

Loading checks the model bytes, normalized dataset/report hashes, ordered feature
schema and declared runtime before deserializing. Text hashes normalize CRLF to LF
so a Windows Git checkout preserves verification. `model_version` is the first
16 hexadecimal characters of the model's SHA-256 hash. Identical inputs and the
locked environment produce deterministic metadata; cross-platform floating-point
implementations may still alter model bytes.

These hashes detect corruption and mixed releases. They do not authenticate a
malicious replacement of both manifest and pickle. Load only trusted, reviewed
local artifacts; never deserialize uploaded models.
