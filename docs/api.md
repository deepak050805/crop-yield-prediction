# Fieldwise HTTP API

The three HTML pages are `GET /`, `GET /predictor`, and `GET /history`.
All endpoints use the same loaded dataset and model release. The default local
base URL is `http://127.0.0.1:10000`.

## Historical observations

`POST /get_data` requires an `application/json` object with exact, known `crop`
and `district` labels, for example:

```json
{"crop": "Wheat", "district": "Amritsar"}
```

The response is an array ordered by `Year`, with `District`, `Crop`, `Year`,
`Temperature`, `Rainfall`, `Humidity`, and `Yield` fields. These are historical
observations; they are separate from model predictions. A valid combination
without observations returns an empty array. The current bundled data covers
2015–2024. CSV export happens in the historical explorer in the browser.

## Yield estimates

`POST /predict` accepts either JSON or an HTML form (`application/x-www-form-urlencoded`
or `multipart/form-data`, without uploaded files):

```json
{
  "crop": "Wheat",
  "district": "Amritsar",
  "rainfall": 700,
  "temperature": 27,
  "humidity": 70
}
```

Weather values must be finite numbers within these inclusive scenario bounds:

| Field | Bounds | Interface unit |
| --- | --- | --- |
| `rainfall` | 0–2000 | mm |
| `temperature` | −20–60 | °C |
| `humidity` | 0–100 | % |

These broad bounds are request plausibility limits, not validated growing
conditions. The dataset's provenance, rainfall accumulation period, and source
yield unit are undocumented; the interface uses assumed units. See
[data.md](data.md) and [model.md](model.md).

Successful responses retain these fields:

| Field | Meaning |
| --- | --- |
| `predicted` | Model estimate for the submitted conditions, rounded to 3 decimals. |
| `optimal` | Highest estimate among the three tested scenarios; retained legacy name. |
| `gap` | Difference between that best tested estimate and the submitted estimate. |
| `best_rain`, `best_temp`, `best_hum` | Conditions producing the best tested estimate. |
| `crop`, `district` | The submitted selection. |
| `warnings` | Reliability messages, including weather extrapolation or negative estimates. |

The alternatives shift rainfall by ±100, temperature by ±2, and humidity by ±5,
clipped to that crop's observed ranges. The submitted values are preserved.
This comparison is observational and is not an exhaustive optimum or a causal
recommendation. Negative estimates are preserved and warned about; invalid,
non-finite, or malformed model output produces a server error.

## Model report and health

`GET /model_info` returns the report verified with the model at startup. It
includes the model version, feature schema, held-out metrics, crop-mean baseline,
and subgroup comparisons. Replacing a file does not change this process's
report; restart with a complete matching release to load an update.

`GET /health/live` returns `{"status":"alive"}`. `GET /health/ready` returns
`{"status":"ready"}` with HTTP 200 when the dataset and predictor are present,
or `{"status":"not_ready"}` with HTTP 503. Readiness does not run inference.
Missing or corrupt artifacts prevent application startup; inspect server logs.

## Request and failure contract

Bodies are limited to 16 KiB. Duplicate JSON keys, duplicate form fields,
uploaded files, malformed JSON, and JSON `NaN`/`Infinity` values are rejected.
JSON bodies must be objects. Category labels must exactly match known labels.

API failure responses contain an `error` message, a machine-readable `code`, and a generated
`request_id`; validation errors also identify `field` when available. The same
ID appears in the `X-Request-ID` response header. Caller-supplied IDs are not
used as the server's correlation ID.

| HTTP status | Typical code | Trigger |
| --- | --- | --- |
| 400 | `invalid_input` | Missing, duplicate, malformed, or invalid input. |
| 400 | `bad_request` | Request host fails the configured allowlist. |
| 404 | `not_found` | Unknown route or unavailable static file. |
| 405 | `method_not_allowed` | Wrong HTTP method; the `Allow` header is retained. |
| 413 | `request_entity_too_large` | Body exceeds the parsing budget. |
| 415 | `unsupported_media_type` | Unsupported content type. |
| 500 | `prediction_failed` / `internal_error` | Inference or unexpected server failure. |
| 503 | `model_unavailable` | Model report is unavailable in the process. |

The JSON error body does not expose exception details. Request logs record
generated IDs, methods, route patterns, statuses, and duration, excluding request
bodies and query values. Dynamic responses are marked `Cache-Control: no-store`;
static assets retain conditional caching. Deployment security settings and TLS
responsibilities are documented in [deployment.md](deployment.md).
