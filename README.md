<div align="center">

<img src="frontend/static/images/logo.png" alt="Crop Yield Prediction logo" width="88">

# Crop Yield Prediction & Weather Analysis

Explore historical crop performance, estimate yield, and compare nearby weather scenarios across Punjab.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Validated_Regression-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)

[Features](#features) · [Quick Start](#quick-start) · [Docker](#run-with-docker) · [Dataset](#dataset) · [API](#api-reference)

</div>

## Overview

A Flask application that brings historical agricultural data and machine learning predictions into a single dashboard. Select a crop and district to explore yield trends, adjust weather inputs, and compare the predicted yield with alternative conditions.

The Fieldwise workspace includes three pages: **Overview**, **Yield Predictor**, and **Historical Explorer**. A trained regression pipeline and a CSV dataset are bundled, so retraining is optional when getting started.

Engineering references: [API contracts](docs/api.md), [local development and checks](docs/development.md), [production deployment and rollback](docs/deployment.md), [dataset provenance and units](docs/data.md), and [model evaluation and release integrity](docs/model.md).

## Features

- **District and crop filters** — explore 20 Punjab districts and 10 crop types.
- **Historical summaries** — view average yield, rainfall, temperature, and humidity for the selected crop and district.
- **Interactive charts** — inspect annual yield trends and rainfall–yield relationships with Chart.js.
- **Historical explorer** — filter years, compare districts, switch between chart and table views, and export filtered records to CSV.
- **Yield prediction** — adjust rainfall, temperature, and humidity to estimate yield in tonnes per hectare.
- **Weather scenario comparison** — compare the selected conditions with two nearby scenarios and display the best predicted result.
- **Prediction insights** — view the yield gap, suggested weather values, and a predicted-versus-best-scenario chart.
- **Transparent evaluation** — inspect held-out error, per-crop results, and a crop-average baseline from the predictor.
- **Docker support** — run the application with Gunicorn in a container.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Web backend | Python, Flask |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn pipelines; Ridge, Random Forest, and Extra Trees model selection |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Model storage | Python pickle |
| Container runtime | Docker, Gunicorn |

## Quick Start

### Prerequisites

- Python 3.12, matching the Docker runtime
- Git and pip
- Internet access to install dependencies; the dashboard loads Google Fonts with local font fallbacks

### 1. Clone the repository

```bash
git clone https://github.com/deepak050805/crop-yield-prediction.git
cd crop-yield-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r backend/requirements.txt "scikit-learn==1.8.0"
```

The bundled model was serialized with scikit-learn 1.8.0. The explicit version above matches that model; the Docker build uses the same version.

### 4. Start the application

```bash
python backend/app.py
```

Open **[http://localhost:10000](http://localhost:10000)**.

The local server uses port `10000` by default. Set the `PORT` environment variable before starting the app to choose another port.

### Development with Live Reload

With Node.js 18+ and the Python dependencies installed, run:

```bash
npm install
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** (or the local URL printed
in the terminal if that port is occupied). The command uses the
project's `.venv` Python when available and starts Flask on local port `5001`.
BrowserSync automatically refreshes changed HTML and JavaScript and injects
CSS updates when you save. Keep this command running while editing; press
`Ctrl+C` to stop both servers. Restart `npm run dev` after changing Python code
or retraining the model. Docker is not required for this development workflow.

## Run with Docker

Start Docker with Linux containers enabled, then run these commands from the repository root:

Create `.env` from the committed example if it does not already exist:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead.
The `.env` file is excluded from Git and the Docker image; pass it at runtime
with `--env-file` when using `docker run`. Docker Compose reads it through
the service's `env_file` setting.

### Docker Compose

```bash
docker compose up --build -d
```

Open **[http://localhost:10000](http://localhost:10000)**.

View logs or stop the service:

```bash
docker compose logs -f web
docker compose down
```

After editing `.env`, run `docker compose up -d` to recreate the service
with the updated environment. If you change the bind port, update the
container port in `compose.yaml` to match.

### Docker CLI

```bash
docker build -t crop-yield-prediction .
docker run --rm --env-file .env -p 10000:10000 --name crop-yield-prediction crop-yield-prediction
```

Open **[http://localhost:10000](http://localhost:10000)**.

The image includes the dataset and trained model. By default, it runs two Gunicorn workers as a non-root user and listens on container port `10000`.
Edit `GUNICORN_CMD_ARGS` in `.env` to configure Gunicorn's bind address, worker
count, and logging. If you change its bind port, update the container side of
the port mapping to match. Restart the container after changing `.env`.

To use a different host port, change the left side of the port mapping:

```bash
docker run --rm --env-file .env -p 8080:10000 --name crop-yield-prediction crop-yield-prediction
```

This makes the dashboard available at `http://localhost:8080`.

To stop the container from another terminal:

```bash
docker stop crop-yield-prediction
```

## Using the Dashboard

1. Choose a **crop**, **district**, and year range on **Overview** to inspect historical averages and charts.
2. Open **Yield Predictor**, adjust the sliders or numeric inputs, or select **Use historical averages**.
3. Select **Predict yield** to view the estimate and best tested scenario. Changed inputs mark the previous result as needing an update.
4. Expand **How reliable is this estimate?** to review evaluation results and limitations.
5. Open **Historical Explorer** to compare districts, inspect the data table, or export CSV records.

| Input | Dashboard range | Unit |
| --- | --- | --- |
| Rainfall | 0–2,000 | mm |
| Temperature | −20–60 | °C |
| Humidity | 0–100 | % |

These control ranges allow scenario exploration; they are not validated growing ranges. The predictor displays local observed ranges and warns when submitted inputs fall outside the crop's historical coverage.

### How the Weather Comparison Works

For each prediction, the backend evaluates three sets of conditions:

| Scenario | Rainfall | Temperature | Humidity |
| --- | --- | --- | --- |
| Selected conditions | User input | User input | User input |
| Higher values | Input + 100 mm | Input + 2 °C | Input + 5 percentage points |
| Lower values | Input − 100 mm | Input − 2 °C | Input − 5 percentage points |

The two alternative scenarios are clipped to the crop's observed weather ranges. The highest prediction becomes the **Best tested scenario**; the API retains the `optimal` field for this value. This is a comparison of three observational model estimates, not an exhaustive optimum or a causal farming recommendation.

## Dataset

The application reads [`backend/data/real_final_dataset.csv`](backend/data/real_final_dataset.csv).

| Property | Coverage |
| --- | --- |
| Records | 2,000 |
| Districts | 20 in Punjab |
| Crops | 10 |
| Years | 2015–2024 |
| Columns | `District`, `Crop`, `Year`, `Temperature`, `Rainfall`, `Humidity`, `Yield` |

**Included crops:** Bajra, Cotton, Gram, Maize, Mustard, Potato, Rice, Sugarcane, Sunflower, and Wheat.

Temperature is displayed in degrees Celsius, rainfall in millimetres, humidity as a percentage, and yield in tonnes per hectare (`t/ha`). Dataset provenance and collection methodology are not documented in the repository.

## Model Training

The training script in [`backend/ml/train.py`](backend/ml/train.py):

1. Removes missing and duplicate records and converts weather and yield columns to numeric values.
2. Reserves the latest two complete years (2023–2024) for final evaluation.
3. Evaluates 11 configurations of Ridge, Random Forest, and Extra Trees using expanding training windows and whole-year validation on 2020, 2021, and 2022. No training fold includes its validation year or future years.
4. Fits scaling and categorical encoding inside each fold's pipeline, and selects by mean absolute error (MAE), preferring simpler candidates within one standard error of the best result.
5. Reports train and holdout MAE, RMSE, and R², plus per-crop holdout scores and a historical crop-average baseline.
6. Refits the selected pipeline on all records for serving and saves it alongside `backend/models/model_metrics.json`.

To retrain from the repository root:

```bash
python backend/ml/train.py
```

Retraining replaces the bundled model and evaluation report. Restart the application afterward to load the new model; rebuild the image when using Docker.

Training and serving both use temperature, rainfall, humidity, crop, and district. `Year` defines evaluation splits and is not a prediction feature.

### Current Evaluation

The selected model is **Ridge regression**. Scores below use training years 2015–2022 and untouched evaluation years 2023–2024, before the serving pipeline is refitted on all records.

| Metric | Training | Held-out years | Crop-average baseline, held-out years |
| --- | --- | --- | --- |
| MAE (t/ha) | 0.2491 | 0.2417 | 0.2425 |
| RMSE (t/ha) | 0.2890 | 0.2812 | 0.2800 |
| R² | 0.9558 | 0.9566 | 0.9570 |

The model's MAE improvement over the baseline is marginal. Overall R² largely reflects differences between crop types; several per-crop R² scores are negative. These results do not establish strong weather-driven prediction within a crop. More informative, verified observations are needed before claiming farm-level accuracy. A small training-to-test error gap does not guarantee freedom from overfitting on other data.

See the full [evaluation report](backend/models/model_metrics.json) for per-crop scores, candidate results, and limitations. Regression R² is not a percentage accuracy score.

### Verification

```bash
python -m unittest discover -s backend/tests -v
```

The API checks cover the three pages, filtering, inference across all crops, input validation, scenario bounds, and evaluation-report consistency.

For loading UI regression checks, install the development dependencies and run:

```bash
npm install
npm run test:ui
```

These checks render the actual Flask templates, so the Python environment and
`backend/requirements.txt` must also be installed. They cover immediate and
pending responses, empty/error states, cancellation, independent comparisons,
image fallbacks, theme tokens, reserved layout dimensions, and reduced motion.

### Loading Components

Reusable Jinja macros in [`frontend/templates/skeletons.html`](frontend/templates/skeletons.html)
provide text, metric cards, images, list items, table rows, and chart placeholders.
Their shared styles live in [`frontend/static/css/skeletons.css`](frontend/static/css/skeletons.css).
Regions use `data-state` (`idle`, `loading`, `success`, `empty`, or `error`) and
`aria-busy`; the shared `FieldwiseLoading.setState(element, state)` helper keeps
both attributes in sync. Skeleton markup is decorative and hidden from assistive
technology. Table and report viewports, image dimensions, and value slots reserve
space while content loads.

There is no artificial display delay. Fast or cached responses may complete
before a skeleton becomes visibly noticeable. Shimmer runs only on active
loading surfaces and is disabled for `prefers-reduced-motion`.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Render the dashboard with crop and district options |
| `GET` | `/predictor` | Render the weather scenario builder and prediction results |
| `GET` | `/history` | Render the historical chart, table, and export workspace |
| `GET` | `/model_info` | Return the saved evaluation report |
| `POST` | `/get_data` | Return historical rows for a crop and district |
| `POST` | `/predict` | Return predicted yield and the best nearby weather scenario |

### Historical data

Send JSON to `/get_data`:

```json
{
  "district": "Amritsar",
  "crop": "Wheat"
}
```

The response is an array of matching dataset records sorted by year. Invalid crop or district values return HTTP `400`; valid combinations without records return an empty array.

### Yield prediction

Send an `application/x-www-form-urlencoded` request to `/predict` with these fields:

| Field | Example |
| --- | --- |
| `rainfall` | `600` |
| `temperature` | `27` |
| `humidity` | `65` |
| `crop` | `Wheat` |
| `district` | `Amritsar` |

A successful JSON response contains:

| Field | Meaning |
| --- | --- |
| `predicted` | Yield predicted for the submitted conditions, in t/ha |
| `optimal` | Highest yield predicted among the three scenarios, in t/ha |
| `gap` | Difference between `optimal` and `predicted`, in t/ha |
| `best_rain` | Rainfall for the best scenario, in mm |
| `best_temp` | Temperature for the best scenario, in °C |
| `best_hum` | Humidity for the best scenario, in % |
| `crop`, `district` | Submitted crop and district |
| `warnings` | Messages for inputs outside the crop's historical weather coverage |

Invalid inputs return an `error` field with HTTP status `400`. Unexpected prediction failures return HTTP `500`.

## Project Structure

```text
crop-yield-prediction/
├── frontend/
│   ├── scripts/
│   │   └── dev.cjs               # Local Flask proxy and browser live reload
│   ├── templates/
│   │   ├── base.html             # Shared navigation and filters
│   │   ├── index.html            # Overview
│   │   ├── predictor.html        # Yield predictor
│   │   ├── history.html          # Historical explorer
│   │   └── skeletons.html        # Shared loading components
│   ├── static/
│   │   ├── css/                  # Dashboard, theme, and skeleton styles
│   │   ├── js/
│   │   │   ├── dashboard.js      # Filters, charts, predictions, and CSV export
│   │   │   ├── dropdowns.js      # Themed dropdown controls
│   │   │   ├── loading.js        # Shared request/image loading states
│   │   │   └── theme.js          # Persistent theme selection
│   │   ├── images/               # Logo, favicon, and plot assets
│   │   └── vendor/chart.js       # Bundled Chart.js library
│   └── tests/loading.test.cjs    # Loading-state and layout regression checks
├── backend/
│   ├── __init__.py
│   ├── app.py                    # Flask routes and prediction logic
│   ├── requirements.txt          # Python dependencies
│   ├── ml/train.py               # Model training and evaluation
│   ├── data/
│   │   └── real_final_dataset.csv
│   ├── models/
│   │   ├── yield_model.pkl       # Bundled trained pipeline
│   │   └── model_metrics.json    # Temporal evaluation report
│   └── tests/test_app.py         # API and model integration checks
├── .env.example
├── .dockerignore
├── compose.yaml
├── Dockerfile
├── procfile                      # Gunicorn process command
├── package.json                  # Root npm commands and development dependencies
├── package-lock.json
└── README.md
```

## Contributing

Open an issue to describe a bug or proposed improvement. For a pull request, keep changes focused, explain the resulting behavior, and include relevant verification steps.
