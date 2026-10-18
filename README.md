<div align="center">

<img src="app/static/logo.png" alt="Crop Yield Prediction logo" width="88">

# Crop Yield Prediction & Weather Analysis

Explore historical crop performance, estimate yield, and compare nearby weather scenarios across Punjab.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)

[Features](#features) · [Quick Start](#quick-start) · [Docker](#run-with-docker) · [Dataset](#dataset) · [API](#api-reference)

</div>

## Overview

A Flask application that brings historical agricultural data and machine learning predictions into a single dashboard. Select a crop and district to explore yield trends, adjust weather inputs, and compare the predicted yield with alternative conditions.

The project includes a trained Random Forest model and a CSV dataset, so retraining is optional when getting started.

## Features

- **District and crop filters** — explore 20 Punjab districts and 10 crop types.
- **Historical summaries** — view average yield, rainfall, temperature, and humidity for the selected crop and district.
- **Interactive charts** — inspect annual yield trends and rainfall–yield relationships with Chart.js.
- **Yield prediction** — adjust rainfall, temperature, and humidity to estimate yield in tonnes per hectare.
- **Weather scenario comparison** — compare the selected conditions with two nearby scenarios and display the best predicted result.
- **Prediction insights** — view the yield gap, suggested weather values, and a predicted-versus-best-scenario chart.
- **Ask AI shortcut** — open an external crop assistant from the dashboard.
- **Docker support** — run the application with Gunicorn in a container.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Web backend | Python, Flask |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, Random Forest Regressor |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Model storage | Python pickle |
| Container runtime | Docker, Gunicorn |

## Quick Start

### Prerequisites

- Python 3.12, matching the Docker runtime
- Git and pip
- Internet access to install dependencies; the dashboard also loads Google Fonts and a chart plugin from external services

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
python -m pip install -r requirements.txt "scikit-learn==1.8.0"
```

The bundled model was serialized with scikit-learn 1.8.0. The explicit version above matches that model; the Docker build uses the same version.

### 4. Start the application

```bash
python app/app.py
```

Open **[http://localhost:10000](http://localhost:10000)**.

The local server uses port `10000` by default. Set the `PORT` environment variable before starting the app to choose another port.

## Run with Docker

Start Docker with Linux containers enabled, then run these commands from the repository root:

Create `.env` from the committed example if it does not already exist:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead.
The `.env` file is excluded from Git and the Docker image; pass it at runtime
with `--env-file`.

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

1. Choose a **crop** and **district** from the header dropdowns.
2. Review the summary cards, annual yield chart, and rainfall scatter plot.
3. Adjust the weather sliders in the **Yield Predictor** panel.
4. Select **Predict Yield** to view the prediction, scenario comparison, and yield gap.

| Input | Dashboard range | Unit |
| --- | --- | --- |
| Rainfall | 300–1,000 | mm |
| Temperature | 15–40 | °C |
| Humidity | 40–90 | % |

### How the Weather Comparison Works

For each prediction, the backend evaluates three sets of conditions:

| Scenario | Rainfall | Temperature | Humidity |
| --- | --- | --- | --- |
| Selected conditions | User input | User input | User input |
| Higher values | Input + 100 mm | Input + 2 °C | Input + 5 percentage points |
| Lower values | Input − 100 mm | Input − 2 °C | Input − 5 percentage points |

The highest prediction becomes the result labeled **optimal** in the dashboard. This is the best of these three scenarios, rather than an exhaustive search for ideal growing conditions. Nearby scenarios can extend beyond the slider ranges.

## Dataset

The application reads [`data/real_final_dataset.csv`](data/real_final_dataset.csv).

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

The training script in [`src/model.py`](src/model.py):

1. Removes missing and duplicate records and converts weather and yield columns to numeric values.
2. Encodes crop and district using one-hot encoding with the first category dropped.
3. Splits the data into 80% training and 20% test sets with `random_state=42`.
4. Tunes a `RandomForestRegressor` with `GridSearchCV`, two-fold cross-validation, and R² scoring.
5. Prints the test-set R² score and saves the selected model to `models/yield_model.pkl`.

To retrain from the repository root:

```bash
python src/model.py
```

Retraining replaces the bundled model file. Restart the application afterward to load the new model.

The training script retains `Year` as an input feature. The current prediction form does not supply a year, and the backend initializes unspecified model features to zero. Prediction quality should be evaluated with this behavior in mind. No measured accuracy score is published in this README.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Render the dashboard with crop and district options |
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

The response is an array of matching dataset records. A combination with no matching records returns an empty array.

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

Prediction exceptions return an `error` field with HTTP status `500`.

## Project Structure

```text
crop-yield-prediction/
├── app/
│   ├── app.py                    # Flask routes and prediction logic
│   ├── static/
│   │   ├── chart.js              # Bundled Chart.js library
│   │   ├── logo.png              # Application logo
│   │   └── yield_plot.png        # Existing plot asset
│   └── templates/
│       └── index.html            # Dashboard, styles, and client-side logic
├── data/
│   └── real_final_dataset.csv    # Historical crop and weather records
├── models/
│   └── yield_model.pkl           # Bundled trained model
├── src/
│   └── model.py                  # Model training and evaluation
├── .dockerignore
├── Dockerfile
├── procfile                      # Gunicorn process command
├── requirements.txt
└── README.md
```

## Contributing

Open an issue to describe a bug or proposed improvement. For a pull request, keep changes focused, explain the resulting behavior, and include relevant verification steps.
