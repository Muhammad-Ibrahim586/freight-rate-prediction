# Freight Rate Prediction

A machine learning solution for predicting freight shipment rates using historical load, route, equipment, distance, weight, and date information.

## Project Overview

This project develops and evaluates regression models to estimate freight rates. The solution uses a chronological train-validation split to evaluate performance on later shipment dates and selects the best-performing model for generating the required predictions.

**Selected model:** Extra Trees Regressor

**Validation performance:**

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 150.82 |
| Root Mean Squared Error (RMSE) | 651.87 |
| R² Score | 0.8175 |

## Repository Structure

```text
Spotter-ML-assessment/
├── data/                       # Assessment-provided datasets
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── evaluate.py             # Regression evaluation metrics
│   ├── predict.py              # Validation and December predictions
│   ├── preprocessing.py        # Feature engineering and preprocessing
│   └── train.py                # Model training and selection
├── reports/
│   ├── assessment_report.pdf
│   └── generate_report.py
├── scorer_results/             # Generated December prediction chart
├── requirements.txt
├── requirements-dev.txt
├── score.py
└── README.md
```

Generated model files and prediction outputs are not necessarily committed to the repository. The prediction CSV required for assessment is provided separately.

## Dataset

Place the assessment-provided files in the `data/` directory:

- `train-test.csv` — labeled historical shipment data.
- `validation.csv` — unlabeled loads requiring predictions.
- `validation-predictions-template.csv` — template containing validation load IDs.
- `december-chart-inputs.csv` — December scenario inputs used for the prediction chart.

The datasets are not included in this repository. Use the files supplied with the assessment.

## Methodology

### 1. Exploratory Data Analysis

The training data contains 48,000 shipment records covering January–October 2025.

Key observations included:

- Shipment distance has a strong positive relationship with the posted freight rate.
- Equipment type influences average shipment rates.
- Missing values were present in shipment weight and market index.
- Some shipment weights were negative and were treated as invalid missing values.
- A small number of high-rate outliers were retained because they may represent legitimate shipments.

### 2. Feature Engineering and Preprocessing

The model uses the following features:

- Pickup location
- Delivery location
- Equipment type
- Shipment distance
- Shipment weight
- Year, month, day, day of week, and day of year

Date-derived features are extracted from the shipment date.

Preprocessing includes:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical features.
- One-hot encoding for categorical features.
- Handling of unseen categorical values using `handle_unknown="ignore"`.

Features unavailable in the December input data, including coordinates, market index, and quote signal, are excluded to maintain a consistent feature set.

### 3. Training and Validation Strategy

A chronological split was used to simulate predicting future shipments from historical data.

| Dataset | Date range | Records |
|---|---|---:|
| Training | January–August 2025 | 38,477 |
| Validation | September–October 2025 | 9,523 |

This approach avoids randomly mixing earlier and later shipment records and provides a more realistic assessment of performance on future dates.

### 4. Model Comparison

Two tree-based ensemble regression models were evaluated:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 172.33 | 658.04 | 0.8141 |
| Extra Trees | **150.82** | **651.87** | **0.8175** |

Extra Trees achieved the lowest validation MAE and was selected for final predictions.

## Setup and Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Running the Project

Run these commands from the repository root.

### 1. Train and Evaluate Models

```bash
python -m src.train
```

This trains the Random Forest and Extra Trees models, evaluates them on the chronological validation set, saves the trained models under `outputs/models/`, and records the selected model.

### 2. Generate Predictions

```bash
python -m src.predict
```

This generates:

- `validation_predictions.csv` — predictions for all validation loads, containing `load_id` and `predicted_rate`.
- `december_predictions.csv` — December scenario inputs with their predicted rates.

### 3. Run Output Checks and Generate the December Chart

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The local output-check script checks the prediction files and generates:

```text
scorer_results/candidate_december.png
```

These checks confirm the expected output structure and basic prediction validity; they should not be interpreted as an official assessment score.

### 4. Generate the Assessment Report

```bash
python reports/generate_report.py
```

The generated report is saved as:

```text
reports/assessment_report.pdf
```

## Final Deliverables

- `validation_predictions.csv` — required validation predictions.
- `reports/assessment_report.pdf` — methodology, validation approach, model comparison, and December prediction chart.
- `scorer_results/candidate_december.png` — generated December prediction chart.
- Source code and dependency files — available in this repository.
- Loom presentation — submitted separately.

## Limitations

- The model is trained on historical data from January–October 2025.
- Market index, quote signal, and geographic coordinates are excluded because they are not available in the December scenario inputs.
- Predictions depend on the quality and representativeness of the supplied historical data.
- Validation performance may differ from performance on future shipment data.
