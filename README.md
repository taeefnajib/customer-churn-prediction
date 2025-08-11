# Customer Churn Prediction

This repository contains a modular, production-ready project converted from the original `notebook.ipynb` for customer churn prediction. It includes:

- Structured Python package with training, evaluation, and inference
- Hydra configuration management, Typer CLI, Loguru logging
- MLflow experiment tracking (via Docker, SQLite backend and local artifacts)
- Gradio frontend to test the model with adjustable decision threshold

## Dataset
The dataset used in this project is the [Internet Service Churn](https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn) dataset from Kaggle. It contains customer records from an internet service provider, with the goal of predicting whether a customer will churn (i.e., leave the service).

**Key features include:**
- `is_tv_subscriber`: Whether the customer subscribes to TV services
- `is_movie_package_subscriber`: Whether the customer subscribes to a movie package
- `subscription_age`: Duration (in months) of the customer's subscription
- `bill_avg`: Average monthly bill
- `reamining_contract`: Remaining months on contract (may be 0 or NaN for no contract)
- `service_failure_count`: Number of service failures experienced
- `download_avg` / `upload_avg`: Average download/upload usage
- `download_over_limit`: Whether the customer exceeded their download limit
- `churn`: Target variable (1 = churned, 0 = retained)

The dataset is provided as a CSV file (`data/internet_service_churn.csv`). For model development, it is split into training and test sets, preserving the churn class distribution. Missing values and outliers are handled during preprocessing, and all transformations are applied consistently to both training and test data.

For more details and to download the dataset, visit the [Kaggle page](https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn).


## Project Structure

```
.
├── app/
│   └── gradio_app.py
├── data/
│   └── internet_service_churn.csv  # existing
├── saved/                          # models and transformers will be saved here
├── src/
│   ├── config/
│   │   └── config.yaml
│   ├── features/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── infer.py
│   └── utils/
│       ├── io.py
│       └── logging.py
├── scripts/
│   ├── main.py     # Typer CLI (train, evaluate, serve)
│   ├── test.py     # Evaluate on test set
│   └── predict.py  # CLI for batch/single prediction
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .env.example
├── notebook.ipynb
└── README.md
```

## Quickstart

1) Install dependencies

```
pip install -r requirements.txt
```

2) Run MLflow locally with Docker Compose

```
copy .env.example .env  # Windows
# or: cp .env.example .env

docker compose up -d mlflow
```
Open the UI at http://localhost:5000

3) Train and evaluate

```
python scripts\main.py train
python scripts\main.py evaluate
```

4) Launch the Gradio app (locally)

```
python scripts\main.py serve --host 0.0.0.0 --port 7860
```

Open the UI at http://localhost:7860

## Configuration (Hydra)

All configuration lives in `src/config/config.yaml`. Override values by editing the YAML or by environment variables.

## MLflow

- Service: http://localhost:5000
- Backend store: SQLite at `./mlflow/mlflow.db`
- Artifact root: `./mlflow/artifacts`
- Default experiment: `churn-pred-exp`

## Gradio UI

The Gradio UI provides an interactive web interface for making predictions and exploring the model. It allows you to:

- Enter customer features manually to predict churn probability and class.
- Upload a CSV file for batch predictions.
- Adjust the decision threshold (including a suggested threshold based on Youden's J statistic).
- (Optionally) Provide the actual churn label to compare predictions.


This project uses Docker only to run MLflow locally. The application (training, evaluation, and Gradio UI) runs on your host Python environment.

## Notes

- No GPU required (CPU-only, ~16 GB RAM assumed).
- Artifacts saved under `saved/` and logged to MLflow.


