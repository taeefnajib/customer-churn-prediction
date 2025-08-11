from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_curve

from src.models.infer import InferenceService, InferenceArtifacts
from src.utils.io import read_csv
from src.features.preprocess import Preprocessor, PreprocessorArtifacts


def _compute_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    j_max_idx = int(np.argmax(j_scores))
    return float(thresholds[j_max_idx])


def launch_app(cfg, host: str = "0.0.0.0", port: int = 7860) -> None:
    artifacts = InferenceArtifacts(
        model_path=cfg.paths.model_path,
        transformer_path=cfg.paths.transformer_path,
        scaler_path=cfg.paths.scaler_path,
        skewed_cols=list(cfg.features.skewed_cols),
        target=cfg.features.target,
    )
    service = InferenceService(artifacts)

    # Pre-compute Youden's J suggestion from training data if available
    try:
        train_df = read_csv(cfg.paths.train_csv)
        # Remove ID column if it exists (same logic as training)
        if cfg.features.id_column in train_df.columns:
            train_df = train_df.drop(columns=[cfg.features.id_column])
        
        preproc = Preprocessor.load(
            PreprocessorArtifacts(cfg.paths.transformer_path, cfg.paths.scaler_path),
            skewed_cols=list(cfg.features.skewed_cols),
        )
        processed = preproc.transform(train_df)
        y_true = processed[cfg.features.target].values
        y_scores = service.predict_proba(train_df)
        youden_threshold = _compute_youden_threshold(y_true, y_scores)
    except Exception as e:
        logger.warning(f"Could not compute Youden threshold from training set: {e}")
        youden_threshold = float(cfg.ui.default_threshold)

    feature_order = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "subscription_age",
        "bill_avg",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
        "download_over_limit",
    ]

    def predict_fn(
        is_tv_subscriber: int,
        is_movie_package_subscriber: int,
        subscription_age: float,
        bill_avg: float,
        reamining_contract: float,
        service_failure_count: int,
        download_avg: float,
        upload_avg: float,
        download_over_limit: int,
        threshold: float,
        actual_churn: int | None,
    ):
        row = {
            "is_tv_subscriber": is_tv_subscriber,
            "is_movie_package_subscriber": is_movie_package_subscriber,
            "subscription_age": subscription_age,
            "bill_avg": bill_avg,
            "reamining_contract": reamining_contract,
            "service_failure_count": service_failure_count,
            "download_avg": download_avg,
            "upload_avg": upload_avg,
            "download_over_limit": download_over_limit,
        }
        df = pd.DataFrame([row])
        prob = float(service.predict_proba(df)[0])
        pred = int(prob >= threshold)
        correctness = None
        if actual_churn is not None:
            correctness = bool(pred == int(actual_churn))
        return prob, pred, youden_threshold, correctness

    with gr.Blocks(title="Customer Churn Prediction") as demo:
        gr.Markdown("# **Customer Churn Prediction** - Provide features, choose a threshold, and (optionally) actual churn.")
        gr.Markdown("Insert the values for the features and click the 'Predict' button to see the churn probability, predicted churn, Youden's J suggested threshold, and whether the prediction is correct.")

        with gr.Row():
            is_tv = gr.Number(label="is_tv_subscriber (0/1)", value=0)
            is_movie = gr.Number(label="is_movie_package_subscriber (0/1)", value=0)
            sub_age = gr.Number(label="subscription_age", value=12)
            bill = gr.Number(label="bill_avg", value=50)
            remain = gr.Number(label="reamining_contract", value=0)
        with gr.Row():
            svc_fail = gr.Number(label="service_failure_count", value=0)
            dl = gr.Number(label="download_avg", value=30.0)
            ul = gr.Number(label="upload_avg", value=10.0)
            over = gr.Number(label="download_over_limit (0/1)", value=0)
            thr = gr.Slider(0.0, 1.0, value=float(cfg.ui.default_threshold), step=0.01, label="Decision Threshold")

        actual = gr.Number(label="Actual churn (0/1, optional)")

        btn = gr.Button("Predict", variant="primary")
        prob_out = gr.Number(label="Churn probability")
        pred_out = gr.Number(label="Predicted churn (0/1)")
        youden_out = gr.Number(label="Youden's J suggested threshold")
        correct_out = gr.Textbox(label="Prediction correct? (if actual provided)")

        btn.click(
            predict_fn,
            inputs=[is_tv, is_movie, sub_age, bill, remain, svc_fail, dl, ul, over, thr, actual],
            outputs=[prob_out, pred_out, youden_out, correct_out],
        )

        gr.Markdown("""
        ### **Demo Datapoints**

        <table>
            <thead>
                <tr>
                    <th>is_tv_subscriber</th>
                    <th>is_movie_package_subscriber</th>
                    <th>subscription_age</th>
                    <th>bill_avg</th>
                    <th>reamining_contract</th>
                    <th>service_failure_count</th>
                    <th>download_avg</th>
                    <th>upload_avg</th>
                    <th>download_over_limit</th>
                    <th>Actual churn</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td>1</td>
                    <td>1.87</td>
                    <td>10</td>
                    <td>None</td>
                    <td>0</td>
                    <td>0</td>
                    <td>0</td>
                    <td>0</td>
                    <td>1</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>1</td>
                    <td>3.22</td>
                    <td>24</td>
                    <td>0.72</td>
                    <td>0</td>
                    <td>53.8</td>
                    <td>2.4</td>
                    <td>0</td>
                    <td>0</td>
                </tr>
            </tbody>
        </table>
        """)



    demo.queue().launch(server_name=host, server_port=port)


