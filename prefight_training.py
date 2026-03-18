#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prefight_dataset_builder import PREFIGHT_PRIOR_FIELDS, PREFIGHT_STATIC_FIELDS


NAME_COLUMN_CANDIDATES = {
    "red": ["Red", "red", "red_fighter", "red_name"],
    "blue": ["Blue", "blue", "blue_fighter", "blue_name"],
    "winner": ["Winner", "winner", "result", "winner_name"],
}
PREFIGHT_SCHEMA_FIELDS = [
    "red_ufc_bouts_prior",
    "blue_ufc_bouts_prior",
    "red_elo_prior",
    "blue_elo_prior",
    "date",
    "event",
    "fight_url",
]
PREFIGHT_NUMERIC_FIELDS = [
    "height_in",
    "weight_lbs",
    "reach_in",
    "age_at_fight",
    "ufc_bouts_prior",
    "ufc_wins_prior",
    "ufc_losses_prior",
    "ufc_draws_prior",
    "days_since_last_fight",
    "recent_form_last3",
    "recent_form_last5",
    "sig_landed_per_min_prior",
    "sig_absorbed_per_min_prior",
    "sig_acc_prior",
    "sig_def_prior",
    "td_landed_per15_prior",
    "td_acc_prior",
    "td_def_prior",
    "sub_att_per15_prior",
    "finish_rate_prior",
    "ko_tko_win_rate_prior",
    "submission_win_rate_prior",
    "decision_win_rate_prior",
    "elo_prior",
    "opponent_avg_elo_prior",
]


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def winner_to_label(winner: str, red_name: str, blue_name: str) -> Optional[int]:
    if not winner:
        return None
    w = str(winner).strip().lower()
    red = str(red_name).strip().lower()
    blue = str(blue_name).strip().lower()
    if not red or not blue:
        return None
    if w == red:
        return 0
    if w == blue:
        return 1
    if w in {"draw", "no contest", "nc", "majority draw", "split draw"}:
        return None
    return None


def detect_prefight_schema(df: pd.DataFrame) -> bool:
    normalized = {str(col).strip() for col in df.columns}
    return all(field in normalized for field in PREFIGHT_SCHEMA_FIELDS)


def validate_prefight_schema(df: pd.DataFrame) -> None:
    missing = [field for field in PREFIGHT_SCHEMA_FIELDS if field not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is not prefight_v1. Missing required columns: " + ", ".join(missing)
        )


def build_prefight_training_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, object]]:
    validate_prefight_schema(df)
    red_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["red"])
    blue_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["blue"])
    winner_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["winner"])
    if not red_col or not blue_col or not winner_col:
        raise ValueError("Prefight dataset is missing red/blue/winner columns.")

    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    row_meta: List[Dict[str, object]] = []
    dropped_for_label = 0
    dropped_for_empty = 0
    dropped_for_date = 0

    for _, row in df.iterrows():
        fight_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(fight_date):
            dropped_for_date += 1
            continue

        red_name = str(row.get(red_col, "")).strip()
        blue_name = str(row.get(blue_col, "")).strip()
        label = winner_to_label(row.get(winner_col), red_name, blue_name)
        if label is None:
            dropped_for_label += 1
            continue

        feature_row: Dict[str, float] = {}
        for field in PREFIGHT_NUMERIC_FIELDS:
            red_val = pd.to_numeric(row.get(f"red_{field}"), errors="coerce")
            blue_val = pd.to_numeric(row.get(f"blue_{field}"), errors="coerce")
            delta = red_val - blue_val if pd.notna(red_val) and pd.notna(blue_val) else np.nan
            feature_row[f"delta_{field}"] = delta
            feature_row[f"abs_delta_{field}"] = abs(delta) if pd.notna(delta) else np.nan

        red_stance = str(row.get("red_stance", "")).strip().lower()
        blue_stance = str(row.get("blue_stance", "")).strip().lower()
        feature_row["stance_match"] = float(bool(red_stance and red_stance == blue_stance))

        numeric_values = [value for value in feature_row.values() if pd.notna(value)]
        if not numeric_values:
            dropped_for_empty += 1
            continue

        feature_rows.append(feature_row)
        labels.append(label)
        row_meta.append(
            {
                "fight_url": row.get("fight_url"),
                "event": row.get("event"),
                "date": pd.Timestamp(fight_date).normalize(),
                "weight_class": row.get("weight_class"),
                "method": row.get("method"),
                "round": row.get("round"),
                "time": row.get("time"),
                "red_fighter": red_name,
                "blue_fighter": blue_name,
                "actual_label": label,
                "actual_winner": red_name if label == 0 else blue_name,
                "red_ufc_bouts_prior": pd.to_numeric(row.get("red_ufc_bouts_prior"), errors="coerce"),
                "blue_ufc_bouts_prior": pd.to_numeric(row.get("blue_ufc_bouts_prior"), errors="coerce"),
            }
        )

    if not feature_rows:
        raise ValueError("No usable prefight_v1 rows remained after filtering.")

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")
    row_meta_df = pd.DataFrame(row_meta)
    return X, y, row_meta_df, {
        "dropped_for_label": dropped_for_label,
        "dropped_for_empty": dropped_for_empty,
        "dropped_for_date": dropped_for_date,
        "kept_rows": len(X),
        "feature_strategy": "prefight_v1_corner_invariant_deltas",
    }

def _allocate_event_counts(event_count: int) -> Tuple[int, int, int]:
    if event_count < 3:
        raise ValueError("prefight_v1 training requires at least 3 distinct dated events.")
    train_count = max(1, int(round(event_count * 0.70)))
    test_count = max(1, int(round(event_count * 0.15)))
    val_count = event_count - train_count - test_count
    while val_count < 1 and train_count > 1:
        train_count -= 1
        val_count = event_count - train_count - test_count
    while val_count < 1 and test_count > 1:
        test_count -= 1
        val_count = event_count - train_count - test_count
    if val_count < 1:
        raise ValueError("Could not allocate non-empty train/validation/test event splits.")
    return train_count, val_count, test_count


def chronological_event_split(row_meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    event_dates = (
        row_meta[["event", "date"]]
        .dropna(subset=["event", "date"])
        .groupby("event", as_index=False)["date"]
        .min()
        .sort_values(["date", "event"])
        .reset_index(drop=True)
    )
    if event_dates.empty:
        raise ValueError("No dated events available for prefight_v1 chronological splitting.")

    train_count, val_count, test_count = _allocate_event_counts(len(event_dates))
    train_events = set(event_dates.iloc[:train_count]["event"])
    val_events = set(event_dates.iloc[train_count : train_count + val_count]["event"])
    test_events = set(event_dates.iloc[train_count + val_count :]["event"])

    train_mask = row_meta["event"].isin(train_events).to_numpy()
    val_mask = row_meta["event"].isin(val_events).to_numpy()
    test_mask = row_meta["event"].isin(test_events).to_numpy()

    split_meta = {
        "split_strategy": "grouped_chronological_events_70_15_15",
        "train_event_count": len(train_events),
        "validation_event_count": len(val_events),
        "test_event_count": len(test_events),
        "train_events": sorted(train_events),
        "validation_events": sorted(val_events),
        "test_events": sorted(test_events),
        "train_date_range": [
            event_dates.iloc[:train_count]["date"].min().strftime("%Y-%m-%d"),
            event_dates.iloc[:train_count]["date"].max().strftime("%Y-%m-%d"),
        ],
        "validation_date_range": [
            event_dates.iloc[train_count : train_count + val_count]["date"].min().strftime("%Y-%m-%d"),
            event_dates.iloc[train_count : train_count + val_count]["date"].max().strftime("%Y-%m-%d"),
        ],
        "test_date_range": [
            event_dates.iloc[train_count + val_count :]["date"].min().strftime("%Y-%m-%d"),
            event_dates.iloc[train_count + val_count :]["date"].max().strftime("%Y-%m-%d"),
        ],
    }
    return train_mask, val_mask, test_mask, split_meta


def _fit_preprocessor(X_train: pd.DataFrame) -> Tuple[Dict[str, float], List[str]]:
    impute_values = X_train.median(numeric_only=True).to_dict()
    impute_values = {k: float(v) for k, v in impute_values.items() if pd.notna(v)}
    filled = X_train.fillna(impute_values).fillna(0.0)
    zero_var_cols = [col for col in filled.columns if filled[col].nunique(dropna=False) <= 1]
    return impute_values, zero_var_cols


def _apply_preprocessor(
    X: pd.DataFrame,
    impute_values: Dict[str, float],
    zero_var_cols: List[str],
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = X.fillna(impute_values).fillna(0.0)
    if zero_var_cols:
        out = out.drop(columns=zero_var_cols, errors="ignore")
    if feature_columns is not None:
        out = out.reindex(columns=feature_columns, fill_value=0.0)
    return out


def _candidate_models() -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
        ),
    }


def _evaluate_model(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "log_loss": float(log_loss(y_eval, y_proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_eval, y_proba[:, 1])),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics, y_pred, y_proba


def _select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[str, object, Dict[str, Dict[str, object]]]:
    candidate_results: Dict[str, Dict[str, object]] = {}
    best_name = ""
    best_model = None
    best_accuracy = -1.0
    best_log_loss = float("inf")

    for name, model in _candidate_models().items():
        try:
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            metrics, _, _ = _evaluate_model(fitted, X_val, y_val)
            candidate_results[name] = metrics
            if metrics["accuracy"] > best_accuracy or (
                metrics["accuracy"] == best_accuracy and metrics["log_loss"] < best_log_loss
            ):
                best_name = name
                best_model = fitted
                best_accuracy = metrics["accuracy"]
                best_log_loss = metrics["log_loss"]
        except Exception as exc:
            candidate_results[name] = {"error": str(exc)}

    if best_model is None:
        raise ValueError("No candidate model could be selected for prefight_v1 training.")
    return best_name, best_model, candidate_results


def _build_test_predictions(
    row_meta: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    preds = row_meta.reset_index(drop=True).copy()
    preds["actual_label"] = y_true.reset_index(drop=True).astype(int)
    preds["predicted_label"] = pd.Series(y_pred).astype(int)
    preds["actual_winner"] = np.where(preds["actual_label"] == 0, preds["red_fighter"], preds["blue_fighter"])
    preds["predicted_winner"] = np.where(preds["predicted_label"] == 0, preds["red_fighter"], preds["blue_fighter"])
    preds["red_win_probability"] = y_proba[:, 0]
    preds["blue_win_probability"] = y_proba[:, 1]
    preds["prediction_confidence"] = np.maximum(y_proba[:, 0], y_proba[:, 1])
    preds["was_correct"] = preds["actual_label"] == preds["predicted_label"]
    return preds

def run_prefight_training(
    input_csv: str,
    model_out: str,
    features_out: str,
    test_predictions_out: str,
    mistakes_out: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    dataset = pd.read_csv(input_csv) if df is None else df.copy()
    validate_prefight_schema(dataset)
    X_raw, y, row_meta, drop_stats = build_prefight_training_matrix(dataset)
    train_mask, val_mask, test_mask, split_meta = chronological_event_split(row_meta)

    X_train_raw = X_raw.loc[train_mask].reset_index(drop=True)
    X_val_raw = X_raw.loc[val_mask].reset_index(drop=True)
    X_test_raw = X_raw.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    row_meta_train = row_meta.loc[train_mask].reset_index(drop=True)
    row_meta_val = row_meta.loc[val_mask].reset_index(drop=True)
    row_meta_test = row_meta.loc[test_mask].reset_index(drop=True)

    if y_train.nunique() < 2:
        raise ValueError("Chronological prefight train split contains only one class. Use more dated events.")
    if y_val.nunique() < 2:
        raise ValueError("Chronological prefight validation split contains only one class. Use more dated events.")

    impute_values, zero_var_cols = _fit_preprocessor(X_train_raw)
    X_train = _apply_preprocessor(X_train_raw, impute_values, zero_var_cols)
    X_val = _apply_preprocessor(X_val_raw, impute_values, zero_var_cols, X_train.columns.tolist())
    X_test = _apply_preprocessor(X_test_raw, impute_values, zero_var_cols, X_train.columns.tolist())
    feature_columns = X_train.columns.tolist()
    if not feature_columns:
        raise ValueError("No model features remained after prefight preprocessing.")

    selected_name, _, candidate_metrics = _select_model(X_train, y_train, X_val, y_val)
    final_model = clone(_candidate_models()[selected_name])
    X_train_val_raw = pd.concat([X_train_raw, X_val_raw], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    row_meta_train_val = pd.concat([row_meta_train, row_meta_val], ignore_index=True)
    final_impute_values, final_zero_var_cols = _fit_preprocessor(X_train_val_raw)
    X_train_val = _apply_preprocessor(X_train_val_raw, final_impute_values, final_zero_var_cols)
    final_feature_columns = X_train_val.columns.tolist()
    X_test_final = _apply_preprocessor(X_test_raw, final_impute_values, final_zero_var_cols, final_feature_columns)
    final_model.fit(X_train_val, y_train_val)

    test_metrics, y_test_pred, y_test_proba = _evaluate_model(final_model, X_test_final, y_test)
    test_predictions = _build_test_predictions(row_meta_test, y_test, y_test_pred, y_test_proba)
    mistakes_df = (
        test_predictions.loc[~test_predictions["was_correct"]]
        .sort_values(by="prediction_confidence", ascending=False)
        .reset_index(drop=True)
    )
    test_predictions.to_csv(test_predictions_out, index=False)
    mistakes_df.to_csv(mistakes_out, index=False)

    final_features = X_train_val.copy()
    final_features["label"] = y_train_val.values
    final_features.to_csv(features_out, index=False)

    feature_means = {k: float(v) for k, v in X_train_val.mean(numeric_only=True).to_dict().items() if pd.notna(v)}
    feature_stds = {k: float(v) for k, v in X_train_val.std(numeric_only=True).to_dict().items() if pd.notna(v)}

    training_summary = {
        "test_accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "log_loss": test_metrics["log_loss"],
        "brier_score": test_metrics["brier_score"],
        "feature_count": len(final_feature_columns),
        "train_rows": len(X_train_raw),
        "validation_rows": len(X_val_raw),
        "test_rows": len(X_test_raw),
        "misclassified_count": int((~test_predictions["was_correct"]).sum()),
        "misclassified_rate": float((~test_predictions["was_correct"]).mean()),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "model_family_selected": selected_name,
        "validation_candidates": candidate_metrics,
        "split_strategy": split_meta["split_strategy"],
        "train_event_count": split_meta["train_event_count"],
        "validation_event_count": split_meta["validation_event_count"],
        "test_event_count": split_meta["test_event_count"],
        "train_date_range": split_meta["train_date_range"],
        "validation_date_range": split_meta["validation_date_range"],
        "test_date_range": split_meta["test_date_range"],
        "test_predictions_path": test_predictions_out,
        "mistakes_path": mistakes_out,
        "drop_stats": drop_stats,
        "validation_accuracy": candidate_metrics[selected_name]["accuracy"],
        "validation_log_loss": candidate_metrics[selected_name]["log_loss"],
        "rolling_validation_mean_accuracy": None,
    }
    metrics = {
        "test_accuracy": training_summary["test_accuracy"],
        "balanced_accuracy": training_summary["balanced_accuracy"],
        "log_loss": training_summary["log_loss"],
        "brier_score": training_summary["brier_score"],
        "train_rows": training_summary["train_rows"],
        "validation_rows": training_summary["validation_rows"],
        "test_rows": training_summary["test_rows"],
        "misclassified_count": training_summary["misclassified_count"],
        "misclassified_rate": training_summary["misclassified_rate"],
        "model_family_selected": selected_name,
        "split_strategy": split_meta["split_strategy"],
    }
    bundle = {
        "model": final_model,
        "feature_columns": final_feature_columns,
        "impute_values": final_impute_values,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "schema_version": 4,
        "dataset_schema": "prefight_v1",
        "source_dataset": input_csv,
        "prefight_static_fields": PREFIGHT_STATIC_FIELDS,
        "prefight_prior_fields": PREFIGHT_PRIOR_FIELDS,
        "selected_model_family": selected_name,
        "split_metadata": split_meta,
        "training_summary": training_summary,
        "metrics": metrics,
    }
    joblib.dump(bundle, model_out)
    metadata_path = Path(model_out).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[train] Prefight rows kept: {drop_stats['kept_rows']}, dropped (label): {drop_stats['dropped_for_label']}, dropped (date): {drop_stats['dropped_for_date']}, dropped (empty): {drop_stats['dropped_for_empty']}")
    print(f"[train] Selected model: {selected_name}")
    print(f"[train] Validation accuracy/log loss: {candidate_metrics[selected_name]['accuracy']:.3f} / {candidate_metrics[selected_name]['log_loss']:.3f}")
    print(f"[train] Test accuracy/balanced/logloss/brier: {test_metrics['accuracy']:.3f} / {test_metrics['balanced_accuracy']:.3f} / {test_metrics['log_loss']:.3f} / {test_metrics['brier_score']:.3f}")
    print(f"[train] Saved prefight model bundle: {model_out}")
    print(f"[train] Saved prefight features: {features_out}")
    print(f"[train] Saved prefight metrics: {metadata_path}")
    print(f"[train] Saved held-out test predictions: {test_predictions_out}")
    print(f"[train] Saved held-out mistakes: {mistakes_out}")
    return bundle
