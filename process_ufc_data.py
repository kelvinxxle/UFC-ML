#!/usr/bin/env python3
"""
Train a UFC fight model on UFC.com-compatible fighter profile features.

This replaces the old fighter-pair column approach and enforces one stable
schema for both training and prediction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from prefight_training import (
    detect_prefight_schema,
    run_prefight_training,
    validate_prefight_schema,
)

from ufc_profile_schema import (
    PROFILE_NUMERIC_FIELDS,
    STANCE_VALUES,
    build_feature_dict,
    normalize_key,
)


DEFAULT_INPUT_CANDIDATES = [
    "ufc_prefight_fights.csv",
    "ufc_profile_fights.csv",
    "expanded_real_ufc_data.csv",
    "expanded_ufc_fight_data.csv",
    "ufc_fight_data.csv",
]


NAME_COLUMN_CANDIDATES = {
    "red": ["Red", "red", "red_fighter", "red_name"],
    "blue": ["Blue", "blue", "blue_fighter", "blue_name"],
    "winner": ["Winner", "winner", "result", "winner_name"],
}

NEUTRAL_MODEL_FEATURES = {
    "stance_match",
}


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_input_file(explicit_path: Optional[str], mode: str = "auto") -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    requested_mode = str(mode).strip().lower()
    candidates = list(DEFAULT_INPUT_CANDIDATES)
    if requested_mode == "legacy":
        candidates = [c for c in candidates if c != "ufc_prefight_fights.csv"]
    elif requested_mode == "prefight_v1":
        candidates = ["ufc_prefight_fights.csv"] + [c for c in candidates if c != "ufc_prefight_fights.csv"]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No dataset found. Expected one of: " + ", ".join(DEFAULT_INPUT_CANDIDATES)
    )


def extract_corner_payload(row: pd.Series, corner: str) -> Dict[str, object]:
    normalized_payload: Dict[str, object] = {}
    prefixes = {f"{corner}_", f"{corner.upper()}_", f"{corner.title()}_"}

    for col, val in row.items():
        key = normalize_key(col)
        for prefix in prefixes:
            prefix_key = normalize_key(prefix)
            if key.startswith(prefix_key):
                stripped = key[len(prefix_key) :]
                if stripped:
                    normalized_payload[stripped] = val
                break

    return normalized_payload


def validate_schema_or_raise(df: pd.DataFrame) -> None:
    if detect_prefight_schema(df):
        validate_prefight_schema(df)
        return

    normalized_cols = {normalize_key(c) for c in df.columns}
    has_red_profile = any(col.startswith("red_") for col in normalized_cols)
    has_blue_profile = any(col.startswith("blue_") for col in normalized_cols)

    if not (has_red_profile and has_blue_profile):
        raise ValueError(
            "Dataset does not contain profile-prefixed columns for both corners. "
            "Run build_profile_aligned_dataset.py first."
        )

    required_profile_markers = {
        "red_wins",
        "blue_wins",
        "red_slpm",
        "blue_slpm",
        "red_str_acc",
        "blue_str_acc",
    }
    if not required_profile_markers.issubset(normalized_cols):
        raise ValueError(
            "Dataset is missing aligned profile fields (wins/SLpM/Str Acc for both corners). "
            "Run build_profile_aligned_dataset.py to generate ufc_profile_fights.csv."
        )


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
    # Skip draws/no contests or unexpected labels.
    if w in {"draw", "no contest", "nc"}:
        return None
    return None


def _payload_total_bouts(payload: Dict[str, object]) -> Optional[int]:
    total = 0.0
    found_any = False
    for key in ("wins", "losses", "draws"):
        value = pd.to_numeric(payload.get(key), errors="coerce")
        if pd.notna(value):
            total += float(value)
            found_any = True
    if not found_any:
        return None
    return int(total)


def build_training_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, object]]:
    red_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["red"])
    blue_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["blue"])
    winner_col = find_first_existing_column(df, NAME_COLUMN_CANDIDATES["winner"])

    if not red_col or not blue_col or not winner_col:
        raise ValueError(
            "Missing required columns for labels. Need Red/Blue/Winner (or equivalent)."
        )

    # First pass: collect valid fights and compute fighter-level schedule-strength context.
    # IMPORTANT: schedule-strength is derived from profile stats only (not fight outcomes)
    # to avoid label leakage.
    valid_rows = []
    dropped_for_label = 0
    for _, row in df.iterrows():
        red_name = str(row.get(red_col, "")).strip()
        blue_name = str(row.get(blue_col, "")).strip()
        label = winner_to_label(row.get(winner_col), red_name, blue_name)
        if label is None:
            dropped_for_label += 1
            continue
        red_payload = extract_corner_payload(row, "red")
        blue_payload = extract_corner_payload(row, "blue")
        base_feature_row = build_feature_dict(red_payload, blue_payload, stance_values=STANCE_VALUES)
        valid_rows.append((row, red_name, blue_name, label, base_feature_row))

    if not valid_rows:
        raise ValueError(
            "No usable rows after winner/label filtering. Check fighter names and winner values."
        )

    opponent_quality = defaultdict(list)
    for _, red_name, blue_name, _, base_feature_row in valid_rows:
        red_profile_wr = float(base_feature_row.get("red_shrunk_win_rate", np.nan))
        blue_profile_wr = float(base_feature_row.get("blue_shrunk_win_rate", np.nan))
        if np.isnan(red_profile_wr):
            red_profile_wr = 0.5
        if np.isnan(blue_profile_wr):
            blue_profile_wr = 0.5
        opponent_quality[red_name].append(blue_profile_wr)
        opponent_quality[blue_name].append(red_profile_wr)

    fighter_schedule_strength = {}
    fighter_schedule_std = {}
    for fighter, values in opponent_quality.items():
        if values:
            fighter_schedule_strength[fighter] = float(np.nanmean(values))
            fighter_schedule_std[fighter] = float(np.nanstd(values))
        else:
            fighter_schedule_strength[fighter] = 0.5
            fighter_schedule_std[fighter] = 0.0

    global_schedule_strength = float(
        np.nanmean(list(fighter_schedule_strength.values()))
        if fighter_schedule_strength
        else 0.5
    )
    if np.isnan(global_schedule_strength):
        global_schedule_strength = 0.5

    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    row_metadata: List[Dict[str, object]] = []
    dropped_for_empty = 0

    for source_row, red_name, blue_name, label, base_feature_row in valid_rows:
        feature_row = dict(base_feature_row)
        red_payload = extract_corner_payload(source_row, "red")
        blue_payload = extract_corner_payload(source_row, "blue")

        red_sos = fighter_schedule_strength.get(red_name, global_schedule_strength)
        blue_sos = fighter_schedule_strength.get(blue_name, global_schedule_strength)
        red_sos_std = fighter_schedule_std.get(red_name, 0.0)
        blue_sos_std = fighter_schedule_std.get(blue_name, 0.0)
        feature_row["red_schedule_strength"] = red_sos
        feature_row["blue_schedule_strength"] = blue_sos
        feature_row["delta_schedule_strength"] = red_sos - blue_sos
        feature_row["red_schedule_std"] = red_sos_std
        feature_row["blue_schedule_std"] = blue_sos_std
        feature_row["delta_schedule_std"] = red_sos_std - blue_sos_std

        # Quality-adjusted shrunk rates: penalize weak schedule, reward tough schedule.
        red_scale = red_sos / global_schedule_strength if global_schedule_strength > 0 else 1.0
        blue_scale = blue_sos / global_schedule_strength if global_schedule_strength > 0 else 1.0
        for metric in ("win_rate", "td_acc", "td_def", "str_acc", "str_def", "td_avg", "sub_avg"):
            red_key = f"red_shrunk_{metric}"
            blue_key = f"blue_shrunk_{metric}"
            if red_key in feature_row and blue_key in feature_row:
                red_adj = feature_row[red_key] * red_scale
                blue_adj = feature_row[blue_key] * blue_scale
                feature_row[f"red_adj_{metric}"] = red_adj
                feature_row[f"blue_adj_{metric}"] = blue_adj
                feature_row[f"delta_adj_{metric}"] = red_adj - blue_adj

        numeric_vals = [
            value
            for key, value in feature_row.items()
            if not key.startswith("red_stance_")
            and not key.startswith("blue_stance_")
            and key not in {"stance_match"}
        ]
        if all(pd.isna(v) for v in numeric_vals):
            dropped_for_empty += 1
            continue

        feature_rows.append(feature_row)
        labels.append(label)
        row_metadata.append(
            {
                "red_fighter": red_name,
                "blue_fighter": blue_name,
                "actual_label": label,
                "actual_winner": red_name if label == 0 else blue_name,
                "event": source_row.get("event"),
                "date": source_row.get("date"),
                "fight_url": source_row.get("fight_url"),
                "weight_class": source_row.get("weight_class"),
                "method": source_row.get("method"),
                "round": source_row.get("round"),
                "red_profile_total_bouts": _payload_total_bouts(red_payload),
                "blue_profile_total_bouts": _payload_total_bouts(blue_payload),
            }
        )

    if not feature_rows:
        raise ValueError(
            "No usable training rows after schema alignment. "
            "Verify the dataset includes UFC.com-style profile fields."
        )

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")
    row_meta_df = pd.DataFrame(row_metadata)
    drop_stats = {
        "dropped_for_label": dropped_for_label,
        "dropped_for_empty": dropped_for_empty,
        "kept_rows": len(X),
        "fighter_context": {
            "schedule_strength": fighter_schedule_strength,
            "schedule_std": fighter_schedule_std,
            "default_schedule_strength": global_schedule_strength,
            "default_schedule_std": float(
                np.nanmean(list(fighter_schedule_std.values())) if fighter_schedule_std else 0.0
            ),
        },
    }
    return X, y, row_meta_df, drop_stats


def drop_redundant_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant or high-variance raw rate features when robust alternatives exist.
    """
    redundant_candidates = [
        # Prefer shrunk/adjusted win-rate variants.
        "red_win_rate",
        "blue_win_rate",
        "delta_win_rate",
        # Prefer shrunk rate features over raw profile rates.
        "red_str_acc",
        "blue_str_acc",
        "delta_str_acc",
        "red_str_def",
        "blue_str_def",
        "delta_str_def",
        "red_td_acc",
        "blue_td_acc",
        "delta_td_acc",
        "red_td_def",
        "blue_td_def",
        "delta_td_def",
        "red_td_avg",
        "blue_td_avg",
        "delta_td_avg",
        "red_sub_avg",
        "blue_sub_avg",
        "delta_sub_avg",
        "red_slpm",
        "blue_slpm",
        "delta_slpm",
        "red_sapm",
        "blue_sapm",
        "delta_sapm",
        # Prefer shrunk strike margin / adjusted composites.
        "red_strike_margin",
        "blue_strike_margin",
        "delta_strike_margin",
        "red_wrestling_blend",
        "blue_wrestling_blend",
        "delta_wrestling_blend",
    ]
    to_drop = [c for c in redundant_candidates if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X


def select_corner_invariant_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only comparison features so the model reasons about fighter A vs fighter B,
    not about the historical red/blue corner assignment itself.
    """
    keep_cols = [
        col
        for col in X.columns
        if col.startswith("delta_") or col in NEUTRAL_MODEL_FEATURES
    ]
    if not keep_cols:
        raise ValueError("No corner-invariant comparison features were found.")

    comparison_X = X[keep_cols].copy()
    delta_cols = [col for col in comparison_X.columns if col.startswith("delta_")]
    for col in delta_cols:
        comparison_X[f"abs_{col}"] = comparison_X[col].abs()
    return comparison_X


def augment_with_swapped_corners(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Duplicate each fight with swapped fighter order and inverted label.

    This removes any learnable edge from the arbitrary corner assignment because
    every matchup is seen equally in both orientations.
    """
    augmented_X = X.copy()
    swapped_X = X.copy()

    directional_cols = [col for col in X.columns if col.startswith("delta_")]
    for col in directional_cols:
        swapped_X[col] = -swapped_X[col]

    swapped_y = 1 - y.astype(int)

    X_final = pd.concat([augmented_X, swapped_X], ignore_index=True)
    y_final = pd.concat([y.reset_index(drop=True), swapped_y.reset_index(drop=True)], ignore_index=True)
    return X_final, y_final, {"original_rows": len(X), "augmented_rows": len(X_final)}


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestClassifier, dict, pd.DataFrame]:
    row_meta = pd.DataFrame(index=X.index)
    if "row_meta" in X.attrs:
        row_meta = X.attrs["row_meta"].copy()
    X = X.copy()
    X.attrs = {}

    X_train_raw, X_test_raw, row_meta_train, row_meta_test, y_train, y_test = train_test_split(
        X, row_meta, y, test_size=0.20, random_state=42, stratify=y
    )

    impute_values = X_train_raw.median(numeric_only=True).to_dict()
    impute_values = {k: float(v) for k, v in impute_values.items() if pd.notna(v)}
    X_train = X_train_raw.fillna(impute_values).fillna(0.0)
    X_test = X_test_raw.fillna(impute_values).fillna(0.0)

    zero_var_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
    if zero_var_cols:
        X_train = X_train.drop(columns=zero_var_cols)
        X_test = X_test.drop(columns=zero_var_cols)

    X_train_aug, y_train_aug, augmentation_stats = augment_with_swapped_corners(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_aug, y_train_aug)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        red_probs = y_proba[:, 0]
        blue_probs = y_proba[:, 1] if y_proba.shape[1] > 1 else 1.0 - red_probs
    else:
        red_probs = (y_pred == 0).astype(float)
        blue_probs = (y_pred == 1).astype(float)

    test_acc = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_text = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1], zero_division=0, output_dict=True
    )

    print(f"Test accuracy: {test_acc:.3f}")
    print("Confusion matrix:")
    print(conf)
    print("Classification report:")
    print(report_text)

    minority_count = int(y.value_counts().min()) if not y.empty else 0
    cv_folds = max(2, min(5, minority_count)) if minority_count >= 2 else 0
    cv_scores = None
    cv_mean = None
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, X_train_aug, y_train_aug, cv=cv_folds, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        print(f"{cv_folds}-fold CV mean accuracy: {cv_mean:.3f}")
    else:
        print("Cross-validation skipped (not enough minority-class samples).")

    test_predictions = row_meta_test.reset_index(drop=True).copy()
    test_predictions["actual_label"] = y_test.reset_index(drop=True).astype(int)
    test_predictions["predicted_label"] = pd.Series(y_pred).astype(int)
    test_predictions["actual_winner"] = np.where(
        test_predictions["actual_label"] == 0,
        test_predictions["red_fighter"],
        test_predictions["blue_fighter"],
    )
    test_predictions["predicted_winner"] = np.where(
        test_predictions["predicted_label"] == 0,
        test_predictions["red_fighter"],
        test_predictions["blue_fighter"],
    )
    test_predictions["red_win_probability"] = red_probs
    test_predictions["blue_win_probability"] = blue_probs
    test_predictions["prediction_confidence"] = np.maximum(red_probs, blue_probs)
    test_predictions["was_correct"] = test_predictions["actual_label"] == test_predictions["predicted_label"]
    mistakes_df = (
        test_predictions.loc[~test_predictions["was_correct"]]
        .sort_values(by="prediction_confidence", ascending=False)
        .reset_index(drop=True)
    )

    metadata = {
        "test_accuracy": float(test_acc),
        "cv_mean_accuracy": cv_mean,
        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
        "cv_folds": cv_folds,
        "train_rows": len(X_train_aug),
        "test_rows": len(X_test),
        "feature_count": X_train.shape[1],
        "zero_variance_dropped": len(zero_var_cols),
        "zero_variance_columns": zero_var_cols,
        "class_distribution": {int(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": conf.tolist(),
        "classification_report": report_dict,
        "feature_columns": X_train.columns.tolist(),
        "impute_values": impute_values,
        "original_rows": len(X),
        "augmented_rows": len(X_train_aug),
        "misclassified_count": int((~test_predictions["was_correct"]).sum()),
        "misclassified_rate": float((~test_predictions["was_correct"]).mean()),
        "train_original_rows": len(X_train),
        "test_prediction_rows": len(test_predictions),
    }
    return model, metadata, test_predictions


def run_training_pipeline(
    input_csv: str,
    model_out: str,
    features_out: str,
    test_predictions_out: str,
    mistakes_out: str,
    mode: str = "auto",
) -> Dict[str, object]:
    input_path = choose_input_file(input_csv, mode=mode)
    df = pd.read_csv(input_path)
    requested_mode = str(mode).strip().lower()
    detected_mode = "prefight_v1" if detect_prefight_schema(df) else "legacy"
    if requested_mode in {"legacy", "prefight_v1"}:
        active_mode = requested_mode
    else:
        active_mode = detected_mode

    if active_mode == "prefight_v1":
        if not detect_prefight_schema(df):
            raise ValueError(
                "Requested prefight_v1 training, but the dataset is not a prefight_v1 build."
            )
        return run_prefight_training(
            input_csv=str(input_path),
            model_out=model_out,
            features_out=features_out,
            test_predictions_out=test_predictions_out,
            mistakes_out=mistakes_out,
            df=df,
        )

    validate_schema_or_raise(df)
    print(f"Raw shape: {df.shape}")

    X_raw, y, row_meta, drop_stats = build_training_matrix(df)
    X_raw = drop_redundant_features(X_raw)
    X_model = select_corner_invariant_features(X_raw)
    X_model.attrs["row_meta"] = row_meta.reset_index(drop=True)
    print(
        "Rows kept: {kept_rows}, dropped (label mismatch/draw): {dropped_for_label}, "
        "dropped (empty profile): {dropped_for_empty}".format(**drop_stats)
    )

    model, train_meta, test_predictions = train_model(X_model, y)
    mistakes_df = (
        test_predictions.loc[~test_predictions["was_correct"]]
        .sort_values(by="prediction_confidence", ascending=False)
        .reset_index(drop=True)
    )
    test_predictions.to_csv(test_predictions_out, index=False)
    mistakes_df.to_csv(mistakes_out, index=False)
    print(
        "Corner-invariant rows: {original_rows}, augmented train rows: {augmented_rows}, "
        "held-out mistakes: {misclassified_count}".format(**train_meta)
    )

    final_X = X_model.fillna(train_meta["impute_values"]).fillna(0.0)
    if train_meta["zero_variance_columns"]:
        final_X = final_X.drop(columns=train_meta["zero_variance_columns"])
    final_X = final_X.reindex(columns=train_meta["feature_columns"], fill_value=0.0)
    final_features = final_X.copy()
    final_features["label"] = y.values
    final_features.to_csv(features_out, index=False)

    feature_means = final_X.mean(numeric_only=True).to_dict()
    feature_stds = final_X.std(numeric_only=True).to_dict()
    feature_means = {k: float(v) for k, v in feature_means.items() if pd.notna(v)}
    feature_stds = {k: float(v) for k, v in feature_stds.items() if pd.notna(v)}

    bundle = {
        "model": model,
        "feature_columns": train_meta["feature_columns"],
        "impute_values": train_meta["impute_values"],
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "fighter_context": drop_stats.get("fighter_context", {}),
        "stance_values": STANCE_VALUES,
        "profile_numeric_fields": PROFILE_NUMERIC_FIELDS,
        "schema_version": 3,
        "source_dataset": str(input_path),
        "training_summary": {
            "test_accuracy": train_meta["test_accuracy"],
            "cv_mean_accuracy": train_meta["cv_mean_accuracy"],
            "cv_scores": train_meta.get("cv_scores"),
            "cv_folds": train_meta.get("cv_folds"),
            "train_rows": train_meta["train_rows"],
            "test_rows": train_meta["test_rows"],
            "feature_count": train_meta["feature_count"],
            "class_distribution": train_meta.get("class_distribution"),
            "confusion_matrix_labels": train_meta.get("confusion_matrix_labels"),
            "confusion_matrix": train_meta.get("confusion_matrix"),
            "classification_report": train_meta.get("classification_report"),
            "model_feature_strategy": "corner_invariant_comparison_only",
            "original_rows": train_meta["original_rows"],
            "augmented_rows": train_meta["augmented_rows"],
            "misclassified_count": train_meta["misclassified_count"],
            "misclassified_rate": train_meta["misclassified_rate"],
            "test_predictions_path": test_predictions_out,
            "mistakes_path": mistakes_out,
        },
        "metrics": {
            "test_accuracy": train_meta["test_accuracy"],
            "cv_mean_accuracy": train_meta["cv_mean_accuracy"],
            "train_rows": train_meta["train_rows"],
            "test_rows": train_meta["test_rows"],
            "original_rows": train_meta["original_rows"],
            "augmented_rows": train_meta["augmented_rows"],
            "misclassified_count": train_meta["misclassified_count"],
            "misclassified_rate": train_meta["misclassified_rate"],
        },
    }
    joblib.dump(bundle, model_out)

    metadata_path = Path(model_out).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")

    print(f"Saved model bundle: {model_out}")
    print(f"Saved features: {features_out}")
    print(f"Saved training metrics: {metadata_path}")
    print(f"Saved held-out test predictions: {test_predictions_out}")
    print(f"Saved held-out mistakes: {mistakes_out}")
    print(
        "Schema aligned for UFC.com-style input. Prediction now expects profile fields "
        "instead of fighter-pair-specific columns."
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UFC profile-aligned model")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument("--mode", choices=["auto", "legacy", "prefight_v1"], default="auto")
    parser.add_argument(
        "--model-out", default="ufc_rf_balanced_smote.pkl", help="Output model path"
    )
    parser.add_argument("--features-out", default="ufc_features.csv", help="Output feature CSV")
    parser.add_argument(
        "--test-predictions-out",
        default="ufc_test_predictions.csv",
        help="Output CSV for held-out test predictions",
    )
    parser.add_argument(
        "--mistakes-out",
        default="ufc_test_mistakes.csv",
        help="Output CSV for held-out test mispredictions only",
    )
    args = parser.parse_args()

    model_out = args.model_out
    features_out = args.features_out
    test_predictions_out = args.test_predictions_out
    mistakes_out = args.mistakes_out
    if args.mode == "prefight_v1":
        if model_out == "ufc_rf_balanced_smote.pkl":
            model_out = "ufc_prefight_model.pkl"
        if features_out == "ufc_features.csv":
            features_out = "ufc_prefight_features.csv"
        if test_predictions_out == "ufc_test_predictions.csv":
            test_predictions_out = "ufc_prefight_test_predictions.csv"
        if mistakes_out == "ufc_test_mistakes.csv":
            mistakes_out = "ufc_prefight_test_mistakes.csv"

    input_path = choose_input_file(args.input, mode=args.mode)
    print(f"Loading dataset: {input_path}")
    run_training_pipeline(
        input_csv=str(input_path),
        model_out=model_out,
        features_out=features_out,
        test_predictions_out=test_predictions_out,
        mistakes_out=mistakes_out,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
