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

from ufc_profile_schema import (
    PROFILE_NUMERIC_FIELDS,
    STANCE_VALUES,
    build_feature_dict,
    normalize_key,
)


DEFAULT_INPUT_CANDIDATES = [
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
def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_input_file(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
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


def build_training_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
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
    dropped_for_empty = 0

    for _, red_name, blue_name, label, base_feature_row in valid_rows:
        feature_row = dict(base_feature_row)

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

    if not feature_rows:
        raise ValueError(
            "No usable training rows after schema alignment. "
            "Verify the dataset includes UFC.com-style profile fields."
        )

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")
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
    return X, y, drop_stats


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


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestClassifier, dict]:
    impute_values = X.median(numeric_only=True).to_dict()
    impute_values = {k: float(v) for k, v in impute_values.items() if pd.notna(v)}
    X = X.fillna(impute_values).fillna(0.0)

    zero_var_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        print(f"{cv_folds}-fold CV mean accuracy: {cv_mean:.3f}")
    else:
        print("Cross-validation skipped (not enough minority-class samples).")

    metadata = {
        "test_accuracy": float(test_acc),
        "cv_mean_accuracy": cv_mean,
        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
        "cv_folds": cv_folds,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "zero_variance_dropped": len(zero_var_cols),
        "zero_variance_columns": zero_var_cols,
        "class_distribution": {int(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": conf.tolist(),
        "classification_report": report_dict,
        "feature_columns": X.columns.tolist(),
        "impute_values": impute_values,
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UFC profile-aligned model")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument(
        "--model-out", default="ufc_rf_balanced_smote.pkl", help="Output model path"
    )
    parser.add_argument("--features-out", default="ufc_features.csv", help="Output feature CSV")
    args = parser.parse_args()

    input_path = choose_input_file(args.input)
    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Raw shape: {df.shape}")

    validate_schema_or_raise(df)
    X_raw, y, drop_stats = build_training_matrix(df)
    X_raw = drop_redundant_features(X_raw)
    print(
        "Rows kept: {kept_rows}, dropped (label mismatch/draw): {dropped_for_label}, "
        "dropped (empty profile): {dropped_for_empty}".format(**drop_stats)
    )

    model, train_meta = train_model(X_raw, y)

    # Rebuild the final imputed feature frame for reproducible predictor input contract.
    final_X = X_raw.fillna(train_meta["impute_values"]).fillna(0.0)
    if train_meta["zero_variance_columns"]:
        final_X = final_X.drop(columns=train_meta["zero_variance_columns"])
    final_X = final_X.reindex(columns=train_meta["feature_columns"], fill_value=0.0)
    final_features = final_X.copy()
    final_features["label"] = y.values
    final_features.to_csv(args.features_out, index=False)

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
        "schema_version": 2,
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
        },
        "metrics": {
            "test_accuracy": train_meta["test_accuracy"],
            "cv_mean_accuracy": train_meta["cv_mean_accuracy"],
            "train_rows": train_meta["train_rows"],
            "test_rows": train_meta["test_rows"],
        },
    }
    joblib.dump(bundle, args.model_out)

    metadata_path = Path(args.model_out).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")

    print(f"Saved model bundle: {args.model_out}")
    print(f"Saved features: {args.features_out}")
    print(f"Saved training metrics: {metadata_path}")
    print(
        "Schema aligned for UFC.com-style input. Prediction now expects profile fields "
        "instead of fighter-pair-specific columns."
    )


if __name__ == "__main__":
    main()
