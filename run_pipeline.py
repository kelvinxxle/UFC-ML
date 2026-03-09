#!/usr/bin/env python3
"""
Single-command UFC pipeline:
1) Scrape fight URLs/results
2) Build profile-aligned dataset
3) Train aligned model
4) Run a prediction
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

from betting_utils import analyze_market
from build_profile_aligned_dataset import UFCStatsProfileDatasetBuilder
from process_ufc_data import (
    build_training_matrix,
    drop_redundant_features,
    select_corner_invariant_features,
    train_model,
    validate_schema_or_raise,
)
from ufc_fight_predictor import UFCFightPredictor
from ufc_profile_schema import PROFILE_NUMERIC_FIELDS, STANCE_VALUES


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


def _fetch_soup(
    session: requests.Session,
    url: str,
    delay_seconds: float = 0.0,
    timeout_seconds: int = 20,
    max_retries: int = 2,
) -> Optional[BeautifulSoup]:
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            return BeautifulSoup(response.content, "html.parser")
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1.0 + attempt)
    return None


def get_event_urls(
    session: requests.Session,
    max_events: int,
    timeout_seconds: int = 20,
    max_retries: int = 2,
) -> List[str]:
    soup = _fetch_soup(
        session,
        EVENTS_URL,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    if soup is None:
        return []
    urls = []
    for a in soup.select("a[href*='/event-details/']"):
        href = a.get("href", "").strip()
        if href and href not in urls:
            urls.append(href)
        if len(urls) >= max_events:
            break
    return urls


def parse_winner_from_event_row(row) -> Optional[str]:
    status_icons = row.select("i.b-fight-details__person-status")
    fighter_links = row.select("td:nth-child(2) a")
    fighter_names = [a.get_text(strip=True) for a in fighter_links if a.get_text(strip=True)]
    if len(status_icons) >= 2 and len(fighter_names) >= 2:
        red_green = any("green" in c.lower() for c in status_icons[0].get("class", []))
        blue_green = any("green" in c.lower() for c in status_icons[1].get("class", []))
        if red_green and not blue_green:
            return fighter_names[0]
        if blue_green and not red_green:
            return fighter_names[1]
    return None


def scrape_fight_index(
    output_csv: str,
    max_events: int = 10,
    request_delay: float = 0.35,
    request_timeout: int = 20,
    request_retries: int = 2,
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)

    event_urls = get_event_urls(
        session,
        max_events=max_events,
        timeout_seconds=request_timeout,
        max_retries=request_retries,
    )
    if not event_urls:
        if Path(output_csv).exists():
            print(
                "[scrape] Could not reach UFCStats event index. Reusing existing raw fights file."
            )
            return pd.read_csv(output_csv)
        raise RuntimeError(
            "Could not reach UFCStats event index and no existing raw fights CSV is available."
        )

    fights: List[Dict[str, str]] = []
    failed_events = 0
    for idx, event_url in enumerate(event_urls, start=1):
        print(f"[scrape] Event {idx}/{len(event_urls)}: {event_url}")
        soup = _fetch_soup(
            session,
            event_url,
            delay_seconds=request_delay,
            timeout_seconds=request_timeout,
            max_retries=request_retries,
        )
        if soup is None:
            failed_events += 1
            print(f"[scrape] Skipping event due to repeated timeout/error: {event_url}")
            continue
        event_tag = soup.select_one("span.b-content__title-highlight")
        event_name = event_tag.get_text(" ", strip=True) if event_tag else ""

        rows = soup.select("tr.b-fight-details__table-row[data-link]")
        for row in rows:
            fight_url = row.get("data-link", "").strip()
            if not fight_url:
                continue

            fighter_links = row.select("td:nth-child(2) a")
            fighter_names = [a.get_text(strip=True) for a in fighter_links if a.get_text(strip=True)]
            if len(fighter_names) < 2:
                continue

            winner = parse_winner_from_event_row(row) or ""
            fights.append(
                {
                    "Red": fighter_names[0],
                    "Blue": fighter_names[1],
                    "Winner": winner,
                    "Fight_URL": fight_url,
                    "Event": event_name,
                }
            )

    df = pd.DataFrame(fights).drop_duplicates(subset=["Fight_URL"]).reset_index(drop=True)
    if df.empty:
        if Path(output_csv).exists():
            print(
                "[scrape] No fights collected this run. Reusing existing raw fights file."
            )
            return pd.read_csv(output_csv)
        raise RuntimeError("Scrape completed but no fights were collected.")
    df.to_csv(output_csv, index=False)
    print(f"[scrape] Saved {len(df)} fights to {output_csv}")
    if failed_events:
        print(f"[scrape] Warning: {failed_events} event pages were skipped due to network timeouts/errors.")
    return df


def train_aligned_model(
    input_csv: str,
    model_out: str,
    features_out: str,
    test_predictions_out: str = "ufc_test_predictions.csv",
    mistakes_out: str = "ufc_test_mistakes.csv",
) -> Dict[str, object]:
    df = pd.read_csv(input_csv)
    print(f"[train] Input shape: {df.shape}")

    validate_schema_or_raise(df)
    X_raw, y, row_meta, drop_stats = build_training_matrix(df)
    X_raw = drop_redundant_features(X_raw)
    X_model = select_corner_invariant_features(X_raw)
    X_model.attrs["row_meta"] = row_meta.reset_index(drop=True)
    print(
        "[train] Rows kept: {kept_rows}, dropped (label mismatch/draw): {dropped_for_label}, "
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
        "[train] Corner-invariant rows: {original_rows}, augmented train rows: {augmented_rows}, "
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
        "source_dataset": input_csv,
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

    print(f"[train] Saved model bundle: {model_out}")
    print(f"[train] Saved features: {features_out}")
    print(f"[train] Saved metrics: {metadata_path}")
    print(f"[train] Saved held-out test predictions: {test_predictions_out}")
    print(f"[train] Saved held-out mistakes: {mistakes_out}")
    return bundle


def load_profile_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_prediction(
    model_path: str,
    features_path: str,
    prediction_out: str,
    red_name: str,
    blue_name: str,
    red_profile_json: Optional[str] = None,
    blue_profile_json: Optional[str] = None,
    red_odds: Optional[str] = None,
    blue_odds: Optional[str] = None,
    bankroll: Optional[float] = None,
    fractional_kelly: float = 0.25,
) -> Dict:
    predictor = UFCFightPredictor(model_path=model_path, features_path=features_path)
    if predictor.model is None:
        raise RuntimeError("Predictor could not load model.")

    if red_profile_json and blue_profile_json:
        red_profile = load_profile_json(red_profile_json)
        blue_profile = load_profile_json(blue_profile_json)
    else:
        example = predictor.get_example_profile_input()
        red_profile = example["red_profile"]
        blue_profile = example["blue_profile"]
        if red_name == "Red Fighter" and blue_name == "Blue Fighter":
            red_name = "Example Red"
            blue_name = "Example Blue"
        print("[predict] No profile JSON provided. Using built-in example profiles.")

    result = predictor.predict_from_ufc_com(
        red_fighter_name=red_name,
        blue_fighter_name=blue_name,
        red_ufc_profile=red_profile,
        blue_ufc_profile=blue_profile,
    )
    if "error" in result:
        raise RuntimeError(result["error"])

    if red_odds and blue_odds:
        result["betting_analysis"] = analyze_market(
            red_name=red_name,
            blue_name=blue_name,
            red_win_probability=float(result["red_win_probability_value"]),
            blue_win_probability=float(result["blue_win_probability_value"]),
            red_odds=red_odds,
            blue_odds=blue_odds,
            bankroll=bankroll,
            fractional_kelly=fractional_kelly,
        )

    with open(prediction_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"[predict] {result['red_fighter']} vs {result['blue_fighter']} -> "
        f"{result['predicted_winner']} ({result['confidence']})"
    )
    print(f"[predict] Saved prediction JSON: {prediction_out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UFC ML end-to-end pipeline")
    parser.add_argument("--raw-fights-csv", default="ufc_fight_data.csv")
    parser.add_argument("--aligned-csv", default="ufc_profile_fights.csv")
    parser.add_argument("--model-out", default="ufc_rf_balanced_smote.pkl")
    parser.add_argument("--features-out", default="ufc_features.csv")
    parser.add_argument("--test-predictions-out", default="ufc_test_predictions.csv")
    parser.add_argument("--mistakes-out", default="ufc_test_mistakes.csv")
    parser.add_argument("--prediction-out", default="pipeline_prediction.json")

    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--max-fights", type=int, default=None)
    parser.add_argument("--scrape-delay", type=float, default=0.35)
    parser.add_argument("--scrape-timeout", type=int, default=20)
    parser.add_argument("--scrape-retries", type=int, default=2)
    parser.add_argument("--profile-delay", type=float, default=0.4)
    parser.add_argument("--profile-timeout", type=int, default=20)
    parser.add_argument("--profile-retries", type=int, default=2)

    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")

    parser.add_argument("--red-name", default="Red Fighter")
    parser.add_argument("--blue-name", default="Blue Fighter")
    parser.add_argument("--red-profile-json", default=None)
    parser.add_argument("--blue-profile-json", default=None)
    parser.add_argument("--red-odds", default=None)
    parser.add_argument("--blue-odds", default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)

    args = parser.parse_args()

    if not args.skip_scrape:
        scrape_fight_index(
            output_csv=args.raw_fights_csv,
            max_events=args.max_events,
            request_delay=args.scrape_delay,
            request_timeout=args.scrape_timeout,
            request_retries=args.scrape_retries,
        )
    else:
        if not Path(args.raw_fights_csv).exists():
            raise FileNotFoundError(
                f"--skip-scrape was set but input file does not exist: {args.raw_fights_csv}"
            )
        print(f"[scrape] Skipped. Using existing: {args.raw_fights_csv}")

    if not args.skip_build:
        builder = UFCStatsProfileDatasetBuilder(
            delay_seconds=args.profile_delay,
            timeout_seconds=args.profile_timeout,
            max_retries=args.profile_retries,
        )
        builder.build(
            input_csv=args.raw_fights_csv,
            output_csv=args.aligned_csv,
            max_fights=args.max_fights,
        )
    else:
        if not Path(args.aligned_csv).exists():
            raise FileNotFoundError(
                f"--skip-build was set but aligned dataset does not exist: {args.aligned_csv}"
            )
        print(f"[build] Skipped. Using existing: {args.aligned_csv}")

    if not args.skip_train:
        train_aligned_model(
            input_csv=args.aligned_csv,
            model_out=args.model_out,
            features_out=args.features_out,
            test_predictions_out=args.test_predictions_out,
            mistakes_out=args.mistakes_out,
        )
    else:
        if not Path(args.model_out).exists() or not Path(args.features_out).exists():
            raise FileNotFoundError(
                "--skip-train was set but model/features files are missing."
            )
        print(f"[train] Skipped. Using existing model: {args.model_out}")

    if not args.skip_predict:
        run_prediction(
            model_path=args.model_out,
            features_path=args.features_out,
            prediction_out=args.prediction_out,
            red_name=args.red_name,
            blue_name=args.blue_name,
            red_profile_json=args.red_profile_json,
            blue_profile_json=args.blue_profile_json,
            red_odds=args.red_odds,
            blue_odds=args.blue_odds,
            bankroll=args.bankroll,
            fractional_kelly=args.fractional_kelly,
        )
    else:
        print("[predict] Skipped.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
