#!/usr/bin/env python3
"""
Interactive UI for UFC ML pipeline.

Run with:
    streamlit run ufc_ml_ui.py
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from betting_utils import analyze_market
from build_profile_aligned_dataset import UFCStatsProfileDatasetBuilder
from run_pipeline import scrape_fight_index, train_aligned_model
from ufc_fight_predictor import UFCFightPredictor
from ufc_profile_schema import normalize_key, normalize_profile_input


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def apply_ufc_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Rajdhani:wght@500;700&display=swap');

        .stApp {
            background:
                radial-gradient(1100px 500px at 85% -10%, rgba(255, 0, 0, 0.20), transparent 50%),
                radial-gradient(900px 500px at -10% 10%, rgba(255, 0, 0, 0.12), transparent 50%),
                #090909;
            color: #f2f2f2;
            font-family: 'Rajdhani', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Oswald', sans-serif !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101010 0%, #151515 100%);
            border-right: 1px solid #2a2a2a;
        }

        .stButton > button {
            background: linear-gradient(90deg, #e10600, #b00500);
            color: #ffffff;
            border: 1px solid #ff3b30;
            font-family: 'Oswald', sans-serif;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .stButton > button:hover {
            border-color: #ff6b63;
            background: linear-gradient(90deg, #f0120b, #c00800);
            color: #ffffff;
        }

        .ufc-banner {
            border: 1px solid #2d2d2d;
            background: linear-gradient(135deg, #0f0f0f 0%, #191919 60%, #250707 100%);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .ufc-banner-title {
            font-family: 'Oswald', sans-serif;
            font-size: 1.4rem;
            letter-spacing: 0.05em;
            color: #ffffff;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .ufc-banner-sub {
            color: #cccccc;
            font-size: 0.95rem;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-family: 'Oswald', sans-serif;
        }
        [data-testid="stMetricLabel"] {
            color: #d0d0d0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .stDataFrame, .stTable {
            border: 1px solid #2b2b2b;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1%}"
    except Exception:
        return str(value)


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):,.2f}"
    except Exception:
        return str(value)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_fighter_directory() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)

    rows = []
    for ch in string.ascii_lowercase:
        url = f"http://ufcstats.com/statistics/fighters?char={ch}&page=all"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        for row in soup.select("tr.b-statistics__table-row"):
            tds = row.select("td")
            if len(tds) < 2:
                continue
            first = tds[0].get_text(" ", strip=True)
            last = tds[1].get_text(" ", strip=True)
            if not first and not last:
                continue
            name = " ".join([first, last]).strip()
            link = (
                tds[0].select_one("a[href*='fighter-details']")
                or tds[1].select_one("a[href*='fighter-details']")
                or row.select_one("a[href*='fighter-details']")
            )
            profile_url = link.get("href", "").strip() if link else ""
            if name and profile_url:
                rows.append({"name": name, "profile_url": profile_url})

    fighters = pd.DataFrame(rows).drop_duplicates(subset=["name"]).sort_values("name")
    fighters = fighters.reset_index(drop=True)
    return fighters


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def fetch_fighter_profile(profile_url: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    session = requests.Session()
    session.headers.update(HEADERS)
    response = session.get(profile_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    raw: Dict[str, object] = {}
    record_tag = soup.select_one("span.b-content__title-record")
    if record_tag:
        raw["record"] = record_tag.get_text(" ", strip=True)

    for li in soup.select("li.b-list__box-list-item"):
        text = " ".join(li.stripped_strings)
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        raw[normalize_key(label)] = value.strip()

    normalized = normalize_profile_input(raw)
    return normalized, raw


def load_training_summary_from_bundle(model_path: str) -> Optional[Dict[str, object]]:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    if isinstance(bundle, dict):
        return bundle.get("training_summary") or bundle.get("metrics")
    return None


def show_training_summary(summary: Dict[str, object]) -> None:
    st.subheader("Training Results")
    left, mid, right = st.columns(3)
    left.metric("Test Accuracy", _safe_float(summary.get("test_accuracy")))
    mid.metric("CV Mean Accuracy", _safe_float(summary.get("cv_mean_accuracy")))
    right.metric("Feature Count", str(summary.get("feature_count", "N/A")))

    st.write(
        "Rows: "
        f"train={summary.get('train_rows', 'N/A')} | "
        f"test={summary.get('test_rows', 'N/A')}"
    )

    class_dist = summary.get("class_distribution")
    if class_dist:
        st.write("Class Distribution:", class_dist)

    conf = summary.get("confusion_matrix")
    labels = summary.get("confusion_matrix_labels", [0, 1])
    if conf:
        conf_df = pd.DataFrame(conf, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
        st.write("Confusion Matrix")
        st.dataframe(conf_df, use_container_width=True)

    report = summary.get("classification_report")
    if report:
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report")
        st.dataframe(report_df, use_container_width=True)

    cv_scores = summary.get("cv_scores")
    if cv_scores:
        st.write(f"CV Scores ({summary.get('cv_folds', len(cv_scores))} folds):", cv_scores)


def prettify_profile_df(profile_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "wins": "Career Wins",
        "losses": "Career Losses",
        "draws": "Career Draws",
        "age": "Age",
        "height_in": "Height (inches)",
        "weight_lbs": "Weight (pounds)",
        "reach_in": "Reach (inches)",
        "slpm": "Significant Strikes Landed Per Minute",
        "sapm": "Significant Strikes Absorbed Per Minute",
        "str_acc": "Significant Strike Accuracy",
        "str_def": "Significant Strike Defense",
        "td_avg": "Takedowns Landed Per 15 Minutes",
        "td_acc": "Takedown Accuracy",
        "td_def": "Takedown Defense",
        "sub_avg": "Submission Attempts Per 15 Minutes",
        "stance": "Stance",
    }
    display = profile_df.copy()
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
    return display


def show_betting_analysis(market_analysis: Dict[str, object]) -> None:
    st.subheader("Betting Edge and Kelly Sizing")
    st.write(market_analysis.get("recommendation_summary", "No betting analysis available."))
    st.caption(market_analysis.get("edge_definition", ""))
    st.caption(market_analysis.get("market_hold_definition", ""))

    best_bet = market_analysis.get("best_bet") or {}
    bankroll = market_analysis.get("bankroll")
    left, mid, right, far = st.columns(4)
    left.metric("Market Hold", _format_pct(market_analysis.get("market_overround")))
    mid.metric(
        "Kelly Fraction",
        _format_pct(market_analysis.get("fractional_kelly_multiplier")),
    )
    right.metric("Best Bet", best_bet.get("fighter", "Pass"))
    far.metric("Edge Grade", best_bet.get("edge_grade", "Pass"))

    rows = []
    for side in market_analysis.get("sides", []):
        rows.append(
            {
                "Fighter": side.get("fighter"),
                "American Odds": f"{int(side['american_odds']):+d}",
                "Model Win Probability": _format_pct(side.get("model_win_probability")),
                "Break-Even Probability": _format_pct(side.get("break_even_probability")),
                "Edge vs Market": _format_pct(side.get("edge")),
                "Edge Grade": side.get("edge_grade"),
                "EV per $1 Risked": _format_money(side.get("expected_value_per_dollar")),
                "Full Kelly": _format_pct(side.get("full_kelly_fraction")),
                "Fractional Kelly": _format_pct(side.get("fractional_kelly_fraction")),
                "Recommended Stake": _format_money(side.get("fractional_kelly_stake")),
                "Expected Profit": _format_money(side.get("expected_profit_on_fractional_stake")),
                "Action": side.get("recommendation"),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if best_bet:
        st.caption(best_bet.get("edge_grade_note", ""))

    if bankroll is not None:
        st.caption(
            f"Recommended stakes assume a bankroll of {_format_money(bankroll)}. "
            "Kelly sizing is a bankroll heuristic, not a guarantee."
        )
    else:
        st.caption("Kelly sizing is shown as a bankroll fraction because no bankroll was provided.")


def show_prediction_output(
    result: Dict[str, object],
    red_profile: Dict[str, object],
    blue_profile: Dict[str, object],
    red_raw: Dict[str, object],
    blue_raw: Dict[str, object],
) -> None:
    st.success(
        f"{result['predicted_winner']} predicted with confidence {result['confidence']}"
    )
    a, b, c = st.columns(3)
    a.metric("Red Win Prob", result["red_win_probability"])
    b.metric("Blue Win Prob", result["blue_win_probability"])
    c.metric("Predicted Winner", result["predicted_winner"])

    betting_analysis = result.get("betting_analysis")
    if betting_analysis:
        show_betting_analysis(betting_analysis)

    st.subheader("Why The Model Picked This Fighter")
    st.write(result.get("reasoning_summary", "No explanation available."))
    reasons = result.get("reasoning") or []
    if reasons:
        reasons_df = pd.DataFrame(reasons)
        keep_cols = [
            "label",
            "favors",
            "value",
            "baseline",
            "z_score",
            "importance",
            "signed_impact",
        ]
        keep_cols = [c for c in keep_cols if c in reasons_df.columns]
        st.dataframe(reasons_df[keep_cols], use_container_width=True)
    st.caption(result.get("explainability_note", ""))

    st.write("Prediction JSON")
    st.json(result)

    st.write("Fetched Fighter Profiles (normalized)")
    profile_df = pd.DataFrame(
        [red_profile, blue_profile],
        index=[result["red_fighter"], result["blue_fighter"]],
    )
    st.dataframe(prettify_profile_df(profile_df), use_container_width=True)

    with st.expander("Raw UFCStats profile fields"):
        raw_df = pd.DataFrame(
            [red_raw, blue_raw],
            index=[result["red_fighter"], result["blue_fighter"]],
        )
        st.dataframe(raw_df, use_container_width=True)


def fighter_combobox(label: str, options: list[str], key: str) -> Optional[str]:
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = None

    return st.selectbox(
        label,
        options=options,
        index=None,
        placeholder="Type fighter name...",
        key=key,
    )


def main() -> None:
    st.set_page_config(page_title="UFC ML UI", layout="wide")
    apply_ufc_theme()
    st.markdown(
        """
        <div class="ufc-banner">
            <div class="ufc-banner-title">UFC Fight Predictor Console</div>
            <div class="ufc-banner-sub">Train on scraped fights, inspect model stats, and run fighter-vs-fighter predictions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline Config")
        raw_fights_csv = st.text_input("Raw fights CSV", value="ufc_fight_data.csv")
        aligned_csv = st.text_input("Aligned profile CSV", value="ufc_profile_fights.csv")
        model_out = st.text_input("Model output", value="ufc_rf_balanced_smote.pkl")
        features_out = st.text_input("Features output", value="ufc_features.csv")
        prediction_out = st.text_input("Prediction JSON output", value="ui_prediction.json")

        max_events = st.slider("Number of events to scrape", min_value=1, max_value=80, value=20)
        max_fights = st.number_input(
            "Max fights for build stage (0 = all scraped fights)",
            min_value=0,
            max_value=5000,
            value=300,
            step=10,
        )
        scrape_delay = st.number_input("Scrape delay (seconds)", min_value=0.0, max_value=5.0, value=0.35, step=0.05)
        scrape_timeout = st.number_input("Scrape timeout (seconds)", min_value=5, max_value=60, value=20, step=1)
        scrape_retries = st.number_input("Scrape retries", min_value=1, max_value=5, value=2, step=1)
        profile_delay = st.number_input("Profile delay (seconds)", min_value=0.0, max_value=5.0, value=0.15, step=0.05)
        profile_timeout = st.number_input("Profile timeout (seconds)", min_value=5, max_value=60, value=20, step=1)
        profile_retries = st.number_input("Profile retries", min_value=1, max_value=5, value=2, step=1)

        st.markdown("---")
        raw_exists = Path(raw_fights_csv).exists()
        aligned_exists = Path(aligned_csv).exists()
        trained_exists = Path(model_out).exists() and Path(features_out).exists()
        saved_summary = load_training_summary_from_bundle(model_out) if trained_exists else None
        saved_set_is_outdated = False
        if isinstance(saved_summary, dict) and (
            "validation_accuracy" in saved_summary or saved_summary.get("split_strategy")
        ):
            saved_set_is_outdated = True
        if aligned_exists:
            try:
                aligned_preview = pd.read_csv(aligned_csv, nrows=1)
                if "profile_source" in aligned_preview.columns:
                    saved_set_is_outdated = True
            except Exception:
                pass
        mode_options = [
            "Use current training set",
            "Run new training set",
        ]
        default_mode_index = 0 if trained_exists and not saved_set_is_outdated else 1
        training_mode = st.radio(
            "Training Set Mode",
            options=mode_options,
            index=default_mode_index,
        )
        if saved_set_is_outdated:
            st.warning(
                "Saved dataset/model were built with the experimental branch. "
                "Run a new training set once to rebuild them under the reverted pipeline."
            )

        if training_mode == "Use current training set":
            run_scrape = False
            run_build = False
            run_train = False
            if trained_exists:
                st.caption("Loads the saved model and features already on disk.")
                if saved_set_is_outdated:
                    st.caption("Using the current saved set will keep the broken experimental artifacts.")
            else:
                st.warning("Saved model/features were not found. Switch to 'Run new training set'.")
        else:
            st.caption("Runs scrape/build/train again and overwrites the current saved outputs.")
            run_scrape = st.checkbox("Run scrape stage", value=True)
            run_build = st.checkbox("Run build stage", value=True)
            run_train = st.checkbox("Run train stage", value=True)
            if raw_exists or aligned_exists or trained_exists:
                st.caption("Existing dataset/model files with the same names will be replaced.")

        st.markdown("---")
        st.header("Betting Inputs")
        st.caption("Optional. Enter sportsbook American odds to calculate EV and Kelly sizing.")
        red_odds_input = st.text_input(
            "Red corner American odds",
            value="",
            placeholder="-145",
            key="red_odds_input",
        )
        blue_odds_input = st.text_input(
            "Blue corner American odds",
            value="",
            placeholder="+125",
            key="blue_odds_input",
        )
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=0.0,
            value=100.0,
            step=50.0,
        )
        fractional_kelly = st.slider(
            "Fractional Kelly share",
            min_value=0.10,
            max_value=1.00,
            value=0.25,
            step=0.05,
        )

    st.header("1) Scrape, Build, Train")
    button_label = (
        "Run New Training Pipeline"
        if training_mode == "Run new training set"
        else "Use Saved Training Set"
    )
    if st.button(button_label, type="primary"):
        status = st.status("Running pipeline...", expanded=True)
        try:
            if training_mode == "Use current training set":
                status.write(f"Loading saved model: {model_out}")
                if not Path(model_out).exists():
                    raise FileNotFoundError(f"Model file missing: {model_out}")
                if not Path(features_out).exists():
                    raise FileNotFoundError(f"Features file missing: {features_out}")
                summary = load_training_summary_from_bundle(model_out)
                if summary:
                    st.session_state["training_summary"] = summary
                status.write("Saved training set loaded.")
            else:
                if run_scrape:
                    status.write("Scraping UFCStats events/fights...")
                    scraped_df = scrape_fight_index(
                        output_csv=raw_fights_csv,
                        max_events=int(max_events),
                        request_delay=float(scrape_delay),
                        request_timeout=int(scrape_timeout),
                        request_retries=int(scrape_retries),
                    )
                    status.write(f"Scraped fights: {len(scraped_df)}")
                else:
                    status.write(f"Skipped scrape stage. Using {raw_fights_csv}")
                    if not Path(raw_fights_csv).exists():
                        raise FileNotFoundError(f"Raw fights CSV missing: {raw_fights_csv}")

                if run_build:
                    status.write("Building profile-aligned dataset...")
                    builder = UFCStatsProfileDatasetBuilder(
                        delay_seconds=float(profile_delay),
                        timeout_seconds=int(profile_timeout),
                        max_retries=int(profile_retries),
                    )
                    aligned_df = builder.build(
                        input_csv=raw_fights_csv,
                        output_csv=aligned_csv,
                        max_fights=None if int(max_fights) == 0 else int(max_fights),
                    )
                    status.write(f"Aligned rows: {len(aligned_df)}")
                else:
                    status.write(f"Skipped build stage. Using {aligned_csv}")
                    if not Path(aligned_csv).exists():
                        raise FileNotFoundError(f"Aligned CSV missing: {aligned_csv}")

                if run_train:
                    status.write("Training model...")
                    bundle = train_aligned_model(
                        input_csv=aligned_csv,
                        model_out=model_out,
                        features_out=features_out,
                    )
                    summary = bundle.get("training_summary") or bundle.get("metrics", {})
                    st.session_state["training_summary"] = summary
                    status.write("Training complete.")
                else:
                    status.write(f"Skipped train stage. Using {model_out}")
                    if not Path(model_out).exists():
                        raise FileNotFoundError(f"Model file missing: {model_out}")
                    summary = load_training_summary_from_bundle(model_out)
                    if summary:
                        st.session_state["training_summary"] = summary

            status.update(label="Pipeline finished successfully.", state="complete")
        except Exception as exc:
            status.update(label="Pipeline failed.", state="error")
            st.exception(exc)

    existing_summary = st.session_state.get("training_summary")
    if not existing_summary:
        existing_summary = load_training_summary_from_bundle(model_out)
    if existing_summary:
        show_training_summary(existing_summary)

    st.header("2) Select Fighters and Predict")
    refresh_col, info_col = st.columns([1, 3])
    with refresh_col:
        if st.button("Refresh Fighter List"):
            st.cache_data.clear()
    with info_col:
        st.caption("Fighter list is pulled from UFCStats fighter directory.")

    fighters_df = pd.DataFrame()
    try:
        fighters_df = fetch_fighter_directory()
    except Exception as exc:
        st.error(f"Could not load fighter directory: {exc}")

    if fighters_df.empty:
        st.warning("No fighter list available yet.")
        return

    name_to_url = dict(zip(fighters_df["name"], fighters_df["profile_url"]))
    fighter_names = fighters_df["name"].tolist()

    st.caption("Type in each fighter field, then use arrow keys + Enter to choose.")

    red_name = fighter_combobox(
        label="Red Corner Fighter",
        options=fighter_names,
        key="red_corner_selected",
    )

    blue_pool = [name for name in fighter_names if name != red_name] if red_name else fighter_names
    blue_name = fighter_combobox(
        label="Blue Corner Fighter",
        options=blue_pool,
        key="blue_corner_selected",
    )

    if not red_name or not blue_name:
        st.info("Select one red-corner fighter and one blue-corner fighter to run prediction.")
        return

    if st.button("Run Fight Prediction"):
        try:
            red_profile, red_raw = fetch_fighter_profile(name_to_url[red_name])
            blue_profile, blue_raw = fetch_fighter_profile(name_to_url[blue_name])

            predictor = UFCFightPredictor(model_path=model_out, features_path=features_out)
            if predictor.model is None:
                raise RuntimeError("Model failed to load. Run train stage first.")

            result = predictor.predict_from_ufc_com(
                red_fighter_name=red_name,
                blue_fighter_name=blue_name,
                red_ufc_profile=red_profile,
                blue_ufc_profile=blue_profile,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            st.session_state["last_prediction_state"] = {
                "result": result,
                "red_name": red_name,
                "blue_name": blue_name,
                "red_profile": red_profile,
                "blue_profile": blue_profile,
                "red_raw": red_raw,
                "blue_raw": blue_raw,
            }
        except Exception as exc:
            st.exception(exc)

    prediction_state = st.session_state.get("last_prediction_state")
    if not prediction_state:
        return

    if prediction_state.get("red_name") != red_name or prediction_state.get("blue_name") != blue_name:
        st.info("Fighter selection changed. Run Fight Prediction again to refresh the result.")
        return

    display_result = dict(prediction_state["result"])
    if red_odds_input.strip() and blue_odds_input.strip():
        try:
            display_result["betting_analysis"] = analyze_market(
                red_name=display_result["red_fighter"],
                blue_name=display_result["blue_fighter"],
                red_win_probability=float(display_result["red_win_probability_value"]),
                blue_win_probability=float(display_result["blue_win_probability_value"]),
                red_odds=red_odds_input,
                blue_odds=blue_odds_input,
                bankroll=float(bankroll) if bankroll > 0 else None,
                fractional_kelly=float(fractional_kelly),
            )
        except Exception as exc:
            st.error(f"Could not calculate betting analysis: {exc}")
    else:
        st.caption("Enter both red and blue American odds in the sidebar to add EV and Kelly sizing.")

    Path(prediction_out).write_text(json.dumps(display_result, indent=2), encoding="utf-8")
    show_prediction_output(
        result=display_result,
        red_profile=prediction_state["red_profile"],
        blue_profile=prediction_state["blue_profile"],
        red_raw=prediction_state["red_raw"],
        blue_raw=prediction_state["blue_raw"],
    )


if __name__ == "__main__":
    main()
