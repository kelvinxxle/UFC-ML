#!/usr/bin/env python3
"""
Shared UFC profile schema and feature engineering utilities.

This module defines a train/inference-compatible feature schema that matches
the kind of stats exposed on UFC fighter profile pages.
"""

from __future__ import annotations

from datetime import date, datetime
import math
import re
from typing import Dict, Iterable, Optional

import numpy as np


# Population priors used for empirical-Bayes shrinkage.
# This reduces inflation for fighters with low sample sizes.
SHRINKAGE_PRIORS = {
    "win_rate": 0.50,
    "str_acc": 0.45,
    "str_def": 0.55,
    "td_acc": 0.38,
    "td_def": 0.60,
    "td_avg": 1.10,
    "sub_avg": 0.35,
    "slpm": 4.00,
    "sapm": 3.90,
}

# Pseudo-count controlling how aggressively we shrink stats for low-experience fighters.
SHRINKAGE_K = 8.0


# Base numeric fields expected per fighter profile.
PROFILE_NUMERIC_FIELDS = [
    "wins",
    "losses",
    "draws",
    "age",
    "height_in",
    "weight_lbs",
    "reach_in",
    "slpm",
    "sapm",
    "str_acc",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
]


# Stance categories observed in UFC-style datasets.
STANCE_VALUES = [
    "orthodox",
    "southpaw",
    "switch",
    "open stance",
    "sideways",
    "other",
    "unknown",
]


PROFILE_KEY_ALIASES = {
    "wins": {"wins", "win", "w"},
    "losses": {"losses", "loss", "l"},
    "draws": {"draws", "draw", "d"},
    "age": {"age"},
    "height_in": {"height", "height_in", "height_inches"},
    "weight_lbs": {"weight", "weight_lbs", "weight_pounds"},
    "reach_in": {"reach", "reach_in", "reach_inches"},
    "slpm": {"slpm", "sig_str_landed_per_min", "strikes_landed_per_min"},
    "sapm": {"sapm", "sig_str_absorbed_per_min", "strikes_absorbed_per_min"},
    "str_acc": {
        "str_acc",
        "str_acc_pct",
        "striking_accuracy",
        "sig_str_acc",
        "sig_strike_accuracy",
    },
    "str_def": {
        "str_def",
        "str_def_pct",
        "striking_defense",
        "sig_str_def",
        "sig_strike_defense",
    },
    "td_avg": {"td_avg", "td_average", "takedown_avg"},
    "td_acc": {"td_acc", "td_acc_pct", "takedown_acc", "takedown_accuracy"},
    "td_def": {"td_def", "td_def_pct", "takedown_def", "takedown_defense"},
    "sub_avg": {"sub_avg", "submission_avg", "submission_avg_15"},
    "stance": {"stance"},
}


def normalize_key(raw_key: str) -> str:
    key = str(raw_key).strip().lower()
    key = key.replace(".", "")
    key = key.replace("%", "pct")
    key = key.replace("/", "_")
    key = re.sub(r"[^a-z0-9_]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def parse_float(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return np.nan
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "--", "n/a"}:
        return np.nan

    # Handle plain percentages here if they slipped through.
    pct_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    num_match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not num_match:
        return np.nan
    return float(num_match.group(0))


def parse_percentage(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val):
            return np.nan
        # If user passes 45 instead of 0.45, normalize.
        return val / 100.0 if val > 1.0 else val

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "--", "n/a"}:
        return np.nan
    pct_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if pct_match:
        return float(pct_match.group(1)) / 100.0
    num = parse_float(text)
    if math.isnan(num):
        return np.nan
    return num / 100.0 if num > 1.0 else num


def parse_height_to_inches(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val):
            return np.nan
        return val

    text = str(value).strip()
    if not text:
        return np.nan

    # 5'11" format.
    ft_in_match = re.search(r"(\d+)\s*'\s*(\d+)", text)
    if ft_in_match:
        feet = int(ft_in_match.group(1))
        inches = int(ft_in_match.group(2))
        return float(feet * 12 + inches)

    # 71 in / 71" format.
    in_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:in|inches|\")", text.lower())
    if in_match:
        return float(in_match.group(1))

    # Last fallback.
    return parse_float(text)


def parse_weight_to_lbs(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val):
            return np.nan
        return val
    text = str(value).strip()
    if not text:
        return np.nan
    return parse_float(text)


def parse_record(value) -> Dict[str, float]:
    if value is None:
        return {"wins": np.nan, "losses": np.nan, "draws": np.nan}
    text = str(value).strip()
    match = re.search(r"(\d+)\s*-\s*(\d+)(?:\s*-\s*(\d+))?", text)
    if not match:
        return {"wins": np.nan, "losses": np.nan, "draws": np.nan}
    wins = float(match.group(1))
    losses = float(match.group(2))
    draws = float(match.group(3) or 0.0)
    return {"wins": wins, "losses": losses, "draws": draws}


def parse_age(value, today: Optional[date] = None) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val):
            return np.nan
        return val

    text = str(value).strip()
    if not text:
        return np.nan

    # Parse date of birth text.
    today = today or date.today()
    dob_formats = ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]
    for fmt in dob_formats:
        try:
            dob = datetime.strptime(text, fmt).date()
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return float(years)
        except ValueError:
            continue

    # If already numeric-like, trust as age.
    maybe_age = parse_float(text)
    if not math.isnan(maybe_age) and maybe_age < 120:
        return maybe_age

    return np.nan


def normalize_stance(value) -> str:
    if value is None:
        return "unknown"
    stance = str(value).strip().lower()
    if not stance or stance in {"nan", "none", "--", "n/a"}:
        return "unknown"
    if stance in {"ortho", "orthodox"}:
        return "orthodox"
    if stance in {"southpaw", "south paw"}:
        return "southpaw"
    if stance in {"switch", "switch stance"}:
        return "switch"
    if stance in {"open stance", "open"}:
        return "open stance"
    if stance in {"sideways"}:
        return "sideways"
    if stance in STANCE_VALUES:
        return stance
    return "other"


def _extract_with_aliases(values: Dict[str, object], canonical_key: str):
    aliases = PROFILE_KEY_ALIASES.get(canonical_key, {canonical_key})
    for alias in aliases:
        if alias in values:
            return values[alias]
    return None


def normalize_profile_input(raw_profile: Dict[str, object]) -> Dict[str, object]:
    """
    Normalize a profile dictionary into canonical UFC.com-style keys.
    """
    lowered = {normalize_key(k): v for k, v in raw_profile.items()}

    # Support composite record fields.
    record_value = _extract_with_aliases(lowered, "record")
    if record_value is None:
        for possible in ("record", "career_record", "w_l_d"):
            if possible in lowered:
                record_value = lowered[possible]
                break
    if record_value is not None:
        parsed_record = parse_record(record_value)
        for rec_key, rec_val in parsed_record.items():
            if rec_key not in lowered or lowered.get(rec_key) in (None, "", np.nan):
                lowered[rec_key] = rec_val

    profile = {}
    for field in PROFILE_NUMERIC_FIELDS:
        value = _extract_with_aliases(lowered, field)
        if field in {"str_acc", "str_def", "td_acc", "td_def"}:
            profile[field] = parse_percentage(value)
        elif field == "height_in":
            profile[field] = parse_height_to_inches(value)
        elif field == "weight_lbs":
            profile[field] = parse_weight_to_lbs(value)
        elif field == "reach_in":
            profile[field] = parse_height_to_inches(value)
        elif field == "age":
            profile[field] = parse_age(value)
        else:
            profile[field] = parse_float(value)

    stance_value = _extract_with_aliases(lowered, "stance")
    profile["stance"] = normalize_stance(stance_value)
    return profile


def build_feature_dict(
    red_profile: Dict[str, object],
    blue_profile: Dict[str, object],
    stance_values: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Build a model feature dictionary from two canonical profiles.
    """
    stance_values = list(stance_values or STANCE_VALUES)

    red = normalize_profile_input(red_profile)
    blue = normalize_profile_input(blue_profile)

    features: Dict[str, float] = {}
    for field in PROFILE_NUMERIC_FIELDS:
        red_val = red.get(field, np.nan)
        blue_val = blue.get(field, np.nan)
        features[f"red_{field}"] = red_val
        features[f"blue_{field}"] = blue_val
        features[f"delta_{field}"] = (
            red_val - blue_val if not (math.isnan(red_val) or math.isnan(blue_val)) else np.nan
        )
        features[f"ratio_{field}"] = (
            red_val / blue_val
            if not (math.isnan(red_val) or math.isnan(blue_val) or blue_val == 0)
            else np.nan
        )

    red_total = sum(
        red.get(k, 0.0)
        for k in ("wins", "losses", "draws")
        if not math.isnan(float(red.get(k, np.nan)))
    )
    blue_total = sum(
        blue.get(k, 0.0)
        for k in ("wins", "losses", "draws")
        if not math.isnan(float(blue.get(k, np.nan)))
    )
    red_win_rate = red.get("wins", np.nan) / red_total if red_total > 0 else np.nan
    blue_win_rate = blue.get("wins", np.nan) / blue_total if blue_total > 0 else np.nan

    features["red_total_fights"] = red_total if red_total > 0 else np.nan
    features["blue_total_fights"] = blue_total if blue_total > 0 else np.nan
    features["red_win_rate"] = red_win_rate
    features["blue_win_rate"] = blue_win_rate
    features["delta_win_rate"] = (
        red_win_rate - blue_win_rate
        if not (math.isnan(red_win_rate) or math.isnan(blue_win_rate))
        else np.nan
    )

    red_reliability = red_total / (red_total + SHRINKAGE_K) if red_total > 0 else 0.0
    blue_reliability = blue_total / (blue_total + SHRINKAGE_K) if blue_total > 0 else 0.0
    features["red_stat_reliability"] = red_reliability
    features["blue_stat_reliability"] = blue_reliability
    features["delta_stat_reliability"] = red_reliability - blue_reliability
    features["red_experience_log"] = math.log1p(red_total) if red_total > 0 else 0.0
    features["blue_experience_log"] = math.log1p(blue_total) if blue_total > 0 else 0.0
    features["delta_experience_log"] = features["red_experience_log"] - features["blue_experience_log"]

    # Shrink high-variance rate/per-minute stats toward UFC-wide priors.
    raw_metric_map = {
        "win_rate": (red_win_rate, blue_win_rate),
        "str_acc": (red.get("str_acc", np.nan), blue.get("str_acc", np.nan)),
        "str_def": (red.get("str_def", np.nan), blue.get("str_def", np.nan)),
        "td_acc": (red.get("td_acc", np.nan), blue.get("td_acc", np.nan)),
        "td_def": (red.get("td_def", np.nan), blue.get("td_def", np.nan)),
        "td_avg": (red.get("td_avg", np.nan), blue.get("td_avg", np.nan)),
        "sub_avg": (red.get("sub_avg", np.nan), blue.get("sub_avg", np.nan)),
        "slpm": (red.get("slpm", np.nan), blue.get("slpm", np.nan)),
        "sapm": (red.get("sapm", np.nan), blue.get("sapm", np.nan)),
    }
    for metric, (red_raw, blue_raw) in raw_metric_map.items():
        prior = SHRINKAGE_PRIORS[metric]
        red_shrunk = (
            red_reliability * red_raw + (1.0 - red_reliability) * prior
            if not math.isnan(red_raw)
            else prior
        )
        blue_shrunk = (
            blue_reliability * blue_raw + (1.0 - blue_reliability) * prior
            if not math.isnan(blue_raw)
            else prior
        )
        features[f"red_shrunk_{metric}"] = red_shrunk
        features[f"blue_shrunk_{metric}"] = blue_shrunk
        features[f"delta_shrunk_{metric}"] = red_shrunk - blue_shrunk

    features["red_shrunk_strike_margin"] = (
        features["red_shrunk_slpm"] - features["red_shrunk_sapm"]
    )
    features["blue_shrunk_strike_margin"] = (
        features["blue_shrunk_slpm"] - features["blue_shrunk_sapm"]
    )
    features["delta_shrunk_strike_margin"] = (
        features["red_shrunk_strike_margin"] - features["blue_shrunk_strike_margin"]
    )

    # Compact interaction features with strong signal.
    red_strike_margin = (
        red.get("slpm", np.nan) - red.get("sapm", np.nan)
        if not (math.isnan(red.get("slpm", np.nan)) or math.isnan(red.get("sapm", np.nan)))
        else np.nan
    )
    blue_strike_margin = (
        blue.get("slpm", np.nan) - blue.get("sapm", np.nan)
        if not (math.isnan(blue.get("slpm", np.nan)) or math.isnan(blue.get("sapm", np.nan)))
        else np.nan
    )
    features["red_strike_margin"] = red_strike_margin
    features["blue_strike_margin"] = blue_strike_margin
    features["delta_strike_margin"] = (
        red_strike_margin - blue_strike_margin
        if not (math.isnan(red_strike_margin) or math.isnan(blue_strike_margin))
        else np.nan
    )

    features["red_wrestling_blend"] = (
        red.get("td_avg", np.nan) * red.get("td_acc", np.nan)
        if not (math.isnan(red.get("td_avg", np.nan)) or math.isnan(red.get("td_acc", np.nan)))
        else np.nan
    )
    features["blue_wrestling_blend"] = (
        blue.get("td_avg", np.nan) * blue.get("td_acc", np.nan)
        if not (math.isnan(blue.get("td_avg", np.nan)) or math.isnan(blue.get("td_acc", np.nan)))
        else np.nan
    )
    if not (
        math.isnan(features["red_wrestling_blend"])
        or math.isnan(features["blue_wrestling_blend"])
    ):
        features["delta_wrestling_blend"] = (
            features["red_wrestling_blend"] - features["blue_wrestling_blend"]
        )
    else:
        features["delta_wrestling_blend"] = np.nan

    red_stance = normalize_stance(red.get("stance"))
    blue_stance = normalize_stance(blue.get("stance"))
    features["stance_match"] = 1.0 if red_stance == blue_stance else 0.0
    for stance in stance_values:
        slug = stance.replace(" ", "_")
        features[f"red_stance_{slug}"] = 1.0 if red_stance == stance else 0.0
        features[f"blue_stance_{slug}"] = 1.0 if blue_stance == stance else 0.0

    return features
