#!/usr/bin/env python3
"""
Bet sizing utilities for sportsbook-style American odds.
"""

from __future__ import annotations

from typing import Dict, Optional, Union


OddsInput = Union[str, int, float]
PASS_EDGE_THRESHOLD = 0.03
PLAYABLE_EDGE_THRESHOLD = 0.05
STRONG_EDGE_THRESHOLD = 0.08


def parse_american_odds(odds: OddsInput) -> int:
    if isinstance(odds, str):
        cleaned = odds.strip().replace(" ", "")
        if not cleaned:
            raise ValueError("American odds cannot be blank.")
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        try:
            parsed = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"Invalid American odds: {odds}") from exc
    elif isinstance(odds, float):
        if not odds.is_integer():
            raise ValueError("American odds must be whole numbers.")
        parsed = int(odds)
    elif isinstance(odds, int):
        parsed = odds
    else:
        raise ValueError(f"Unsupported odds type: {type(odds).__name__}")

    if parsed == 0:
        raise ValueError("American odds cannot be 0.")
    if -100 < parsed < 100:
        raise ValueError("American odds must be <= -100 or >= +100.")
    return parsed


def american_profit_multiple(odds: OddsInput) -> float:
    parsed = parse_american_odds(odds)
    return parsed / 100.0 if parsed > 0 else 100.0 / abs(parsed)


def american_decimal_odds(odds: OddsInput) -> float:
    return 1.0 + american_profit_multiple(odds)


def american_implied_probability(odds: OddsInput) -> float:
    parsed = parse_american_odds(odds)
    if parsed > 0:
        return 100.0 / (parsed + 100.0)
    abs_odds = abs(parsed)
    return abs_odds / (abs_odds + 100.0)


def expected_value_per_dollar(win_probability: float, odds: OddsInput) -> float:
    _validate_probability(win_probability)
    payout_multiple = american_profit_multiple(odds)
    lose_probability = 1.0 - win_probability
    return (win_probability * payout_multiple) - lose_probability


def full_kelly_fraction(win_probability: float, odds: OddsInput) -> float:
    _validate_probability(win_probability)
    payout_multiple = american_profit_multiple(odds)
    lose_probability = 1.0 - win_probability
    return ((payout_multiple * win_probability) - lose_probability) / payout_multiple


def analyze_bet(
    fighter_name: str,
    win_probability: float,
    odds: OddsInput,
    bankroll: Optional[float] = None,
    fractional_kelly: float = 0.25,
) -> Dict[str, object]:
    _validate_probability(win_probability)
    if fractional_kelly <= 0:
        raise ValueError("Fractional Kelly must be greater than 0.")
    if bankroll is not None and bankroll < 0:
        raise ValueError("Bankroll cannot be negative.")

    parsed_odds = parse_american_odds(odds)
    implied_probability = american_implied_probability(parsed_odds)
    payout_multiple = american_profit_multiple(parsed_odds)
    decimal_odds = american_decimal_odds(parsed_odds)
    edge = win_probability - implied_probability
    ev_per_dollar = expected_value_per_dollar(win_probability, parsed_odds)
    raw_kelly = full_kelly_fraction(win_probability, parsed_odds)
    capped_kelly = max(raw_kelly, 0.0)
    fractional_fraction = capped_kelly * fractional_kelly
    edge_grade = grade_edge(edge=edge, ev_per_dollar=ev_per_dollar, kelly_fraction=capped_kelly)
    edge_grade_note = describe_edge_grade(edge_grade)

    full_stake = bankroll * capped_kelly if bankroll is not None else None
    fractional_stake = bankroll * fractional_fraction if bankroll is not None else None
    expected_profit = fractional_stake * ev_per_dollar if fractional_stake is not None else None

    return {
        "fighter": fighter_name,
        "american_odds": parsed_odds,
        "decimal_odds": decimal_odds,
        "profit_multiple": payout_multiple,
        "model_win_probability": win_probability,
        "implied_probability": implied_probability,
        "break_even_probability": implied_probability,
        "edge": edge,
        "edge_vs_market": edge,
        "edge_grade": edge_grade,
        "edge_grade_note": edge_grade_note,
        "expected_value_per_dollar": ev_per_dollar,
        "full_kelly_fraction": capped_kelly,
        "raw_kelly_fraction": raw_kelly,
        "fractional_kelly_multiplier": fractional_kelly,
        "fractional_kelly_fraction": fractional_fraction,
        "full_kelly_stake": full_stake,
        "fractional_kelly_stake": fractional_stake,
        "expected_profit_on_fractional_stake": expected_profit,
        "recommendation": edge_grade,
    }


def analyze_market(
    red_name: str,
    blue_name: str,
    red_win_probability: float,
    blue_win_probability: float,
    red_odds: OddsInput,
    blue_odds: OddsInput,
    bankroll: Optional[float] = None,
    fractional_kelly: float = 0.25,
) -> Dict[str, object]:
    red_side = analyze_bet(
        fighter_name=red_name,
        win_probability=red_win_probability,
        odds=red_odds,
        bankroll=bankroll,
        fractional_kelly=fractional_kelly,
    )
    blue_side = analyze_bet(
        fighter_name=blue_name,
        win_probability=blue_win_probability,
        odds=blue_odds,
        bankroll=bankroll,
        fractional_kelly=fractional_kelly,
    )

    sides = [red_side, blue_side]
    positive_ev = [side for side in sides if side["expected_value_per_dollar"] > 0]
    best_bet = None
    if positive_ev:
        best_bet = max(
            positive_ev,
            key=lambda side: (
                float(side["expected_value_per_dollar"]),
                float(side["fractional_kelly_fraction"]),
            ),
        )

    market_overround = (
        float(red_side["implied_probability"]) + float(blue_side["implied_probability"]) - 1.0
    )
    if best_bet is None:
        summary = "No positive expected-value wager based on the current odds."
    else:
        summary = (
            f"Best edge vs market is {best_bet['fighter']} at {best_bet['american_odds']:+d}. "
            f"Grade: {best_bet['edge_grade']}. "
            f"Expected value is {best_bet['expected_value_per_dollar']:.3f} per $1 risked."
        )

    return {
        "bankroll": bankroll,
        "fractional_kelly_multiplier": fractional_kelly,
        "market_overround": market_overround,
        "edge_definition": "Edge vs market = model win probability minus sportsbook implied probability.",
        "market_hold_definition": "Market hold = combined sportsbook vig across both sides of the market.",
        "recommendation_summary": summary,
        "best_bet": best_bet,
        "sides": sides,
    }


def _validate_probability(probability: float) -> None:
    if probability < 0 or probability > 1:
        raise ValueError("Win probability must be between 0 and 1.")


def grade_edge(edge: float, ev_per_dollar: float, kelly_fraction: float) -> str:
    if ev_per_dollar <= 0 or kelly_fraction <= 0:
        return "Pass"
    if edge < PASS_EDGE_THRESHOLD:
        return "Pass"
    if edge < PLAYABLE_EDGE_THRESHOLD:
        return "Lean"
    if edge < STRONG_EDGE_THRESHOLD:
        return "Playable"
    return "Strong"


def describe_edge_grade(grade: str) -> str:
    notes = {
        "Pass": "Edge cushion is too thin or negative after accounting for the market price.",
        "Lean": "Positive edge, but still close enough to model error that sizing should stay conservative.",
        "Playable": "Meaningful edge cushion relative to the market price.",
        "Strong": "Large edge cushion, but verify the line and model calibration before sizing up.",
    }
    return notes.get(grade, "")
