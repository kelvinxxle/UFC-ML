#!/usr/bin/env python3
"""
UFC Fight Predictor using UFC.com-compatible fighter profile inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ufc_profile_schema import STANCE_VALUES, build_feature_dict


class UFCFightPredictor:
    def __init__(
        self,
        model_path: str = "ufc_rf_balanced_smote.pkl",
        features_path: str = "ufc_features.csv",
    ):
        self.model = None
        self.feature_columns = []
        self.impute_values = {}
        self.feature_means = {}
        self.feature_stds = {}
        self.fighter_context = {}
        self.stance_values = STANCE_VALUES

        try:
            loaded = joblib.load(model_path)
        except Exception as exc:
            print(f"Error loading model bundle '{model_path}': {exc}")
            return

        # Preferred path: new model bundle format from process_ufc_data.py.
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.feature_columns = list(loaded.get("feature_columns", []))
            self.impute_values = dict(loaded.get("impute_values", {}))
            self.feature_means = dict(loaded.get("feature_means", {}))
            self.feature_stds = dict(loaded.get("feature_stds", {}))
            self.fighter_context = dict(loaded.get("fighter_context", {}))
            self.stance_values = list(loaded.get("stance_values", STANCE_VALUES))
            if hasattr(self.model, "feature_names_in_"):
                model_feature_names = [str(c) for c in self.model.feature_names_in_]
                if model_feature_names:
                    self.feature_columns = model_feature_names
            print(f"Loaded aligned model bundle: {model_path}")
            print(f"Model expects {len(self.feature_columns)} features")
            return

        # Backward compatibility: plain estimator + external feature file.
        self.model = loaded
        try:
            feature_df = pd.read_csv(features_path)
            self.feature_columns = [col for col in feature_df.columns if col != "label"]
            if hasattr(self.model, "feature_names_in_"):
                model_feature_names = [str(c) for c in self.model.feature_names_in_]
                if model_feature_names:
                    self.feature_columns = model_feature_names
            self.feature_means = (
                feature_df[self.feature_columns].mean(numeric_only=True).to_dict()
                if self.feature_columns
                else {}
            )
            self.feature_stds = (
                feature_df[self.feature_columns].std(numeric_only=True).to_dict()
                if self.feature_columns
                else {}
            )
            print(f"Loaded legacy model: {model_path}")
            print(f"Loaded feature schema from: {features_path}")
            print(f"Model expects {len(self.feature_columns)} features")
        except Exception as exc:
            print(f"Error loading feature schema '{features_path}': {exc}")
            self.model = None
            self.feature_columns = []

    def _lookup_schedule_context(self, fighter_name: str) -> tuple[float, float]:
        if not self.fighter_context:
            return 0.5, 0.0
        strength_map = self.fighter_context.get("schedule_strength", {})
        std_map = self.fighter_context.get("schedule_std", {})
        default_strength = float(self.fighter_context.get("default_schedule_strength", 0.5))
        default_std = float(self.fighter_context.get("default_schedule_std", 0.0))
        return (
            float(strength_map.get(fighter_name, default_strength)),
            float(std_map.get(fighter_name, default_std)),
        )

    def _build_aligned_frame(
        self,
        red_profile: Dict,
        blue_profile: Dict,
        red_fighter_name: Optional[str] = None,
        blue_fighter_name: Optional[str] = None,
    ) -> pd.DataFrame:
        feature_dict = build_feature_dict(
            red_profile=red_profile,
            blue_profile=blue_profile,
            stance_values=self.stance_values,
        )

        # Add opponent-quality context features if present in model schema.
        red_sos, red_sos_std = self._lookup_schedule_context(red_fighter_name or "")
        blue_sos, blue_sos_std = self._lookup_schedule_context(blue_fighter_name or "")
        default_sos = (
            float(self.fighter_context.get("default_schedule_strength", 0.5))
            if self.fighter_context
            else 0.5
        )
        feature_dict["red_schedule_strength"] = red_sos
        feature_dict["blue_schedule_strength"] = blue_sos
        feature_dict["delta_schedule_strength"] = red_sos - blue_sos
        feature_dict["red_schedule_std"] = red_sos_std
        feature_dict["blue_schedule_std"] = blue_sos_std
        feature_dict["delta_schedule_std"] = red_sos_std - blue_sos_std

        red_scale = red_sos / default_sos if default_sos > 0 else 1.0
        blue_scale = blue_sos / default_sos if default_sos > 0 else 1.0
        for metric in ("win_rate", "td_acc", "td_def", "str_acc", "str_def", "td_avg", "sub_avg"):
            red_key = f"red_shrunk_{metric}"
            blue_key = f"blue_shrunk_{metric}"
            if red_key in feature_dict and blue_key in feature_dict:
                red_adj = feature_dict[red_key] * red_scale
                blue_adj = feature_dict[blue_key] * blue_scale
                feature_dict[f"red_adj_{metric}"] = red_adj
                feature_dict[f"blue_adj_{metric}"] = blue_adj
                feature_dict[f"delta_adj_{metric}"] = red_adj - blue_adj

        df = pd.DataFrame([feature_dict]).reindex(columns=self.feature_columns)
        df = df.fillna(self.impute_values).fillna(0.0)
        return df

    def _predict_ordered_frame(self, X: pd.DataFrame) -> tuple[int, float, float]:
        pred = int(self.model.predict(X)[0])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            red_prob = float(proba[0])
            blue_prob = float(proba[1]) if len(proba) > 1 else float(1.0 - red_prob)
        else:
            red_prob = float(pred == 0)
            blue_prob = float(pred == 1)
        return pred, red_prob, blue_prob

    @staticmethod
    def _feature_to_label(feature: str) -> str:
        full_term_map = {
            "slpm": "Significant Strikes Landed Per Minute",
            "sapm": "Significant Strikes Absorbed Per Minute",
            "str_acc": "Significant Strike Accuracy",
            "str_def": "Significant Strike Defense",
            "td_avg": "Takedowns Landed Per 15 Minutes",
            "td_acc": "Takedown Accuracy",
            "td_def": "Takedown Defense",
            "sub_avg": "Submission Attempts Per 15 Minutes",
            "wins": "Career Wins",
            "losses": "Career Losses",
            "draws": "Career Draws",
            "win_rate": "Win Rate",
            "strike_margin": "Net Significant Striking Margin",
            "wrestling_blend": "Wrestling Effectiveness Blend",
            "schedule_strength": "Strength of Opponent Schedule",
            "schedule_std": "Opponent Quality Variability",
            "stat_reliability": "Stat Reliability (Sample Size Adjusted)",
            "experience_log": "Experience (Log-Scaled Fight Count)",
            "height_in": "Height (inches)",
            "reach_in": "Reach (inches)",
            "weight_lbs": "Weight (pounds)",
            "age": "Age",
        }

        def metric_phrase(metric: str) -> str:
            if metric.startswith("shrunk_"):
                return f"Reliability-Adjusted {metric_phrase(metric[len('shrunk_'):])}"
            if metric.startswith("adj_"):
                return f"Opponent-Adjusted {metric_phrase(metric[len('adj_'):])}"
            return full_term_map.get(metric, metric.replace("_", " ").title())

        if feature.startswith("delta_"):
            metric = feature.replace("delta_", "")
            return metric_phrase(metric)
        if feature.startswith("red_"):
            metric = feature.replace("red_", "")
            return f"Red {metric_phrase(metric)}"
        if feature.startswith("blue_"):
            metric = feature.replace("blue_", "")
            return f"Blue {metric_phrase(metric)}"
        if feature.startswith("ratio_"):
            metric = feature.replace("ratio_", "")
            return f"Ratio of {metric_phrase(metric)}"
        return feature.replace("_", " ").title()

    def _build_reasoning(self, X: pd.DataFrame, prediction_code: int, top_n: int = 5) -> List[Dict]:
        if not hasattr(self.model, "feature_importances_"):
            return []

        row = X.iloc[0]
        names = (
            [str(c) for c in self.model.feature_names_in_]
            if hasattr(self.model, "feature_names_in_")
            else self.feature_columns
        )
        importances = self.model.feature_importances_
        if len(names) != len(importances):
            return []

        scored = []
        for feat, importance in zip(names, importances):
            if importance <= 0:
                continue
            # Restrict to differential features to avoid redundant red/blue duplicates in explanations.
            if not feat.startswith("delta_"):
                continue

            value = float(row.get(feat, 0.0))
            baseline = float(self.feature_means.get(feat, 0.0))
            std = float(self.feature_stds.get(feat, 1.0))
            if std <= 1e-9:
                std = 1.0
            z_score = (value - baseline) / std
            signed_impact = float(importance) * z_score
            scored.append(
                {
                    "feature": feat,
                    "label": self._feature_to_label(feat),
                    "importance": float(importance),
                    "value": value,
                    "baseline": baseline,
                    "z_score": float(z_score),
                    "signed_impact": float(signed_impact),
                }
            )

        if not scored:
            return []

        if prediction_code == 0:
            favored = [r for r in scored if r["signed_impact"] > 0]
            direction = "red"
        else:
            favored = [r for r in scored if r["signed_impact"] < 0]
            direction = "blue"
            for row_item in favored:
                row_item["signed_impact"] = abs(row_item["signed_impact"])

        favored.sort(key=lambda r: r["signed_impact"], reverse=True)
        top = []
        seen_buckets = set()
        for row_item in favored:
            feat = row_item["feature"]
            if feat.startswith("delta_adj_"):
                bucket = feat[len("delta_adj_") :]
            elif feat.startswith("delta_shrunk_"):
                bucket = feat[len("delta_shrunk_") :]
            elif feat.startswith("delta_"):
                bucket = feat[len("delta_") :]
            else:
                bucket = feat
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            top.append(row_item)
            if len(top) >= top_n:
                break
        for row_item in top:
            row_item["favors"] = direction
        return top

    @staticmethod
    def _build_reasoning_summary(reasons: List[Dict], pred_code: int) -> str:
        if not reasons:
            return "No stable feature-based explanation was available for this model output."
        side = "Red corner" if pred_code == 0 else "Blue corner"
        labels = ", ".join(r["label"] for r in reasons[:3])
        return f"Model leans toward {side} mainly due to: {labels}."

    def predict_fight(
        self,
        red_fighter_name: str,
        blue_fighter_name: str,
        red_profile: Dict,
        blue_profile: Dict,
    ) -> Dict:
        if self.model is None:
            return {"error": "Model not loaded"}
        if not self.feature_columns:
            return {"error": "Feature schema not available"}

        try:
            X_forward = self._build_aligned_frame(
                red_profile,
                blue_profile,
                red_fighter_name=red_fighter_name,
                blue_fighter_name=blue_fighter_name,
            )
            _, forward_red_prob, _ = self._predict_ordered_frame(X_forward)

            X_swapped = self._build_aligned_frame(
                blue_profile,
                red_profile,
                red_fighter_name=blue_fighter_name,
                blue_fighter_name=red_fighter_name,
            )
            _, swapped_red_prob, _ = self._predict_ordered_frame(X_swapped)

            # Average both fighter orders so the final probability is invariant
            # to which side was arbitrarily labeled red or blue at inference time.
            red_prob = float(np.clip((forward_red_prob + (1.0 - swapped_red_prob)) / 2.0, 0.0, 1.0))
            blue_prob = float(1.0 - red_prob)
            pred = 0 if red_prob >= blue_prob else 1

            winner = red_fighter_name if pred == 0 else blue_fighter_name
            confidence = max(red_prob, blue_prob)
            reasoning = self._build_reasoning(X_forward, prediction_code=pred, top_n=5)
            reasoning_summary = self._build_reasoning_summary(reasoning, pred_code=pred)

            return {
                "red_fighter": red_fighter_name,
                "blue_fighter": blue_fighter_name,
                "predicted_winner": winner,
                "prediction_code": pred,  # 0 = Red corner, 1 = Blue corner
                "confidence": f"{confidence:.1%}",
                "confidence_value": confidence,
                "red_win_probability": f"{red_prob:.1%}",
                "red_win_probability_value": red_prob,
                "blue_win_probability": f"{blue_prob:.1%}",
                "blue_win_probability_value": blue_prob,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_schema": "ufc_profile_aligned_v2",
                "probability_method": "corner_swap_averaged",
                "reasoning_summary": reasoning_summary,
                "reasoning": reasoning,
                "explainability_note": "Feature-importance heuristic explanation (not causal proof).",
            }
        except Exception as exc:
            return {"error": f"Prediction failed: {exc}"}

    def predict_from_ufc_com(
        self,
        red_fighter_name: str,
        blue_fighter_name: str,
        red_ufc_profile: Dict,
        blue_ufc_profile: Dict,
    ) -> Dict:
        """
        Convenience alias for UFC.com profile inputs.
        """
        return self.predict_fight(
            red_fighter_name=red_fighter_name,
            blue_fighter_name=blue_fighter_name,
            red_profile=red_ufc_profile,
            blue_profile=blue_ufc_profile,
        )

    @staticmethod
    def get_example_profile_input() -> Dict[str, Dict]:
        """
        Example UFC.com-style profile payloads.
        """
        return {
            "red_profile": {
                "record": "18-3-0",
                "height": "5'11\"",
                "weight": "155 lbs",
                "reach": "72\"",
                "stance": "Orthodox",
                "age": 30,
                "SLpM": 4.85,
                "SApM": 3.21,
                "Str. Acc.": "48%",
                "Str. Def": "57%",
                "TD Avg.": 1.42,
                "TD Acc.": "39%",
                "TD Def.": "71%",
                "Sub. Avg.": 0.4,
            },
            "blue_profile": {
                "record": "16-4-0",
                "height": "5'10\"",
                "weight": "155 lbs",
                "reach": "71\"",
                "stance": "Southpaw",
                "age": 28,
                "SLpM": 4.11,
                "SApM": 3.55,
                "Str. Acc.": "44%",
                "Str. Def": "53%",
                "TD Avg.": 0.98,
                "TD Acc.": "34%",
                "TD Def.": "68%",
                "Sub. Avg.": 0.2,
            },
        }


def main() -> None:
    print("=== UFC Fight Predictor (Profile-Aligned) ===")
    predictor = UFCFightPredictor()
    if predictor.model is None:
        print("Cannot run demo without a trained model bundle.")
        print("Run: python process_ufc_data.py --input ufc_profile_fights.csv")
        return

    example = predictor.get_example_profile_input()
    result = predictor.predict_from_ufc_com(
        red_fighter_name="Red Fighter",
        blue_fighter_name="Blue Fighter",
        red_ufc_profile=example["red_profile"],
        blue_ufc_profile=example["blue_profile"],
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Predicted winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Red win probability: {result['red_win_probability']}")
        print(f"Blue win probability: {result['blue_win_probability']}")


if __name__ == "__main__":
    main()
