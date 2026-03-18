#!/usr/bin/env python3
"""
Build a profile-aligned UFC training dataset.

It reads fight URLs, scrapes fighter profile stats, and outputs one stable
schema compatible with UFC.com-style prediction inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from prefight_dataset_builder import PrefightDatasetBuilder
from ufc_profile_schema import PROFILE_NUMERIC_FIELDS, normalize_key, normalize_profile_input


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def pick_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def parse_label_from_text(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s+[A-Za-z][A-Za-z .]+:\s|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


class UFCStatsProfileDatasetBuilder:
    def __init__(
        self,
        delay_seconds: float = 0.4,
        timeout_seconds: int = 20,
        max_retries: int = 2,
        cache_dir: str = ".cache",
    ):
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.profile_cache_path = self.cache_dir / "ufc_profile_cache.json"
        self.fight_context_cache_path = self.cache_dir / "ufc_fight_context_cache.json"

        self.profile_cache: Dict[str, Dict] = self._load_json_cache(self.profile_cache_path)
        self.fight_context_cache: Dict[str, Dict] = self._load_json_cache(self.fight_context_cache_path)

    @staticmethod
    def _load_json_cache(path: Path) -> Dict[str, Dict]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_json_cache(path: Path, payload: Dict[str, Dict]) -> None:
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)
                return BeautifulSoup(response.content, "html.parser")
            except Exception:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1.0 + attempt)
        return None

    @staticmethod
    def _extract_labeled_value(soup: BeautifulSoup, label: str) -> str:
        target = f"{label.lower()}:"
        selectors = [
            "i.b-fight-details__text-item",
            "p.b-fight-details__text",
            "li.b-list__box-list-item",
        ]
        for selector in selectors:
            for node in soup.select(selector):
                text = " ".join(node.stripped_strings)
                if text.lower().startswith(target):
                    return text.split(":", 1)[1].strip()
        text_blob = " ".join(soup.stripped_strings)
        return parse_label_from_text(text_blob, label)

    def _extract_fight_context(self, fight_url: str) -> Dict[str, str]:
        if fight_url in self.fight_context_cache:
            return dict(self.fight_context_cache[fight_url])

        soup = self._get_soup(fight_url)
        if soup is None:
            return {}

        context = {
            "fight_url": fight_url,
            "red_fighter": "",
            "blue_fighter": "",
            "winner": "",
            "red_profile_url": "",
            "blue_profile_url": "",
            "event": "",
            "date": "",
            "weight_class": "",
            "method": "",
            "round": "",
            "time": "",
        }

        fighter_links = soup.select("a.b-link.b-fight-details__person-link")
        if len(fighter_links) >= 2:
            context["red_fighter"] = fighter_links[0].get_text(strip=True)
            context["blue_fighter"] = fighter_links[1].get_text(strip=True)
            context["red_profile_url"] = fighter_links[0].get("href", "")
            context["blue_profile_url"] = fighter_links[1].get("href", "")
        else:
            fallback_links = soup.select("h3.b-fight-details__person-name a")
            if len(fallback_links) >= 2:
                context["red_fighter"] = fallback_links[0].get_text(strip=True)
                context["blue_fighter"] = fallback_links[1].get_text(strip=True)
                context["red_profile_url"] = fallback_links[0].get("href", "")
                context["blue_profile_url"] = fallback_links[1].get("href", "")

        for person in soup.select("div.b-fight-details__person"):
            status_icon = person.select_one("i.b-fight-details__person-status")
            if not status_icon:
                continue
            classes = status_icon.get("class", [])
            if any("green" in c.lower() for c in classes):
                winner_link = person.select_one("a.b-link.b-fight-details__person-link")
                if winner_link:
                    context["winner"] = winner_link.get_text(strip=True)
                else:
                    winner_link = person.select_one("h3.b-fight-details__person-name a")
                    if winner_link:
                        context["winner"] = winner_link.get_text(strip=True)
                break

        title = soup.select_one("span.b-content__title-highlight")
        if title:
            context["event"] = title.get_text(strip=True)
        if not context["event"]:
            event_link = soup.find("a", href=re.compile(r"/event-details/"))
            if event_link:
                context["event"] = event_link.get_text(" ", strip=True)

        wc_tag = soup.select_one("i.b-fight-details__fight-title")
        if wc_tag:
            context["weight_class"] = wc_tag.get_text(" ", strip=True)

        context["method"] = self._extract_labeled_value(soup, "Method")
        context["round"] = self._extract_labeled_value(soup, "Round")
        context["time"] = self._extract_labeled_value(soup, "Time")
        context["date"] = self._extract_labeled_value(soup, "Date")

        self.fight_context_cache[fight_url] = dict(context)
        return context

    def _extract_profile(self, profile_url: str) -> Dict[str, object]:
        if not profile_url:
            return normalize_profile_input({})
        if profile_url in self.profile_cache:
            return dict(self.profile_cache[profile_url])

        soup = self._get_soup(profile_url)
        if soup is None:
            return normalize_profile_input({})

        raw: Dict[str, object] = {}
        record_tag = soup.select_one("span.b-content__title-record")
        if record_tag:
            record_text = record_tag.get_text(" ", strip=True)
            match = re.search(r"(\d+)\s*-\s*(\d+)(?:\s*-\s*(\d+))?", record_text)
            if match:
                raw["wins"] = match.group(1)
                raw["losses"] = match.group(2)
                raw["draws"] = match.group(3) or "0"

        for li in soup.select("li.b-list__box-list-item"):
            text = " ".join(li.stripped_strings)
            if ":" not in text:
                continue
            label, value = text.split(":", 1)
            raw[normalize_key(label)] = value.strip()

        profile = normalize_profile_input(raw)
        self.profile_cache[profile_url] = profile
        return profile

    def build(
        self,
        input_csv: str,
        output_csv: str,
        max_fights: Optional[int] = None,
        mode: str = "legacy",
        manifest_out: Optional[str] = None,
        history_strategy: str = "exact",
    ) -> pd.DataFrame:
        if str(mode).strip().lower() == "prefight_v1":
            prefight_builder = PrefightDatasetBuilder(
                delay_seconds=self.delay_seconds,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                cache_dir=str(self.cache_dir),
            )
            manifest_path = manifest_out or str(Path(output_csv).with_name("ufc_prefight_manifest.json"))
            return prefight_builder.build_prefight(
                input_csv=input_csv,
                output_csv=output_csv,
                manifest_out=manifest_path,
                max_fights=max_fights,
                history_strategy=history_strategy,
            )

        src = pd.read_csv(input_csv)
        url_col = pick_column(src.columns, ["Fight_URL", "fight_url", "url"])
        red_col = pick_column(src.columns, ["Red", "red", "red_fighter"])
        blue_col = pick_column(src.columns, ["Blue", "blue", "blue_fighter"])
        winner_col = pick_column(src.columns, ["Winner", "winner"])

        if not url_col:
            raise ValueError(f"Missing fight URL column in {input_csv}.")

        existing_map = {}
        reuse_count = 0
        fetch_count = 0
        if Path(output_csv).exists():
            try:
                existing_df = pd.read_csv(output_csv)
                if "fight_url" in existing_df.columns and "profile_source" not in existing_df.columns:
                    existing_map = {
                        str(row.get("fight_url", "")).strip(): row.to_dict()
                        for _, row in existing_df.iterrows()
                        if str(row.get("fight_url", "")).strip()
                    }
            except Exception:
                existing_map = {}

        rows = []
        iterable = src if max_fights is None else src.head(int(max_fights))
        total = len(iterable)

        for idx, row in iterable.iterrows():
            fight_url = str(row.get(url_col, "")).strip()
            if not fight_url:
                continue

            if fight_url in existing_map:
                rows.append(existing_map[fight_url])
                reuse_count += 1
            else:
                context = self._extract_fight_context(fight_url)
                red_name = context.get("red_fighter") or str(row.get(red_col, "")).strip()
                blue_name = context.get("blue_fighter") or str(row.get(blue_col, "")).strip()
                winner = context.get("winner") or str(row.get(winner_col, "")).strip()

                red_profile = self._extract_profile(context.get("red_profile_url", ""))
                blue_profile = self._extract_profile(context.get("blue_profile_url", ""))

                out_row = {
                    "red_fighter": red_name,
                    "blue_fighter": blue_name,
                    "winner": winner,
                    "fight_url": fight_url,
                    "event": context.get("event", ""),
                    "date": context.get("date", ""),
                    "weight_class": context.get("weight_class", ""),
                    "method": context.get("method", ""),
                    "round": context.get("round", ""),
                    "time": context.get("time", ""),
                }
                for key, value in red_profile.items():
                    out_row[f"red_{key}"] = value
                for key, value in blue_profile.items():
                    out_row[f"blue_{key}"] = value
                rows.append(out_row)
                fetch_count += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"Processed {idx + 1}/{total} fights")

        out_df = pd.DataFrame(rows)
        profile_cols = (
            [f"red_{f}" for f in PROFILE_NUMERIC_FIELDS]
            + ["red_stance"]
            + [f"blue_{f}" for f in PROFILE_NUMERIC_FIELDS]
            + ["blue_stance"]
        )
        ordered_cols = [
            "red_fighter",
            "blue_fighter",
            "winner",
            "fight_url",
            "event",
            "date",
            "weight_class",
            "method",
            "round",
            "time",
        ] + profile_cols
        keep_cols = [c for c in ordered_cols if c in out_df.columns]
        remaining_cols = [c for c in out_df.columns if c not in keep_cols]
        out_df = out_df[keep_cols + remaining_cols]

        out_df.to_csv(output_csv, index=False)
        self._save_json_cache(self.profile_cache_path, self.profile_cache)
        self._save_json_cache(self.fight_context_cache_path, self.fight_context_cache)
        print(f"Saved {len(out_df)} rows to {output_csv}")
        print(
            f"Build summary: reused {reuse_count} existing fights, fetched {fetch_count} fights from UFCStats."
        )
        return out_df


def main():
    parser = argparse.ArgumentParser(description="Build UFC datasets")
    parser.add_argument("--input", default="ufc_fight_data.csv", help="Input fight CSV with Fight_URL")
    parser.add_argument("--output", default="ufc_prefight_fights.csv", help="Output CSV")
    parser.add_argument(
        "--manifest-out",
        default="ufc_prefight_manifest.json",
        help="Manifest JSON path for prefight_v1 mode",
    )
    parser.add_argument("--max-fights", type=int, default=None, help="Limit number of fights")
    parser.add_argument("--mode", choices=["legacy", "prefight_v1"], default="prefight_v1")
    parser.add_argument(
        "--delay", type=float, default=0.4, help="Delay between HTTP requests (seconds)"
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP request timeout (seconds)")
    parser.add_argument("--retries", type=int, default=2, help="Request retries per URL")
    parser.add_argument("--cache-dir", default=".cache", help="Directory for response caches")
    parser.add_argument(
        "--history-strategy",
        choices=["exact", "input_window_only"],
        default="exact",
        help="Prefight history collection strategy for prefight_v1 mode",
    )
    args = parser.parse_args()

    builder = UFCStatsProfileDatasetBuilder(
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        cache_dir=args.cache_dir,
    )
    output_csv = args.output
    if args.mode == "legacy" and output_csv == "ufc_prefight_fights.csv":
        output_csv = "ufc_profile_fights.csv"
    builder.build(
        input_csv=args.input,
        output_csv=output_csv,
        max_fights=args.max_fights,
        mode=args.mode,
        manifest_out=args.manifest_out,
        history_strategy=args.history_strategy,
    )


if __name__ == "__main__":
    main()
