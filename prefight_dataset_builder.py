#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ufc_profile_schema import normalize_key, normalize_profile_input, normalize_stance, parse_age


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
PREFIGHT_STATIC_FIELDS = ["height_in", "weight_lbs", "reach_in", "stance", "age_at_fight"]
PREFIGHT_PRIOR_FIELDS = [
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
ELO_INITIAL = 1500.0
ELO_K = 32.0
DRAW_LABELS = {"draw", "majority draw", "split draw"}
SKIP_RESULT_LABELS = {"no contest", "nc"}
DATE_FORMATS = [
    "%Y-%m-%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
]


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def parse_label_from_text(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s+[A-Za-z][A-Za-z .]+:\s|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


class PrefightDatasetBuilder:
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
        self.fight_context_cache_path = self.cache_dir / "ufc_fight_context_cache.json"
        self.static_profile_cache_path = self.cache_dir / "ufc_static_profile_cache.json"
        self.event_date_cache_path = self.cache_dir / "ufc_event_date_cache.json"
        self.fighter_history_cache_path = self.cache_dir / "ufc_fighter_history_cache.json"

        self.fight_context_cache: Dict[str, Dict] = self._load_json_cache(self.fight_context_cache_path)
        self.static_profile_cache: Dict[str, Dict] = self._load_json_cache(self.static_profile_cache_path)
        self.event_date_cache: Dict[str, str] = self._load_json_cache(self.event_date_cache_path)
        self.fighter_history_cache: Dict[str, List[str]] = self._load_json_cache(self.fighter_history_cache_path)
        self.event_name_date_cache = self._build_event_name_date_cache()
        self.prepared_fight_cache: Dict[str, Optional[Dict[str, object]]] = {}

    @staticmethod
    def _load_json_cache(path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_json_cache(path: Path, payload: Dict) -> None:
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _persist_caches(self) -> None:
        self._save_json_cache(self.fight_context_cache_path, self.fight_context_cache)
        self._save_json_cache(self.static_profile_cache_path, self.static_profile_cache)
        self._save_json_cache(self.event_date_cache_path, self.event_date_cache)
        self._save_json_cache(self.fighter_history_cache_path, self.fighter_history_cache)

    def _build_event_name_date_cache(self) -> Dict[str, str]:
        event_dates: Dict[str, List[pd.Timestamp]] = {}
        for context in self.fight_context_cache.values():
            if not isinstance(context, dict):
                continue
            event_name = str(context.get("event", "")).strip()
            parsed_date = self._parse_date_value(context.get("date"))
            if event_name and parsed_date is not None:
                event_dates.setdefault(event_name, []).append(parsed_date)
        resolved = {}
        for event_name, values in event_dates.items():
            counts = Counter(v.strftime("%Y-%m-%d") for v in values)
            resolved[event_name] = counts.most_common(1)[0][0]
        return resolved

    @staticmethod
    def _normalize_name(name: object) -> str:
        return re.sub(r"\s+", " ", str(name or "")).strip().lower()

    def _fighter_lookup_key(self, fighter_name: object, profile_url: object) -> str:
        profile = str(profile_url or "").strip()
        if profile:
            return f"url::{profile}"
        normalized_name = self._normalize_name(fighter_name)
        return f"name::{normalized_name}" if normalized_name else ""

    def _cached_history_urls_for_fighter(self, fighter_name: object, profile_url: object) -> List[str]:
        normalized_name = self._normalize_name(fighter_name)
        profile = str(profile_url or "").strip()
        urls: List[str] = []
        seen = set()
        for fight_url, context in self.fight_context_cache.items():
            if not isinstance(context, dict):
                continue
            matched = False
            for corner in ("red", "blue"):
                context_profile = str(context.get(f"{corner}_profile_url", "")).strip()
                context_name = self._normalize_name(context.get(f"{corner}_fighter", ""))
                if profile and context_profile and context_profile == profile:
                    matched = True
                    break
                if normalized_name and context_name and context_name == normalized_name:
                    matched = True
                    break
            if matched and fight_url not in seen:
                seen.add(fight_url)
                urls.append(fight_url)
        return urls

    def _history_urls_for_fighter(self, fighter_name: object, profile_url: object) -> List[str]:
        urls: List[str] = []
        seen = set()
        for fight_url in self._cached_history_urls_for_fighter(fighter_name, profile_url):
            fight_url = str(fight_url).strip()
            if fight_url and fight_url not in seen:
                seen.add(fight_url)
                urls.append(fight_url)
        profile = str(profile_url or "").strip()
        if profile:
            for fight_url in self._extract_fighter_history_urls(profile):
                fight_url = str(fight_url).strip()
                if fight_url and fight_url not in seen:
                    seen.add(fight_url)
                    urls.append(fight_url)
        return urls

    @staticmethod
    def _enqueue_fighter(queue: deque, queued_keys: set[str], fighter_name: object, profile_url: object, key: str) -> None:
        if not key or key in queued_keys:
            return
        queued_keys.add(key)
        queue.append((str(fighter_name or "").strip(), str(profile_url or "").strip()))

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
            "li.b-list__box-list-item_type_block",
        ]
        for selector in selectors:
            for node in soup.select(selector):
                text = " ".join(node.stripped_strings)
                if text.lower().startswith(target):
                    return text.split(":", 1)[1].strip()
        return parse_label_from_text(" ".join(soup.stripped_strings), label)

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            coerced = pd.to_numeric(value, errors="coerce")
            return float(coerced) if pd.notna(coerced) else float("nan")
        except Exception:
            return float("nan")

    @staticmethod
    def _parse_pair_metric(text: str) -> Tuple[float, float]:
        match = re.search(r"(\d+)\s+of\s+(\d+)", str(text))
        if not match:
            return float("nan"), float("nan")
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _text_to_float(text: str) -> float:
        try:
            value = pd.to_numeric(str(text).strip(), errors="coerce")
            return float(value) if pd.notna(value) else float("nan")
        except Exception:
            return float("nan")

    def _parse_fight_stats_from_table(self, soup: BeautifulSoup) -> Dict[str, float]:
        stats = {
            "red_sig_landed": float("nan"),
            "red_sig_attempted": float("nan"),
            "blue_sig_landed": float("nan"),
            "blue_sig_attempted": float("nan"),
            "red_td_landed": float("nan"),
            "red_td_attempted": float("nan"),
            "blue_td_landed": float("nan"),
            "blue_td_attempted": float("nan"),
            "red_sub_att": float("nan"),
            "blue_sub_att": float("nan"),
        }
        for row in soup.select("tr.b-fight-details__table-row"):
            tds = row.select("td")
            if len(tds) < 8:
                continue
            sig_values = [p.get_text(" ", strip=True) for p in tds[2].select("p") if p.get_text(strip=True)]
            td_values = [p.get_text(" ", strip=True) for p in tds[5].select("p") if p.get_text(strip=True)]
            sub_values = [p.get_text(" ", strip=True) for p in tds[7].select("p") if p.get_text(strip=True)]
            if len(sig_values) >= 2:
                stats["red_sig_landed"], stats["red_sig_attempted"] = self._parse_pair_metric(sig_values[0])
                stats["blue_sig_landed"], stats["blue_sig_attempted"] = self._parse_pair_metric(sig_values[1])
            if len(td_values) >= 2:
                stats["red_td_landed"], stats["red_td_attempted"] = self._parse_pair_metric(td_values[0])
                stats["blue_td_landed"], stats["blue_td_attempted"] = self._parse_pair_metric(td_values[1])
            if len(sub_values) >= 2:
                stats["red_sub_att"] = self._text_to_float(sub_values[0])
                stats["blue_sub_att"] = self._text_to_float(sub_values[1])
            if any(pd.notna(v) for v in stats.values()):
                break
        return stats

    def _context_needs_refresh(self, context: Dict[str, object]) -> bool:
        if not context:
            return True
        required_keys = [
            "event",
            "event_url",
            "date",
            "red_profile_url",
            "blue_profile_url",
            "red_sig_landed",
            "red_td_landed",
            "red_sub_att",
        ]
        for key in required_keys:
            value = context.get(key)
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            if key.startswith("red_") and not isinstance(value, str) and pd.isna(pd.to_numeric(value, errors="coerce")):
                return True
        return False

    def _parse_date_value(self, value: object) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.normalize()
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a", "--"}:
            return None
        for fmt in DATE_FORMATS:
            try:
                return pd.Timestamp(datetime.strptime(text, fmt).date())
            except ValueError:
                continue
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).normalize()

    @staticmethod
    def _format_date_value(value: Optional[pd.Timestamp]) -> str:
        if value is None or pd.isna(value):
            return ""
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    def _calculate_duration_seconds(self, duration_value: object, round_value: object, time_value: object) -> float:
        duration = self._safe_float(duration_value)
        if pd.notna(duration):
            return duration
        round_number = pd.to_numeric(round_value, errors="coerce")
        time_text = str(time_value).strip()
        if pd.isna(round_number) or not time_text:
            return float("nan")
        match = re.fullmatch(r"(\d+):(\d{2})", time_text)
        if not match:
            return float("nan")
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float((int(round_number) - 1) * 300 + minutes * 60 + seconds)

    def _extract_fight_context(self, fight_url: str) -> Dict[str, object]:
        cached_context = dict(self.fight_context_cache.get(fight_url, {}))
        if cached_context and not self._context_needs_refresh(cached_context):
            return cached_context
        soup = self._get_soup(fight_url)
        if soup is None:
            return cached_context
        context: Dict[str, object] = {
            "fight_url": fight_url,
            "red_fighter": "",
            "blue_fighter": "",
            "winner": "",
            "red_profile_url": "",
            "blue_profile_url": "",
            "event": "",
            "event_url": "",
            "date": "",
            "weight_class": "",
            "method": "",
            "round": "",
            "time": "",
            "duration_seconds": float("nan"),
            "red_sig_landed": float("nan"),
            "red_sig_attempted": float("nan"),
            "blue_sig_landed": float("nan"),
            "blue_sig_attempted": float("nan"),
            "red_td_landed": float("nan"),
            "red_td_attempted": float("nan"),
            "blue_td_landed": float("nan"),
            "blue_td_attempted": float("nan"),
            "red_sub_att": float("nan"),
            "blue_sub_att": float("nan"),
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
            if any("green" in c.lower() for c in status_icon.get("class", [])):
                winner_link = person.select_one("a.b-link.b-fight-details__person-link")
                if not winner_link:
                    winner_link = person.select_one("h3.b-fight-details__person-name a")
                if winner_link:
                    context["winner"] = winner_link.get_text(strip=True)
                break

        title = soup.select_one("span.b-content__title-highlight")
        if title:
            context["event"] = title.get_text(strip=True)
        event_link = soup.find("a", href=re.compile(r"/event-details/"))
        if event_link:
            if not context["event"]:
                context["event"] = event_link.get_text(" ", strip=True)
            context["event_url"] = event_link.get("href", "").strip()

        wc_tag = soup.select_one("i.b-fight-details__fight-title")
        if wc_tag:
            context["weight_class"] = wc_tag.get_text(" ", strip=True)
        context["method"] = self._extract_labeled_value(soup, "Method")
        context["round"] = self._extract_labeled_value(soup, "Round")
        context["time"] = self._extract_labeled_value(soup, "Time")
        context["date"] = self._extract_labeled_value(soup, "Date")
        context["duration_seconds"] = self._calculate_duration_seconds(
            context.get("duration_seconds"),
            context.get("round"),
            context.get("time"),
        )
        context.update(self._parse_fight_stats_from_table(soup))
        self.fight_context_cache[fight_url] = dict(context)
        return context

    def _extract_static_profile(self, profile_url: str) -> Dict[str, object]:
        if not profile_url:
            return {}
        if profile_url in self.static_profile_cache:
            return dict(self.static_profile_cache[profile_url])
        soup = self._get_soup(profile_url)
        if soup is None:
            return {}

        raw: Dict[str, object] = {}
        for li in soup.select("li.b-list__box-list-item"):
            text = " ".join(li.stripped_strings)
            if ":" not in text:
                continue
            label, value = text.split(":", 1)
            raw[normalize_key(label)] = value.strip()

        normalized = normalize_profile_input(raw)
        static_profile = {
            "height_in": normalized.get("height_in"),
            "weight_lbs": normalized.get("weight_lbs"),
            "reach_in": normalized.get("reach_in"),
            "stance": normalize_stance(raw.get("stance") or normalized.get("stance")),
            "dob": str(raw.get("dob", "")).strip(),
        }
        self.static_profile_cache[profile_url] = static_profile
        return static_profile

    def _extract_fighter_history_urls(self, profile_url: str) -> List[str]:
        if not profile_url:
            return []
        cached = self.fighter_history_cache.get(profile_url)
        if isinstance(cached, list) and cached:
            return [str(url).strip() for url in cached if str(url).strip()]

        soup = self._get_soup(profile_url)
        if soup is None:
            return [str(url).strip() for url in cached] if isinstance(cached, list) else []

        fight_urls: List[str] = []
        for row in soup.select("tr[data-link]"):
            data_link = str(row.get("data-link", "")).strip()
            if "/fight-details/" in data_link and data_link not in fight_urls:
                fight_urls.append(data_link)
        for anchor in soup.select("a[href*='/fight-details/']"):
            href = str(anchor.get("href", "")).strip()
            if href and href not in fight_urls:
                fight_urls.append(href)

        self.fighter_history_cache[profile_url] = fight_urls
        return fight_urls

    def _extract_event_date_from_page(self, event_url: str) -> str:
        if not event_url:
            return ""
        soup = self._get_soup(event_url)
        if soup is None:
            return ""
        date_text = self._extract_labeled_value(soup, "Date")
        if date_text:
            return date_text
        match = re.search(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", " ".join(soup.stripped_strings))
        return match.group(1) if match else ""

    def _resolve_event_date(self, event_name: str, event_url: str, direct_date: object) -> Optional[pd.Timestamp]:
        parsed = self._parse_date_value(direct_date)
        if parsed is not None:
            return parsed
        if event_url:
            parsed = self._parse_date_value(self.event_date_cache.get(event_url))
            if parsed is not None:
                return parsed
        if event_name:
            parsed = self._parse_date_value(self.event_name_date_cache.get(event_name))
            if parsed is not None:
                return parsed
        if event_url:
            live_date = self._extract_event_date_from_page(event_url)
            parsed = self._parse_date_value(live_date)
            if parsed is not None:
                self.event_date_cache[event_url] = self._format_date_value(parsed)
                if event_name:
                    self.event_name_date_cache[event_name] = self._format_date_value(parsed)
                return parsed
        return None

    @staticmethod
    def _method_bucket(method: object) -> str:
        text = str(method or "").strip().lower()
        if not text:
            return "other"
        if "decision" in text:
            return "decision"
        if "submission" in text:
            return "submission"
        if re.search(r"\bko\b|\btko\b|knockout", text):
            return "ko_tko"
        return "other"

    @staticmethod
    def _score_result(winner: object, red_name: str, blue_name: str) -> Optional[str]:
        winner_text = str(winner or "").strip().lower()
        red = str(red_name or "").strip().lower()
        blue = str(blue_name or "").strip().lower()
        if not red or not blue:
            return None
        if winner_text == red:
            return "red"
        if winner_text == blue:
            return "blue"
        if winner_text in DRAW_LABELS:
            return "draw"
        if winner_text in SKIP_RESULT_LABELS or not winner_text:
            return None
        return None

    @staticmethod
    def _empty_fighter_state() -> Dict[str, object]:
        return {
            "bouts": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "last_fight_date": None,
            "recent_results": [],
            "seconds": 0.0,
            "sig_landed": 0.0,
            "sig_attempted": 0.0,
            "sig_absorbed_landed": 0.0,
            "sig_absorbed_attempted": 0.0,
            "td_landed": 0.0,
            "td_attempted": 0.0,
            "td_allowed_landed": 0.0,
            "td_allowed_attempted": 0.0,
            "sub_att": 0.0,
            "finish_wins": 0,
            "ko_tko_wins": 0,
            "submission_wins": 0,
            "decision_wins": 0,
            "elo": ELO_INITIAL,
            "opponent_elo_sum": 0.0,
            "opponents_faced": 0,
        }

    @staticmethod
    def _recent_form(results: List[float], window: int) -> float:
        if len(results) < window:
            return 0.5
        values = results[-window:]
        return float(sum(values) / len(values)) if values else 0.5

    def _age_at_fight(self, dob_value: object, fight_date: pd.Timestamp) -> float:
        if not dob_value:
            return float("nan")
        age = parse_age(dob_value, today=pd.Timestamp(fight_date).date())
        return float(age) if pd.notna(age) else float("nan")

    def _static_missing(self, static_profile: Dict[str, object]) -> bool:
        if not static_profile:
            return True
        for key in ("height_in", "weight_lbs", "reach_in", "stance", "dob"):
            value = static_profile.get(key)
            if key == "stance":
                if value and str(value).strip().lower() != "unknown":
                    return False
            else:
                numeric = pd.to_numeric(value, errors="coerce")
                if pd.notna(numeric) or (value and str(value).strip()):
                    return False
        return True

    def _snapshot_prefight_state(
        self,
        fighter_name: str,
        profile_url: str,
        fight_date: pd.Timestamp,
        fighter_states: Dict[str, Dict[str, object]],
        missing_static_profiles: set[str],
    ) -> Dict[str, object]:
        state = fighter_states.setdefault(fighter_name, self._empty_fighter_state())
        static_profile = self._extract_static_profile(profile_url)
        if self._static_missing(static_profile):
            missing_static_profiles.add(fighter_name)
        seconds = float(state["seconds"])
        bouts = int(state["bouts"])
        opponent_avg_elo = (
            float(state["opponent_elo_sum"]) / int(state["opponents_faced"])
            if int(state["opponents_faced"]) > 0
            else ELO_INITIAL
        )
        last_fight_date = state.get("last_fight_date")
        days_since_last_fight = float("nan")
        if isinstance(last_fight_date, pd.Timestamp):
            days_since_last_fight = float((fight_date - last_fight_date).days)
        sig_attempted = float(state["sig_attempted"])
        sig_abs_attempted = float(state["sig_absorbed_attempted"])
        td_attempted = float(state["td_attempted"])
        td_allowed_attempted = float(state["td_allowed_attempted"])
        return {
            "height_in": self._safe_float(static_profile.get("height_in")),
            "weight_lbs": self._safe_float(static_profile.get("weight_lbs")),
            "reach_in": self._safe_float(static_profile.get("reach_in")),
            "stance": normalize_stance(static_profile.get("stance")),
            "age_at_fight": self._age_at_fight(static_profile.get("dob"), fight_date),
            "ufc_bouts_prior": bouts,
            "ufc_wins_prior": int(state["wins"]),
            "ufc_losses_prior": int(state["losses"]),
            "ufc_draws_prior": int(state["draws"]),
            "days_since_last_fight": days_since_last_fight,
            "recent_form_last3": self._recent_form(state["recent_results"], 3),
            "recent_form_last5": self._recent_form(state["recent_results"], 5),
            "sig_landed_per_min_prior": float(state["sig_landed"]) * 60.0 / seconds if seconds > 0 else float("nan"),
            "sig_absorbed_per_min_prior": float(state["sig_absorbed_landed"]) * 60.0 / seconds if seconds > 0 else float("nan"),
            "sig_acc_prior": float(state["sig_landed"]) / sig_attempted if sig_attempted > 0 else float("nan"),
            "sig_def_prior": 1.0 - float(state["sig_absorbed_landed"]) / sig_abs_attempted if sig_abs_attempted > 0 else float("nan"),
            "td_landed_per15_prior": float(state["td_landed"]) * 900.0 / seconds if seconds > 0 else float("nan"),
            "td_acc_prior": float(state["td_landed"]) / td_attempted if td_attempted > 0 else float("nan"),
            "td_def_prior": 1.0 - float(state["td_allowed_landed"]) / td_allowed_attempted if td_allowed_attempted > 0 else float("nan"),
            "sub_att_per15_prior": float(state["sub_att"]) * 900.0 / seconds if seconds > 0 else float("nan"),
            "finish_rate_prior": float(state["finish_wins"]) / bouts if bouts > 0 else float("nan"),
            "ko_tko_win_rate_prior": float(state["ko_tko_wins"]) / bouts if bouts > 0 else float("nan"),
            "submission_win_rate_prior": float(state["submission_wins"]) / bouts if bouts > 0 else float("nan"),
            "decision_win_rate_prior": float(state["decision_wins"]) / bouts if bouts > 0 else float("nan"),
            "elo_prior": float(state["elo"]),
            "opponent_avg_elo_prior": opponent_avg_elo,
        }
    def _update_fighter_state(
        self,
        state: Dict[str, object],
        result_score: float,
        fight_date: pd.Timestamp,
        own_sig_landed: float,
        own_sig_attempted: float,
        opp_sig_landed: float,
        opp_sig_attempted: float,
        own_td_landed: float,
        own_td_attempted: float,
        opp_td_landed: float,
        opp_td_attempted: float,
        own_sub_att: float,
        duration_seconds: float,
        method_bucket: str,
        opponent_pre_elo: float,
        new_elo: float,
    ) -> None:
        state["bouts"] = int(state["bouts"]) + 1
        if result_score == 1.0:
            state["wins"] = int(state["wins"]) + 1
            if method_bucket in {"ko_tko", "submission"}:
                state["finish_wins"] = int(state["finish_wins"]) + 1
            if method_bucket in {"ko_tko", "submission", "decision"}:
                state[f"{method_bucket}_wins"] = int(state[f"{method_bucket}_wins"]) + 1
        elif result_score == 0.5:
            state["draws"] = int(state["draws"]) + 1
        else:
            state["losses"] = int(state["losses"]) + 1
        state["recent_results"] = list(state["recent_results"]) + [result_score]
        state["seconds"] = float(state["seconds"]) + (duration_seconds if pd.notna(duration_seconds) else 0.0)
        state["sig_landed"] = float(state["sig_landed"]) + (own_sig_landed if pd.notna(own_sig_landed) else 0.0)
        state["sig_attempted"] = float(state["sig_attempted"]) + (own_sig_attempted if pd.notna(own_sig_attempted) else 0.0)
        state["sig_absorbed_landed"] = float(state["sig_absorbed_landed"]) + (opp_sig_landed if pd.notna(opp_sig_landed) else 0.0)
        state["sig_absorbed_attempted"] = float(state["sig_absorbed_attempted"]) + (opp_sig_attempted if pd.notna(opp_sig_attempted) else 0.0)
        state["td_landed"] = float(state["td_landed"]) + (own_td_landed if pd.notna(own_td_landed) else 0.0)
        state["td_attempted"] = float(state["td_attempted"]) + (own_td_attempted if pd.notna(own_td_attempted) else 0.0)
        state["td_allowed_landed"] = float(state["td_allowed_landed"]) + (opp_td_landed if pd.notna(opp_td_landed) else 0.0)
        state["td_allowed_attempted"] = float(state["td_allowed_attempted"]) + (opp_td_attempted if pd.notna(opp_td_attempted) else 0.0)
        state["sub_att"] = float(state["sub_att"]) + (own_sub_att if pd.notna(own_sub_att) else 0.0)
        state["last_fight_date"] = fight_date
        state["opponent_elo_sum"] = float(state["opponent_elo_sum"]) + opponent_pre_elo
        state["opponents_faced"] = int(state["opponents_faced"]) + 1
        state["elo"] = float(new_elo)

    @staticmethod
    def _elo_expected(elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def _prepare_fight_from_url(self, fight_url: str) -> Optional[Dict[str, object]]:
        fight_url = str(fight_url).strip()
        if not fight_url:
            return None
        if fight_url in self.prepared_fight_cache:
            cached = self.prepared_fight_cache[fight_url]
            return dict(cached) if isinstance(cached, dict) else None

        context = self._extract_fight_context(fight_url)
        red_name = str(context.get("red_fighter", "")).strip()
        blue_name = str(context.get("blue_fighter", "")).strip()
        winner = str(context.get("winner", "")).strip()
        event_name = str(context.get("event", "")).strip()
        event_url = str(context.get("event_url", "")).strip()
        fight_date = self._resolve_event_date(event_name, event_url, context.get("date"))
        if not red_name or not blue_name or fight_date is None:
            self.prepared_fight_cache[fight_url] = None
            return None

        result_flag = self._score_result(winner, red_name, blue_name)
        if result_flag is None:
            self.prepared_fight_cache[fight_url] = None
            return None

        prepared = {
            "fight_url": fight_url,
            "event": event_name,
            "event_url": event_url,
            "date": fight_date,
            "weight_class": str(context.get("weight_class", "")).strip(),
            "method": str(context.get("method", "")).strip(),
            "round": str(context.get("round", "")).strip(),
            "time": str(context.get("time", "")).strip(),
            "duration_seconds": self._calculate_duration_seconds(
                context.get("duration_seconds"),
                context.get("round"),
                context.get("time"),
            ),
            "red_fighter": red_name,
            "blue_fighter": blue_name,
            "winner": winner,
            "result_flag": result_flag,
            "red_profile_url": str(context.get("red_profile_url", "")).strip(),
            "blue_profile_url": str(context.get("blue_profile_url", "")).strip(),
            "red_sig_landed": self._safe_float(context.get("red_sig_landed")),
            "red_sig_attempted": self._safe_float(context.get("red_sig_attempted")),
            "blue_sig_landed": self._safe_float(context.get("blue_sig_landed")),
            "blue_sig_attempted": self._safe_float(context.get("blue_sig_attempted")),
            "red_td_landed": self._safe_float(context.get("red_td_landed")),
            "red_td_attempted": self._safe_float(context.get("red_td_attempted")),
            "blue_td_landed": self._safe_float(context.get("blue_td_landed")),
            "blue_td_attempted": self._safe_float(context.get("blue_td_attempted")),
            "red_sub_att": self._safe_float(context.get("red_sub_att")),
            "blue_sub_att": self._safe_float(context.get("blue_sub_att")),
        }
        self.prepared_fight_cache[fight_url] = dict(prepared)
        return prepared

    def _build_prepared_fights_for_urls(
        self,
        fight_urls: Iterable[str],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> List[Dict[str, object]]:
        cutoff = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else None
        prepared_fights: List[Dict[str, object]] = []
        seen_urls = set()
        for fight_url in fight_urls:
            fight_url = str(fight_url).strip()
            if not fight_url or fight_url in seen_urls:
                continue
            seen_urls.add(fight_url)
            prepared = self._prepare_fight_from_url(fight_url)
            if prepared is None:
                continue
            if cutoff is not None and prepared["date"] > cutoff:
                continue
            prepared_fights.append(dict(prepared))
        prepared_fights.sort(key=lambda item: (item["date"], item["fight_url"]))
        return prepared_fights

    def _advance_states_with_fight(
        self,
        fight: Dict[str, object],
        fighter_states: Dict[str, Dict[str, object]],
    ) -> None:
        fight_date = fight["date"]
        red_state = fighter_states.setdefault(fight["red_fighter"], self._empty_fighter_state())
        blue_state = fighter_states.setdefault(fight["blue_fighter"], self._empty_fighter_state())
        red_pre_elo = float(red_state["elo"])
        blue_pre_elo = float(blue_state["elo"])
        red_expected = self._elo_expected(red_pre_elo, blue_pre_elo)
        blue_expected = self._elo_expected(blue_pre_elo, red_pre_elo)
        if fight["result_flag"] == "red":
            red_score, blue_score = 1.0, 0.0
        elif fight["result_flag"] == "blue":
            red_score, blue_score = 0.0, 1.0
        else:
            red_score = blue_score = 0.5
        red_new_elo = red_pre_elo + ELO_K * (red_score - red_expected)
        blue_new_elo = blue_pre_elo + ELO_K * (blue_score - blue_expected)
        method_bucket = self._method_bucket(fight.get("method"))
        self._update_fighter_state(
            red_state,
            red_score,
            fight_date,
            float(fight["red_sig_landed"]),
            float(fight["red_sig_attempted"]),
            float(fight["blue_sig_landed"]),
            float(fight["blue_sig_attempted"]),
            float(fight["red_td_landed"]),
            float(fight["red_td_attempted"]),
            float(fight["blue_td_landed"]),
            float(fight["blue_td_attempted"]),
            float(fight["red_sub_att"]),
            float(fight["duration_seconds"]),
            method_bucket,
            blue_pre_elo,
            red_new_elo,
        )
        self._update_fighter_state(
            blue_state,
            blue_score,
            fight_date,
            float(fight["blue_sig_landed"]),
            float(fight["blue_sig_attempted"]),
            float(fight["red_sig_landed"]),
            float(fight["red_sig_attempted"]),
            float(fight["blue_td_landed"]),
            float(fight["blue_td_attempted"]),
            float(fight["red_td_landed"]),
            float(fight["red_td_attempted"]),
            float(fight["blue_sub_att"]),
            float(fight["duration_seconds"]),
            method_bucket,
            red_pre_elo,
            blue_new_elo,
        )

    def _build_exact_history_universe(
        self,
        target_fights: List[Dict[str, object]],
        cutoff_date: pd.Timestamp,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        fight_universe: Dict[str, Dict[str, object]] = {
            str(fight["fight_url"]): dict(fight) for fight in target_fights if str(fight.get("fight_url", "")).strip()
        }
        queue = deque()
        queued_keys: set[str] = set()
        processed_fighter_keys: set[str] = set()
        fighters_seen_by_name: set[str] = set()
        fighters_seen_by_profile: set[str] = set()
        for fight in target_fights:
            for corner in ("red", "blue"):
                fighter_name = str(fight.get(f"{corner}_fighter", "")).strip()
                profile_url = str(fight.get(f"{corner}_profile_url", "")).strip()
                key = self._fighter_lookup_key(fighter_name, profile_url)
                self._enqueue_fighter(queue, queued_keys, fighter_name, profile_url, key)

        history_url_count = 0
        while queue:
            fighter_name, profile_url = queue.popleft()
            key = self._fighter_lookup_key(fighter_name, profile_url)
            if not key or key in processed_fighter_keys:
                continue
            processed_fighter_keys.add(key)
            normalized_name = self._normalize_name(fighter_name)
            if normalized_name:
                fighters_seen_by_name.add(normalized_name)
            if profile_url:
                fighters_seen_by_profile.add(profile_url)

            history_urls = self._history_urls_for_fighter(fighter_name, profile_url)
            history_url_count += len(history_urls)
            for fight_url in history_urls:
                prepared = self._prepare_fight_from_url(fight_url)
                if prepared is None:
                    continue
                if prepared["date"] > cutoff_date:
                    continue
                fight_key = str(prepared["fight_url"])
                if fight_key not in fight_universe:
                    fight_universe[fight_key] = dict(prepared)
                for corner in ("red", "blue"):
                    other_name = str(prepared.get(f"{corner}_fighter", "")).strip()
                    other_profile_url = str(prepared.get(f"{corner}_profile_url", "")).strip()
                    other_key = self._fighter_lookup_key(other_name, other_profile_url)
                    self._enqueue_fighter(queue, queued_keys, other_name, other_profile_url, other_key)

        ordered_fights = sorted(fight_universe.values(), key=lambda item: (item["date"], item["fight_url"]))
        metadata = {
            "history_strategy": "exact",
            "history_cutoff_date": self._format_date_value(cutoff_date),
            "historical_universe_fights_processed": len(ordered_fights),
            "closure_unique_fighters": len(fighters_seen_by_name),
            "closure_unique_profile_urls": len(fighters_seen_by_profile),
            "closure_fighter_expansions": len(processed_fighter_keys),
            "history_urls_examined": history_url_count,
        }
        return ordered_fights, metadata

    def build_live_prefight_profiles(
        self,
        fighters: Dict[str, str],
        as_of_date: Optional[object] = None,
    ) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
        target_date = self._parse_date_value(as_of_date) or pd.Timestamp.now().normalize()
        target_fights: List[Dict[str, object]] = []
        for fighter_name, profile_url in fighters.items():
            for fight_url in self._history_urls_for_fighter(fighter_name, profile_url):
                prepared = self._prepare_fight_from_url(fight_url)
                if prepared is None:
                    continue
                if prepared["date"] > target_date:
                    continue
                target_fights.append(dict(prepared))
        deduped_targets = {str(fight["fight_url"]): fight for fight in target_fights}
        prepared_fights, universe_meta = self._build_exact_history_universe(
            list(deduped_targets.values()),
            cutoff_date=target_date,
        )

        fighter_states: Dict[str, Dict[str, object]] = {}
        missing_static_profiles: set[str] = set()
        for fight in prepared_fights:
            self._advance_states_with_fight(fight, fighter_states)

        snapshots = {}
        requested_fight_counts: Dict[str, int] = {}
        for fighter_name, profile_url in fighters.items():
            requested_fight_counts[fighter_name] = len(self._history_urls_for_fighter(fighter_name, profile_url))
            snapshots[fighter_name] = self._snapshot_prefight_state(
                fighter_name=fighter_name,
                profile_url=profile_url,
                fight_date=target_date,
                fighter_states=fighter_states,
                missing_static_profiles=missing_static_profiles,
            )

        self._persist_caches()
        metadata = {
            "as_of_date": self._format_date_value(target_date),
            "global_fights_processed": len(prepared_fights),
            "requested_fight_counts": requested_fight_counts,
            "fighters_missing_static_profile_data": sorted(missing_static_profiles),
            **universe_meta,
        }
        return snapshots, metadata

    def build_prefight(
        self,
        input_csv: str,
        output_csv: str,
        manifest_out: str,
        max_fights: Optional[int] = None,
        history_strategy: str = "exact",
    ) -> pd.DataFrame:
        src = pd.read_csv(input_csv)
        url_col = pick_column(src.columns, ["Fight_URL", "fight_url", "url"])
        red_col = pick_column(src.columns, ["Red", "red", "red_fighter"])
        blue_col = pick_column(src.columns, ["Blue", "blue", "blue_fighter"])
        winner_col = pick_column(src.columns, ["Winner", "winner"])
        event_col = pick_column(src.columns, ["Event", "event"])
        event_url_col = pick_column(src.columns, ["Event_URL", "event_url"])
        event_date_col = pick_column(src.columns, ["Event_Date", "event_date", "date"])
        if not url_col:
            raise ValueError(f"Missing fight URL column in {input_csv}.")

        active_history_strategy = str(history_strategy or "exact").strip().lower()
        if active_history_strategy not in {"exact", "input_window_only"}:
            raise ValueError("history_strategy must be 'exact' or 'input_window_only'.")

        manifest = {
            "build_timestamp": datetime.now().isoformat(timespec="seconds"),
            "build_mode": "prefight_v1",
            "input_file": input_csv,
            "total_fights_seen": int(len(src)),
            "total_rows_emitted": 0,
            "dropped_fights_by_reason": {},
            "fights_missing_date": [],
            "fighters_missing_static_profile_data": [],
            "schema_version": "prefight_v1",
            "history_strategy": active_history_strategy,
        }
        drop_reasons: Counter[str] = Counter()
        missing_date_urls: List[str] = []
        missing_static_profiles: set[str] = set()
        source_fights: List[Dict[str, object]] = []

        for idx, row in src.iterrows():
            fight_url = str(row.get(url_col, "")).strip()
            if not fight_url:
                drop_reasons["missing_fight_url"] += 1
                continue

            prepared = self._prepare_fight_from_url(fight_url)
            if prepared is not None:
                source_fights.append(dict(prepared))
                if (idx + 1) % 25 == 0 or (idx + 1) == len(src):
                    print(f"Prepared {idx + 1}/{len(src)} fights for prefight dataset")
                continue

            context = self._extract_fight_context(fight_url)
            red_name = str(context.get("red_fighter") or row.get(red_col, "")).strip()
            blue_name = str(context.get("blue_fighter") or row.get(blue_col, "")).strip()
            winner = str(context.get("winner") or row.get(winner_col, "")).strip()
            event_name = str(context.get("event") or row.get(event_col, "")).strip()
            event_url = str(context.get("event_url") or row.get(event_url_col, "")).strip()
            direct_date = context.get("date") or row.get(event_date_col, "")
            fight_date = self._resolve_event_date(event_name, event_url, direct_date)
            if not red_name or not blue_name:
                drop_reasons["missing_fighter_name"] += 1
                continue
            if fight_date is None:
                drop_reasons["missing_date"] += 1
                missing_date_urls.append(fight_url)
                continue
            result_flag = self._score_result(winner, red_name, blue_name)
            if result_flag is None:
                drop_reasons["missing_or_unusable_winner"] += 1
                continue
            prepared_from_row = {
                "fight_url": fight_url,
                "event": event_name,
                "event_url": event_url,
                "date": fight_date,
                "weight_class": str(context.get("weight_class", "")).strip(),
                "method": str(context.get("method", "")).strip(),
                "round": str(context.get("round", "")).strip(),
                "time": str(context.get("time", "")).strip(),
                "duration_seconds": self._calculate_duration_seconds(
                    context.get("duration_seconds"),
                    context.get("round"),
                    context.get("time"),
                ),
                "red_fighter": red_name,
                "blue_fighter": blue_name,
                "winner": winner,
                "result_flag": result_flag,
                "red_profile_url": str(context.get("red_profile_url", "")).strip(),
                "blue_profile_url": str(context.get("blue_profile_url", "")).strip(),
                "red_sig_landed": self._safe_float(context.get("red_sig_landed")),
                "red_sig_attempted": self._safe_float(context.get("red_sig_attempted")),
                "blue_sig_landed": self._safe_float(context.get("blue_sig_landed")),
                "blue_sig_attempted": self._safe_float(context.get("blue_sig_attempted")),
                "red_td_landed": self._safe_float(context.get("red_td_landed")),
                "red_td_attempted": self._safe_float(context.get("red_td_attempted")),
                "blue_td_landed": self._safe_float(context.get("blue_td_landed")),
                "blue_td_attempted": self._safe_float(context.get("blue_td_attempted")),
                "red_sub_att": self._safe_float(context.get("red_sub_att")),
                "blue_sub_att": self._safe_float(context.get("blue_sub_att")),
            }
            self.prepared_fight_cache[fight_url] = dict(prepared_from_row)
            source_fights.append(prepared_from_row)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(src):
                print(f"Prepared {idx + 1}/{len(src)} fights for prefight dataset")

        source_fights.sort(key=lambda item: (item["date"], item["fight_url"]))
        target_fights = list(source_fights)
        if max_fights is not None and int(max_fights) > 0:
            target_fights = source_fights[-int(max_fights):]
        if not target_fights:
            raise ValueError("No usable prefight target fights were available after filtering.")
        target_urls = {fight["fight_url"] for fight in target_fights}

        if active_history_strategy == "exact":
            cutoff_date = max(fight["date"] for fight in target_fights)
            history_fights, history_meta = self._build_exact_history_universe(target_fights, cutoff_date=cutoff_date)
        else:
            history_fights = list(source_fights)
            history_meta = {
                "history_strategy": "input_window_only",
                "history_cutoff_date": self._format_date_value(max(fight["date"] for fight in target_fights)),
                "historical_universe_fights_processed": len(history_fights),
                "closure_unique_fighters": len(
                    {
                        self._normalize_name(fight.get("red_fighter"))
                        for fight in history_fights
                        if self._normalize_name(fight.get("red_fighter"))
                    }
                    | {
                        self._normalize_name(fight.get("blue_fighter"))
                        for fight in history_fights
                        if self._normalize_name(fight.get("blue_fighter"))
                    }
                ),
                "closure_unique_profile_urls": len(
                    {
                        str(fight.get("red_profile_url", "")).strip()
                        for fight in history_fights
                        if str(fight.get("red_profile_url", "")).strip()
                    }
                    | {
                        str(fight.get("blue_profile_url", "")).strip()
                        for fight in history_fights
                        if str(fight.get("blue_profile_url", "")).strip()
                    }
                ),
                "closure_fighter_expansions": None,
                "history_urls_examined": None,
            }

        fighter_states: Dict[str, Dict[str, object]] = {}
        output_rows: List[Dict[str, object]] = []
        total_history = len(history_fights)
        for idx, fight in enumerate(history_fights, start=1):
            fight_date = fight["date"]
            red_snapshot = self._snapshot_prefight_state(
                fight["red_fighter"],
                str(fight.get("red_profile_url", "")),
                fight_date,
                fighter_states,
                missing_static_profiles,
            )
            blue_snapshot = self._snapshot_prefight_state(
                fight["blue_fighter"],
                str(fight.get("blue_profile_url", "")),
                fight_date,
                fighter_states,
                missing_static_profiles,
            )
            if fight["fight_url"] in target_urls:
                row = {
                    "fight_url": fight["fight_url"],
                    "event": fight["event"],
                    "date": self._format_date_value(fight_date),
                    "weight_class": fight["weight_class"],
                    "method": fight["method"],
                    "round": fight["round"],
                    "time": fight["time"],
                    "duration_seconds": fight["duration_seconds"],
                    "red_fighter": fight["red_fighter"],
                    "blue_fighter": fight["blue_fighter"],
                    "winner": fight["winner"],
                }
                for field, value in red_snapshot.items():
                    row[f"red_{field}"] = value
                for field, value in blue_snapshot.items():
                    row[f"blue_{field}"] = value
                output_rows.append(row)

            self._advance_states_with_fight(fight, fighter_states)
            if idx % 25 == 0 or idx == total_history:
                print(f"Built prefight rows from {idx}/{total_history} chronological fights")

        output_df = pd.DataFrame(output_rows)
        ordered_cols = ["fight_url", "event", "date", "weight_class", "method", "round", "time", "duration_seconds", "red_fighter", "blue_fighter", "winner"]
        for corner in ("red", "blue"):
            ordered_cols.extend(f"{corner}_{field}" for field in PREFIGHT_STATIC_FIELDS + PREFIGHT_PRIOR_FIELDS)
        keep_cols = [col for col in ordered_cols if col in output_df.columns]
        remaining_cols = [col for col in output_df.columns if col not in keep_cols]
        output_df = output_df[keep_cols + remaining_cols]
        output_df.to_csv(output_csv, index=False)

        manifest["total_rows_emitted"] = int(len(output_df))
        manifest["target_fights_selected"] = int(len(target_fights))
        manifest["source_prepared_fights"] = int(len(source_fights))
        manifest["dropped_fights_by_reason"] = dict(drop_reasons)
        manifest["fights_missing_date"] = missing_date_urls
        manifest["fighters_missing_static_profile_data"] = sorted(missing_static_profiles)
        manifest.update(history_meta)
        Path(manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._persist_caches()
        print(f"Saved {len(output_df)} prefight rows to {output_csv}")
        print(f"Saved prefight manifest to {manifest_out}")
        print(
            f"Prefight history strategy: {active_history_strategy} | "
            f"target fights: {len(target_fights)} | history universe fights: {len(history_fights)}"
        )
        if drop_reasons:
            print(f"Prefight build summary: {dict(drop_reasons)}")
        return output_df
