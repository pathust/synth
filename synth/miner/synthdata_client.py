"""
synthdata_client.py

Client for api.synthdata.co — public API (no auth required).

Endpoints covered:
    /validation/scores/historical   (CRPS + prompt_score per miner)
    /validation/scores/latest       (latest snapshot)
    /v2/leaderboard/historical      (neuron rewards over time)
    /v2/leaderboard/latest          (current rewards snapshot)
    /leaderboard/historical         (full metagraph: incentive, emission, stake, rank)
    /leaderboard/latest             (current metagraph snapshot)
    /validation/miner               (single-miner validation check)
"""

import time
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

VALID_ASSETS = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]

# Chunking: max days per request to avoid timeouts / huge payloads
_CHUNK_DAYS_SCORES = 7
_CHUNK_DAYS_LEADERBOARD = 3

_DEFAULT_TIMEOUT = 60
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0        # seconds, doubles each retry
_RATE_LIMIT_SLEEP = 1.0     # seconds between requests


def _iso(dt: datetime) -> str:
    """Datetime -> RFC 3339 string (always UTC, trailing Z)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO 8601 → aware UTC datetime."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


class SynthDataClient:
    BASE_URL = "https://api.synthdata.co"

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.timeout = timeout

    # ──────────────────────────────────────────────────────────────────
    # Low-level request with retry + rate-limit
    # ──────────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict) -> Optional[list | dict]:
        url = f"{self.BASE_URL}{path}"
        last_exc = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    time.sleep(_RATE_LIMIT_SLEEP)
                    return resp.json()

                if resp.status_code in (400, 404):
                    logger.warning("[SynthData] %d %s params=%s → %s",
                                   resp.status_code, path, params, resp.text[:200])
                    return None

                if resp.status_code >= 500:
                    logger.warning("[SynthData] %d on %s (attempt %d/%d)",
                                   resp.status_code, path, attempt, _MAX_RETRIES)
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                else:
                    logger.error("[SynthData] %d %s → %s",
                                 resp.status_code, path, resp.text[:200])
                    return None

            except requests.exceptions.RequestException as exc:
                logger.warning("[SynthData] Request error %s (attempt %d/%d): %s",
                               path, attempt, _MAX_RETRIES, exc)
                last_exc = exc

            backoff = _RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(backoff)

        logger.error("[SynthData] All %d retries exhausted for %s: %s",
                     _MAX_RETRIES, path, last_exc)
        return None

    # ──────────────────────────────────────────────────────────────────
    # Validation Scores
    # ──────────────────────────────────────────────────────────────────

    def get_historical_scores(
        self,
        asset: str,
        start_date: str,
        end_date: str,
        time_length: int = 86400,
        time_increment: int = 300,
        miner_uid: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch historical CRPS validation scores, auto-chunking by week.

        Returns flat list of dicts:
            {miner_uid, asset, prompt_score, scored_time, crps, time_length}
        """
        dt_start = _parse_date(start_date)
        dt_end = _parse_date(end_date)
        all_records: List[Dict] = []

        chunk_start = dt_start
        while chunk_start < dt_end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS_SCORES), dt_end)

            params: dict = {
                "from": _iso(chunk_start),
                "to": _iso(chunk_end),
                "asset": asset,
                "time_length": time_length,
                "time_increment": time_increment,
            }
            if miner_uid is not None:
                params["miner_uid"] = miner_uid

            logger.info("[SynthData] scores %s %s → %s (len=%d, inc=%d)",
                        asset, _iso(chunk_start), _iso(chunk_end),
                        time_length, time_increment)

            data = self._get("/validation/scores/historical", params)
            if data and isinstance(data, list):
                all_records.extend(data)
                logger.info("[SynthData]   → %d records (total %d)",
                            len(data), len(all_records))

            chunk_start = chunk_end

        return all_records

    def get_latest_scores(
        self,
        asset: str = "BTC",
        time_length: int = 86400,
        time_increment: int = 300,
    ) -> List[Dict]:
        """Fetch latest snapshot of validation scores for all miners."""
        data = self._get("/validation/scores/latest", {
            "asset": asset,
            "time_length": time_length,
            "time_increment": time_increment,
        })
        return data if isinstance(data, list) else []

    # ──────────────────────────────────────────────────────────────────
    # Leaderboard v2 (rewards-focused)
    # ──────────────────────────────────────────────────────────────────

    def get_leaderboard_v2_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """
        Fetch v2 leaderboard history (rewards per neuron over time).
        Auto-chunks by 3-day windows.

        Returns: [{updated_at, neuron_uid, rewards, coldkey, ip_address}, ...]
        """
        dt_start = _parse_date(start_date)
        dt_end = _parse_date(end_date)
        all_records: List[Dict] = []

        chunk_start = dt_start
        while chunk_start < dt_end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS_LEADERBOARD), dt_end)

            params = {
                "start_time": _iso(chunk_start),
                "end_time": _iso(chunk_end),
            }
            logger.info("[SynthData] leaderboard/v2 %s → %s",
                        _iso(chunk_start), _iso(chunk_end))

            data = self._get("/v2/leaderboard/historical", params)
            if data and isinstance(data, list):
                all_records.extend(data)
                logger.info("[SynthData]   → %d records (total %d)",
                            len(data), len(all_records))

            chunk_start = chunk_end

        return all_records

    def get_leaderboard_v2_latest(self) -> List[Dict]:
        """Current rewards snapshot for all neurons."""
        data = self._get("/v2/leaderboard/latest", {})
        return data if isinstance(data, list) else []

    # ──────────────────────────────────────────────────────────────────
    # Leaderboard v1 (full metagraph: incentive, emission, stake, rank)
    # ──────────────────────────────────────────────────────────────────

    def get_leaderboard_v1_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        """
        Fetch v1 leaderboard history (full metagraph data).
        Auto-chunks by 3-day windows.

        Returns: [{updated_at, neuron_uid, incentive, emission,
                   stake, rank, pruning_score, coldkey, ip_address}, ...]
        """
        dt_start = _parse_date(start_date)
        dt_end = _parse_date(end_date)
        all_records: List[Dict] = []

        chunk_start = dt_start
        while chunk_start < dt_end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS_LEADERBOARD), dt_end)

            params = {
                "start_time": _iso(chunk_start),
                "end_time": _iso(chunk_end),
            }
            logger.info("[SynthData] leaderboard/v1 %s → %s",
                        _iso(chunk_start), _iso(chunk_end))

            data = self._get("/leaderboard/historical", params)
            if data and isinstance(data, list):
                all_records.extend(data)
                logger.info("[SynthData]   → %d records (total %d)",
                            len(data), len(all_records))

            chunk_start = chunk_end

        return all_records

    def get_leaderboard_v1_latest(self) -> List[Dict]:
        """Current metagraph snapshot (incentive, emission, stake, rank)."""
        data = self._get("/leaderboard/latest", {})
        return data if isinstance(data, list) else []

    # ──────────────────────────────────────────────────────────────────
    # Miner validation status
    # ──────────────────────────────────────────────────────────────────

    def get_miner_validation(self, uid: int) -> Optional[Dict]:
        """Check if a miner passes validation. Returns {validated, reason, response_time}."""
        return self._get("/validation/miner", {"uid": uid})

    # ──────────────────────────────────────────────────────────────────
    # Convenience: top miners by CRPS
    # ──────────────────────────────────────────────────────────────────

    def get_top_miners_by_crps(
        self,
        asset: str = "BTC",
        days_back: int = 14,
        top_n: int = 20,
        time_length: int = 86400,
    ) -> List[Dict]:
        """
        Fetch recent scores and rank miners by mean CRPS (lower = better).

        Returns list sorted by mean CRPS, each dict has:
            {miner_uid, mean_crps, median_crps, count, std_crps}
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days_back)

        scores = self.get_historical_scores(
            asset=asset,
            start_date=_iso(start),
            end_date=_iso(now),
            time_length=time_length,
        )
        if not scores:
            return []

        from collections import defaultdict
        import statistics

        by_miner: dict[int, list[float]] = defaultdict(list)
        for s in scores:
            uid = s.get("miner_uid")
            crps = s.get("crps")
            if uid is not None and crps is not None:
                by_miner[int(uid)].append(float(crps))

        result = []
        for uid, crps_list in by_miner.items():
            result.append({
                "miner_uid": uid,
                "mean_crps": statistics.mean(crps_list),
                "median_crps": statistics.median(crps_list),
                "std_crps": statistics.stdev(crps_list) if len(crps_list) > 1 else 0.0,
                "count": len(crps_list),
            })

        result.sort(key=lambda x: x["mean_crps"])
        return result[:top_n]

    # ──────────────────────────────────────────────────────────────────
    # Backward compatibility aliases
    # ──────────────────────────────────────────────────────────────────

    def get_historical_validation_scores(self, asset, start_date, end_date,
                                         time_length=86400, time_increment=300):
        """Legacy alias used by fetch_daemon."""
        return self.get_historical_scores(
            asset=asset, start_date=start_date, end_date=end_date,
            time_length=time_length, time_increment=time_increment,
        )

    def get_latest_validation_scores(self, asset, time_length=86400,
                                     time_increment=300):
        """Legacy alias."""
        return self.get_latest_scores(
            asset=asset, time_length=time_length, time_increment=time_increment,
        )
