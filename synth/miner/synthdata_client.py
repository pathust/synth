"""
synthdata_client.py

Client for api.synthdata.co to fetch historical validation scores.
Used for realistic mainnet backtesting.
"""

import requests
import datetime
from typing import List, Dict, Optional

class SynthDataClient:
    BASE_URL = "https://api.synthdata.co"

    def __init__(self):
        self.session = requests.Session()

    def get_historical_validation_scores(
        self, 
        asset: str, 
        start_date: str, 
        end_date: str, 
        time_length: int = 86400,
        time_increment: int = 300
    ) -> List[Dict]:
        """
        Fetch historical validation scores.
        Dates should be ISO8601 strings (e.g. YYYY-MM-DD).
        Results may be large, handle with care.
        """
        url = f"{self.BASE_URL}/validation/scores/historical"
        params = {
            "from": start_date,
            "to": end_date,
            "asset": asset,
            "time_length": time_length,
            "time_increment": time_increment
        }
        
        print(f"[SynthData] Fetching validation scores for {asset} from {start_date} to {end_date} (length={time_length})")
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f"[SynthData] Retrieved {len(data)} score records for {asset}.")
            return data
        except requests.exceptions.RequestException as e:
            print(f"[SynthData ERROR] Failed to fetch validation scores: {e}")
            return []

    def get_latest_validation_scores(
        self,
        asset: str,
        time_length: int = 86400,
        time_increment: int = 300
    ) -> List[Dict]:
        """
        Fetch latest snapshot of validation scores.
        """
        url = f"{self.BASE_URL}/validation/scores/latest"
        params = {
            "asset": asset,
            "time_length": time_length,
            "time_increment": time_increment
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"[SynthData ERROR] Failed to fetch latest validation scores: {e}")
            return []
