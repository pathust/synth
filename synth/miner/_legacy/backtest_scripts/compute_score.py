"""
compute_score.py

Ported from sn50/synth/miner/compute_score.py (62 lines).

No changes from sn50 — all imports are available in synth:
    - adjust_predictions → synth/utils/helpers.py
    - calculate_crps_for_miner → synth/validator/crps_calculation.py
    - DataHandler, ValidatorRequest → synth/miner/data_handler.py, synth/db/models.py
    - prompt_config → synth/validator/prompt_config.py

Flow:
    cal_reward(data_handler, validator_request, miner_prediction)
      → get real prices from Pyth API (cached to JSON file)
      → adjust_predictions() → trim prediction format
      → calculate_crps_for_miner() → CRPS score
      → return (score, detailed_crps_data, prediction, real_prices)
"""

import numpy as np
import os
import json
import hashlib
import time
from synth.utils.helpers import adjust_predictions
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.miner.data_handler import DataHandler
from synth.db.models import ValidatorRequest
from synth.validator import prompt_config

LOG_REAL_PRICES = "synth/miner/logs/real_prices"
os.makedirs(LOG_REAL_PRICES, exist_ok=True)


def cal_reward(
    data_handler: DataHandler,
    validator_request: ValidatorRequest,
    miner_prediction,
):
    """
    Calculate CRPS reward score for a miner's prediction.
    sn50 L11-62

    Args:
        data_handler: DataHandler instance for fetching real prices
        validator_request: ValidatorRequest with asset, start_time, etc.
        miner_prediction: Prediction tuple (start_time, time_increment, *paths)

    Returns:
        tuple: (score, detailed_crps_data, miner_prediction, real_prices)
               score = -1 if error, otherwise sum of CRPS scores
    """
    # Select scoring intervals based on time_length
    scoring_intervals = (
        prompt_config.HIGH_FREQUENCY.scoring_intervals
        if validator_request.time_length == prompt_config.HIGH_FREQUENCY.time_length
        else prompt_config.LOW_FREQUENCY.scoring_intervals
    )

    # Cache real prices to avoid re-fetching (sn50 L20-33)
    dict_validator_request = validator_request.__dict__
    dict_validator_request.pop("_sa_instance_state", None)
    print(f"Validator request: {dict_validator_request}")
    hash_request = hashlib.sha256(
        str(dict_validator_request).encode()
    ).hexdigest()
    path_real_prices = os.path.join(LOG_REAL_PRICES, f"{hash_request}.json")

    if os.path.exists(path_real_prices):
        with open(path_real_prices, "r") as f:
            real_prices = json.load(f)
    else:
        t1 = time.time()
        real_prices = data_handler.get_real_prices(
            **{"validator_request": validator_request}
        )
        print(f"Time to get real prices: {time.time() - t1} seconds")
        with open(path_real_prices, "w") as f:
            json.dump(real_prices, f)

    print(
        f"Real prices first 10 points: {real_prices[:10]}\n"
        f"and last 10 points: {real_prices[-10:]}"
    )

    if len(real_prices) == 0:
        return -1, [], miner_prediction, real_prices

    # Adjust prediction format and calculate CRPS
    predictions_path = adjust_predictions(miner_prediction)
    simulation_runs = np.array(predictions_path).astype(float)

    try:
        score, detailed_crps_data = calculate_crps_for_miner(
            simulation_runs,
            np.array(real_prices),
            validator_request.time_increment,
            scoring_intervals,
        )
    except Exception as e:
        print(f"Error calculating CRPS for miner: {e}")
        return -1, [], miner_prediction, real_prices

    if np.isnan(score):
        print("CRPS calculation returned NaN for miner")
        return -1, detailed_crps_data, miner_prediction, real_prices

    return score, detailed_crps_data, miner_prediction, real_prices
