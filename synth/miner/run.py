"""
run.py

Ported from sn50/synth/miner/run.py (146 lines).

Changes from sn50:
    1. Adapted validate_responses() call: 4 args (sn50) → 3 args (synth)
       sn50 L41-46: validate_responses(prediction, simulation_input,
                     datetime.fromisoformat(...), "0")
       synth:       validate_responses(prediction, simulation_input, "0")
    2. Kept simple_test(), get_random_date(), benchmark_test() intact.
    3. Changed import: ValidatorRequest from synth.db.models (not data_handler)

Usage:
    cd /Users/taiphan/Documents/synth
    conda activate synth
    python synth/miner/run.py
"""

import json
import os
from datetime import datetime, timedelta, timezone

from synth.miner.simulations import generate_simulations
from synth.simulation_input import SimulationInput
from synth.utils.helpers import get_current_time, round_time_to_minutes
from synth.validator.response_validation_v2 import validate_responses

from synth.miner.compute_score import cal_reward
from synth.miner.data_handler import DataHandler
from synth.db.models import ValidatorRequest
from synth.validator import prompt_config

data_handler = DataHandler()

# python synth/miner/run.py


def simple_test(
    asset: str,
    start_time: datetime,
    test_prompt_config: prompt_config.PromptConfig,
):
    """
    Run a single simulation test and calculate CRPS score.
    sn50 L15-63

    Args:
        asset: Asset to simulate (e.g., "BTC")
        start_time: Start time for the simulation
        test_prompt_config: PromptConfig with scoring parameters

    Returns:
        dict: {"score": float, "prediction": tuple, "real_prices": list}
    """
    start_time = round_time_to_minutes(
        start_time, test_prompt_config.timeout_extra_seconds
    )
    simulation_input = SimulationInput(
        asset=asset,
        start_time=start_time.isoformat(),
        time_increment=test_prompt_config.time_increment,
        time_length=test_prompt_config.time_length,
        num_simulations=test_prompt_config.num_simulations,
    )
    print("start_time", simulation_input.start_time)

    prediction = generate_simulations(
        simulation_input,
        simulation_input.asset,
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
        seed=42,
    )["predictions"]

    # Validate format — adapted for synth's 3-arg signature
    # sn50 uses 4 args: (prediction, simulation_input, request_time, "0")
    # synth uses 3 args: (prediction, simulation_input, "0")
    format_validation = validate_responses(
        prediction,
        simulation_input,
        "0",
    )
    print(format_validation)

    # Calculate CRPS score against real prices
    validator_request = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=test_prompt_config.time_length,
        time_increment=test_prompt_config.time_increment,
    )

    score, detailed_crps_data, miner_prediction, real_prices = cal_reward(
        data_handler, validator_request, prediction
    )
    print("Score:", score)
    return {
        "score": score,
        "prediction": prediction,
        "real_prices": real_prices,
    }


def get_random_date(start_date, end_date, num_dates, seed=42):
    """
    Generate random dates between start_date and end_date.
    sn50 L65-80
    """
    import random

    random.seed(seed)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    random_dates = []
    for _ in range(num_dates):
        random_number_of_days = random.randint(0, days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        random_dates.append(random_date)

    return random_dates


def benchmark_test(
    num_runs: int,
    test_prompt_config: prompt_config.PromptConfig,
    version_name: str,
    seed: int = 42,
):
    """
    Run benchmark tests over multiple dates and assets.
    Logs results to JSON files.
    sn50 L82-128

    Args:
        num_runs: Number of test runs (used for get_random_date)
        test_prompt_config: PromptConfig with scoring parameters
        version_name: Version name for log directory
        seed: Random seed for date generation

    Returns:
        dict: {symbol: [scores...]}
    """
    import numpy as np

    SYMBOLS = ["BTC"]
    LOG_DIR = "synth/miner/logs/benchmark_test"
    os.makedirs(LOG_DIR, exist_ok=True)

    START_DATE = datetime(2025, 2, 1, 0, 0, 0)
    END_DATE = datetime(2025, 11, 27, 14, 12, 0)

    # Generate test dates
    dates = get_random_date(START_DATE, END_DATE, num_dates=num_runs, seed=seed)

    scores = {}
    for symbol in SYMBOLS:
        log_dir = os.path.join(LOG_DIR, symbol, version_name)
        os.makedirs(log_dir, exist_ok=True)

        scores[symbol] = []
        for date_ in dates:
            result = simple_test(symbol, date_, test_prompt_config)
            scores[symbol].append(result["score"])
            log_dt = {
                "date": date_.isoformat(),
                "symbol": symbol,
                "score": result["score"],
                "prediction": result["prediction"],
                "real_prices": result["real_prices"],
            }
            with open(
                os.path.join(log_dir, f"{date_.isoformat()}.json"), "w"
            ) as f:
                json.dump(log_dt, f)

    print(f"Dates: {dates}")
    print(f"Scores: {scores}")
    for symbol in SYMBOLS:
        sco = [x for x in scores[symbol] if x != "skip"]
        print(f"Symbol: {symbol}, Average score: {np.mean(sco)}")
    return scores


if __name__ == "__main__":
    asset = "BTC"
    time_increment = 300
    time_length = 86400
    num_simulations = 1000

    start_time = datetime(2025, 12, 8, 5, 12, 0)
    test_prompt_config = prompt_config.LOW_FREQUENCY

    # Single test
    # simple_test(asset, start_time, test_prompt_config)

    # Benchmark test
    scores = benchmark_test(30, test_prompt_config, "v2_garch")
