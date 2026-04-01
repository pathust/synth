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

from synth.miner.entry_new import generate_simulations
from synth.simulation_input import SimulationInput
from synth.utils.helpers import get_current_time, round_time_to_minutes
from synth.validator.response_validation_v2 import validate_responses, CORRECT

from synth.miner.compute_score import cal_reward

# Thư mục log khi format validation != CORRECT
INCORRECT_LOG_DIR = "synth/miner/logs/incorrect_validation"
os.makedirs(INCORRECT_LOG_DIR, exist_ok=True)


def _log_incorrect_validation(
    asset: str,
    start_time: datetime,
    simulation_input: SimulationInput,
    format_validation: str,
    prediction,
) -> None:
    """
    Ghi file log chi tiết khi validate_responses không trả về CORRECT.
    File ghi rõ: thời điểm, asset, tham số prompt, lỗi (incorrect ở đâu, tại sao), snippet prediction.
    """
    ts = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_asset = (asset or "unknown").replace(os.path.sep, "_")
    filename = f"incorrect_{ts}_{safe_asset}.json"
    path = os.path.join(INCORRECT_LOG_DIR, filename)

    # Snippet prediction (tránh ghi cả list quá dài)
    prediction_snippet = None
    if prediction is not None:
        if isinstance(prediction, (list, tuple)):
            if len(prediction) <= 3:
                prediction_snippet = list(prediction)
            else:
                prediction_snippet = {
                    "len": len(prediction),
                    "start_time": prediction[0] if len(prediction) > 0 else None,
                    "time_increment": prediction[1] if len(prediction) > 1 else None,
                    "num_paths": len(prediction) - 2 if len(prediction) > 2 else 0,
                    "first_path_len": len(prediction[2]) if len(prediction) > 2 and isinstance(prediction[2], (list, tuple)) else None,
                    "first_path_sample": list(prediction[2][:5]) if len(prediction) > 2 and isinstance(prediction[2], (list, tuple)) and len(prediction[2]) >= 5 else None,
                }
        else:
            prediction_snippet = {"type": type(prediction).__name__, "repr": repr(prediction)[:500]}

    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "start_time": start_time.isoformat(),
        "time_increment": simulation_input.time_increment,
        "time_length": simulation_input.time_length,
        "num_simulations": simulation_input.num_simulations,
        "incorrect_reason": format_validation,
        "prediction_snippet": prediction_snippet,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        print(f"[INCORRECT] Logged to {path}")
    except Exception as e:
        print(f"[INCORRECT] Failed to write log {path}: {e}")
from synth.miner.data_handler import DataHandler
from synth.miner.price_aggregation import aggregate_1m_to_5m
from synth.db.models import ValidatorRequest
from synth.validator import prompt_config

data_handler = DataHandler()


def get_5m_prices_from_db(asset: str):
    """
    Lấy data 5m từ DB. Nếu chưa có 5m thì load 1m từ DB rồi aggregate sang 5m.

    Returns:
        dict: {"5m": {timestamp_str: price, ...}} hoặc {} nếu không có dữ liệu.
    """
    hist = data_handler.load_price_data(asset, "5m")
    if hist and "5m" in hist and hist["5m"]:
        return hist
    loaded_1m = data_handler.load_price_data(asset, "1m")
    if not loaded_1m or "1m" not in loaded_1m or not loaded_1m["1m"]:
        return {}
    prices_5m = aggregate_1m_to_5m(loaded_1m["1m"])
    if not prices_5m:
        return {}
    return {"5m": prices_5m}

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

    if format_validation != CORRECT:
        _log_incorrect_validation(
            asset=simulation_input.asset,
            start_time=start_time,
            simulation_input=simulation_input,
            format_validation=format_validation,
            prediction=prediction,
        )

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
    assets: list[str] = None,
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

    # Dùng toàn bộ asset của prompt (LOW_FREQUENCY = BTC, ETH, XAU, SOL, ...) hoặc dùng assets truyền vào
    SYMBOLS = assets if assets is not None else test_prompt_config.asset_list
    LOG_DIR = "synth/miner/logs/benchmark_test"
    os.makedirs(LOG_DIR, exist_ok=True)

    START_DATE = datetime(2026, 3, 12, 0, 0, 0)
    END_DATE = datetime(2026, 3, 12, 23, 59, 59)

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
    asset = "SOL"
    time_increment = 60
    time_length = 3600
    num_simulations = 1000

    start_time = datetime(2026, 3, 12, 5, 12, 0)
    test_prompt_config = prompt_config.HIGH_FREQUENCY

    # Single test
    # simple_test(asset, start_time, test_prompt_config)

    # Benchmark test
    # Pass [asset] as a list to allow benchmark for only that asset
    scores = benchmark_test(30, test_prompt_config, "entry_new", assets=[asset])
