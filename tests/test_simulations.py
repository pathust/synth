import numpy as np

from synth.miner.entry import generate_simulations
from synth.simulation_input import SimulationInput
from synth.utils.helpers import get_current_time, round_time_to_minutes
from synth.validator.response_validation_v2 import CORRECT, validate_responses


def test_generate_simulations(monkeypatch):
    monkeypatch.setattr(
        "synth.miner.entry._run_ensemble_via_builder",
        lambda *args, **kwargs: np.ones((50, 289), dtype=float),
    )
    monkeypatch.setattr(
        "synth.miner.entry.validate_responses",
        lambda *args, **kwargs: CORRECT,
    )
    current_time = get_current_time()
    start_time = round_time_to_minutes(current_time, 120)
    start_time_str = start_time.isoformat()

    sim_input = SimulationInput(
        asset="BTC",
        time_increment=300,
        time_length=86400,
        num_simulations=50,
    )
    result_dict = generate_simulations(
        simulation_input=sim_input,
        asset="BTC",
        start_time=start_time_str,
        time_increment=300,
        time_length=86400,
        num_simulations=50,
    )

    result = result_dict.get("predictions")
    assert isinstance(result, tuple)
    assert len(result) == 52
    assert all(
        isinstance(sub_array, list) and len(sub_array) == 289
        for sub_array in result[2:]
    )


def test_run(monkeypatch):
    monkeypatch.setattr(
        "synth.miner.entry._run_ensemble_via_builder",
        lambda *args, **kwargs: np.ones((50, 289), dtype=float),
    )
    monkeypatch.setattr(
        "synth.miner.entry.validate_responses",
        lambda *args, **kwargs: CORRECT,
    )
    simulation_input = SimulationInput(
        asset="BTC",
        time_increment=300,
        time_length=86400,
        num_simulations=50,
    )

    current_time = get_current_time()
    start_time = round_time_to_minutes(current_time, 120)
    simulation_input.start_time = start_time.isoformat()

    print("start_time", simulation_input.start_time)
    prediction_dict = generate_simulations(
        simulation_input=simulation_input,
        asset=simulation_input.asset,
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
    )
    prediction = prediction_dict.get("predictions")

    format_validation = validate_responses(
        prediction,
        simulation_input,
        "0",
    )

    print(format_validation)

    assert format_validation == CORRECT
