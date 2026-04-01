from __future__ import annotations

import numpy as np

from synth.miner.strategies.new_strategy_template import NewStrategyTemplate


def _sample_prices(n: int = 600, s0: float = 70000.0) -> dict[str, float]:
    data = {}
    for i in range(n):
        data[str(1700000000 + i * 60)] = s0 + i * 0.1
    return data


def test_shape():
    st = NewStrategyTemplate()
    prices = _sample_prices()
    out = st.simulate(
        prices_dict=prices,
        asset="BTC",
        time_increment=60,
        time_length=3600,
        n_sims=50,
        seed=42,
    )
    assert out.shape == (50, 61)


def test_reproducibility():
    st = NewStrategyTemplate()
    prices = _sample_prices()
    a = st.simulate(prices, "BTC", 60, 3600, 20, seed=42)
    b = st.simulate(prices, "BTC", 60, 3600, 20, seed=42)
    assert np.allclose(a, b)


def test_no_nan_inf_positive():
    st = NewStrategyTemplate()
    prices = _sample_prices()
    out = st.simulate(prices, "BTC", 60, 3600, 20, seed=1)
    assert np.isfinite(out).all()
    assert (out > 0).all()
