from __future__ import annotations

import os

from synth.miner.config.strategy_store import StrategyStore


def test_strategy_store_loads_yaml(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "\n".join(
            [
                'version: "2.0"',
                'updated_at: "2026-03-31T10:00:00Z"',
                "fallback_chain:",
                '  L2_model: "garch_v2"',
                '  L3_model: "garch_v4"',
                "routing:",
                "  BTC:",
                "    high:",
                '      ensemble_method: "weighted_average"',
                "      models:",
                '        - name: "garch_v4"',
                "          weight: 1.0",
                "          params: {}",
            ]
        ),
        encoding="utf-8",
    )
    store = StrategyStore(path=str(path))
    configs = store.get_strategy_list("BTC", 3600)
    assert len(configs) == 1
    assert configs[0].strategy_name == "garch_v4"
    chain = store.get_fallback_chain()
    assert chain[0] == "garch_v2"
    assert chain[1] == "garch_v4"


def test_strategy_store_reload_after_atomic_replace(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "\n".join(
            [
                'version: "2.0"',
                "routing:",
                "  BTC:",
                "    high:",
                "      models:",
                '        - name: "garch_v4"',
                "          weight: 1.0",
                "          params: {}",
            ]
        ),
        encoding="utf-8",
    )
    store = StrategyStore(path=str(path))
    first = store.get_strategy_list("BTC", 3600)
    assert first[0].strategy_name == "garch_v4"

    tmp_new = tmp_path / "strategies.tmp.yaml"
    tmp_new.write_text(
        "\n".join(
            [
                'version: "2.0"',
                "routing:",
                "  BTC:",
                "    high:",
                "      models:",
                '        - name: "garch_v2_2"',
                "          weight: 1.0",
                "          params: {}",
            ]
        ),
        encoding="utf-8",
    )
    os.replace(tmp_new, path)
    second = store.get_strategy_list("BTC", 3600)
    assert second[0].strategy_name == "garch_v2_2"
