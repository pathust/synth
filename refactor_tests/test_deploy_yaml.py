from __future__ import annotations

import yaml

from synth.miner.deploy.applier import validate_config
from synth.miner.deploy.exporter import export_best_config


def test_export_best_config_writes_yaml(tmp_path):
    output_path = tmp_path / "strategies.yaml"
    scan_results = [
        {
            "strategy": "garch_v2",
            "asset": "BTC",
            "frequency": "high",
            "avg_score": 0.1,
            "successful_runs": 3,
            "kwargs": {"p": 1, "q": 1},
        },
        {
            "strategy": "garch_v4",
            "asset": "BTC",
            "frequency": "high",
            "avg_score": 0.2,
            "successful_runs": 3,
            "kwargs": {},
        },
    ]
    config = export_best_config(
        scan_results=scan_results,
        top_n=2,
        output_path=str(output_path),
        backup=False,
        min_successful_runs=1,
    )
    assert ("BTC", "high") in config
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["version"] == "2.0"
    assert "routing" in payload
    assert payload["routing"]["BTC"]["high"]["models"][0]["name"] == "garch_v2"


def test_validate_config_yaml(monkeypatch, tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "\n".join(
            [
                'version: "2.0"',
                "routing:",
                "  BTC:",
                "    high:",
                "      models:",
                '        - name: "garch_v2"',
                "          weight: 1.0",
                "          params: {}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "synth.miner.strategies.registry.StrategyRegistry.auto_discover",
        lambda self: None,
    )
    monkeypatch.setattr(
        "synth.miner.strategies.registry.StrategyRegistry.list_all",
        lambda self: ["garch_v2", "garch_v4"],
    )
    warnings = validate_config(str(path))
    assert warnings == []
