# Review Chi Tiet: Backtest, Strategy, Tuning

Ngay review: 2026-04-01  
Reviewer: Codex (code + command verification trong `conda -n synth`)

## 1. Executive summary

Cum `backtest/strategy/tuning` da co khung to chuc ro (runner, tuner, regime engine, exporter), nhung hien co mot so loi logic co the lam sai ket luan tuning va routing deploy:

- Co loi horizon/frequency trong regime backtest pipeline.
- Co loi scoring interval gap (naming mismatch + indexing bug).
- Co drift du lieu high/low giua cac luong backtest slot va runtime simulation.
- Co mot so bug CLI/tooling lam hu workflow review nhanh.

## 2. Findings (sap theo muc do nghiem trong)

### [P1] Regime backtest dang tune/eval theo `config.frequency`, bo qua `case.market_type`

**Bang chung**

- Trong `evaluate_case`, tune/validation/test deu truyen `config.frequency`, khong dung `case_key.market_type`:
  - `synth/miner/backtest/regime_engine.py:455-459`
  - `synth/miner/backtest/regime_engine.py:469-473`
  - `synth/miner/backtest/regime_engine.py:525-529`
- Khi export runtime, crypto case `market_type=high|low` lai duoc map thanh frequency high/low:
  - `synth/miner/backtest/regime_engine.py:729-733`
- CLI default frequency la `low`:
  - `synth/miner/run_regime_backtest.py:43`

**Tac dong**

- Case `btc/high/*` co the duoc chon strategy dua tren backtest low (24h) nhung van export vao bucket high (1h).
- Routing YAML sinh ra de deploy co nguy co "high-route duoc tune bang low-horizon".

**Khuyen nghi**

- Tinh `effective_frequency` theo `case_key.market_type` (crypto high/low), khong su dung 1 bien toan cuc cho moi case.
- Bo sung test assert frequency theo case.

---

### [P1] Scoring interval "gap" cua HFT khong duoc kich hoat dung

**Bang chung**

- Prompt config dat ten interval dang `_gaps`:
  - `synth/validator/prompt_config.py:66-77`
- CRPS logic lai check suffix `_gap`:
  - `synth/validator/crps_calculation.py:37-39`
  - `synth/miner/backtest/metrics.py:56-58`
- Runtime verify:
  - `conda run -n synth python -c "from synth.validator.prompt_config import HIGH_FREQUENCY; ..."`
  - Ket qua: `0` key ket thuc `_gap`, `12` key ket thuc `_gaps`.

**Tac dong**

- Toan bo block gap scoring HFT khong bao gio duoc bat theo y do.
- CRPS thuc te khac voi design scoring trong prompt config.

**Khuyen nghi**

- Chot 1 naming convention (`_gap` hoac `_gaps`) va dong bo prompt + scorer.
- Them unit test cho suffix parser.

---

### [P1] Nhanh gap mode con co bug index neu duoc kich hoat

**Bang chung**

- Khi `is_gap=True`, code cat `interval_prices = interval_prices[:1]`:
  - `synth/validator/crps_calculation.py:167-170`
- Lenh verify nhanh:
  - `conda run -n synth python -c "... calculate_price_changes_over_intervals(..., is_gap=True) ..."`
  - Output shape `(1, 3)` voi input 2 paths => mat toan bo paths khac, khong phai "lay 1 gap theo thoi gian".

**Tac dong**

- Neu fix suffix de bat gap mode ma khong fix index nay, scoring se sai nghiem trong.

**Khuyen nghi**

- Dinh nghia ro "gap interval" can output dang nao, sau do sua slicing theo truc thoi gian (khong theo truc so path).
- Them regression test cho shape + gia tri gap mode.

---

### [P1] Backtest slot va runtime simulation deu dang "uu tien high" cho asset dual-frequency

**Bang chung**

- `BacktestEngine._get_prompt_config` uu tien HIGH neu asset co nhan high:
  - `synth/miner/backtest/engine.py:70-74`
- `run_slots` khong nhan frequency argument; luon dung cfg tu asset:
  - `synth/miner/backtest/engine.py:87-95`
- `UnifiedDataLoader` cung map theo asset labels:
  - `synth/miner/data/dataloader.py:22-26`
- Verify:
  - `conda run -n synth python -c "from synth.miner.backtest.engine import BacktestEngine; ..."`
  - Output: `BTC -> high`, `NVDAX -> low`.

**Tac dong**

- Khong the danh gia slot low cho BTC/ETH/SOL/XAU trong `run_slots`.
- Co nguy co dung data granularity 1m cho path can 5m o low-horizon.

**Khuyen nghi**

- Truyen explicit `frequency`/`time_increment` vao `run_slots` va dataloader APIs.

---

### [P2] `run_strategy_scan --metric DIR_ACC` bi hong ngay tai startup

**Bang chung**

- CLI cho phep `DIR_ACC`:
  - `synth/miner/run_strategy_scan.py:84-87`
- Runner chi support metrics trong `METRICS`:
  - `synth/miner/backtest/runner.py:40-43`
  - `synth/miner/backtest/metrics.py:170-174`
- Verify command:
  - `conda run -n synth python -m synth.miner.run_strategy_scan --assets BTC --frequencies low --num-runs 1 --num-sims 1 --metric DIR_ACC`
  - Ket qua: `ValueError: Unknown metric 'DIR_ACC'. Available: ['CRPS', 'MAE', 'RMSE']`.

**Tac dong**

- CLI contract sai, gay fail cho nguoi dung va CI script.

**Khuyen nghi**

- Hoac bo `DIR_ACC` khoi choices, hoac implement metric tuong ung trong `METRICS`.

---

### [P2] Backtest low-frequency co the fail du DB co 1m data (do bo qua ket qua aggregate)

**Bang chung**

- `fetch_price_data(..., only_load=True)` co nhanh aggregate 1m -> 5m va return data:
  - `synth/miner/my_simulation.py:67-83`
- `run_single` goi ham nay nhung khong dung return; sau do load lai truc tiep 5m tu DB:
  - `synth/miner/backtest/runner.py:75-80`

**Tac dong**

- Neu DB thieu 5m nhung co 1m, backtest van FAIL "No data for asset/5m".

**Khuyen nghi**

- Su dung data tra ve tu `fetch_price_data`, hoac persist aggregated 5m truoc khi `load_price_data`.

---

### [P2] Seed ensemble khong reproducible giua cac process

**Bang chung**

- Sub-seed dung Python `hash(...)`:
  - `synth/miner/ensemble/builder.py:18-20`
  - `synth/miner/strategies/ensemble_weighted.py:97`
- Verify 2 process lien tiep:
  - Lan 1: `_make_sub_seed(42,'garch_v4') -> 82778`
  - Lan 2: `_make_sub_seed(42,'garch_v4') -> 82248`

**Tac dong**

- Cung config + seed nhung ket qua ensemble khac nhau qua moi lan restart process.
- Giam do tin cay cua tuning comparison va reproducibility.

**Khuyen nghi**

- Dung hash on dinh (vd `hashlib.sha256(strategy_name.encode())`) de sinh sub-seed.

---

### [P3] `run_duel --list` crash do metadata field da cu

**Bang chung**

- CLI truy cap `strat.supported_assets` va `strat.supported_frequencies`:
  - `synth/miner/backtest/run_duel.py:57-58`
- BaseStrategy hien tai khong co 2 field nay:
  - `synth/miner/strategies/base.py:67-69`
- Verify command:
  - `conda run -n synth python -m synth.miner.backtest.run_duel --list`
  - Ket qua: `AttributeError: 'ArimaEquityStrategy' object has no attribute 'supported_assets'`.

**Tac dong**

- Workflow inspect nhanh strategy inventory bi vo.

**Khuyen nghi**

- Sua CLI sang dung metadata hien tai (`supported_asset_types`) hoac method `supports_*`.

---

### [P3] `scan_all` khong filter compatibility theo frequency

**Bang chung**

- `scan_all` lap theo `asset x freq` nhung strategy list chi filter theo asset:
  - `synth/miner/backtest/runner.py:277-304`

**Tac dong**

- Co the quet nhung combo khong hop le theo frequency policy, tang noise/chi phi scan.

**Khuyen nghi**

- Them filter `strategy.supports_frequency(freq)` va/hoac filter theo prompt support matrix.

## 3. Test coverage gaps

1. Chua co test bat loi `_gap` vs `_gaps` cho scoring intervals.
2. Chua co test cho behavior frequency-per-case trong `PredictionBacktestEngine`.
3. Chua co test contract cho CLI metric choices (`run_strategy_scan`).
4. Chua co test smoke cho `run_duel --list`.

## 4. Ke hoach xu ly de xuat

### Nhom A (can lam ngay)

1. Fix regime engine frequency-per-case.
2. Fix scoring interval parser + gap slicing.
3. Fix `run_strategy_scan` metric contract.

### Nhom B (gan)

1. Truyen explicit frequency/time_increment vao dataloader va `run_slots`.
2. On dinh hoa seed generation (hashlib).
3. Fix `run_duel --list`.

### Nhom C (chat luong he thong)

1. Bo sung integration tests cho low/high end-to-end.
2. Chot mot data path duy nhat cho backtest vs runtime de tranh drift.

## 5. Lenh verify da su dung

- `conda run -n synth python -m synth.miner.backtest.run_duel --list`
- `conda run -n synth python -m synth.miner.run_strategy_scan --assets BTC --frequencies low --num-runs 1 --num-sims 1 --metric DIR_ACC`
- `conda run -n synth python -c "from synth.miner.backtest.engine import BacktestEngine; ..."`
- `conda run -n synth python -c "from synth.miner.data.dataloader import UnifiedDataLoader; ..."`
- `conda run -n synth python -c "from synth.validator.prompt_config import HIGH_FREQUENCY; ..."`
- `conda run -n synth python -c "import numpy as np; from synth.validator.crps_calculation import calculate_price_changes_over_intervals as f; ..."`
- `conda run -n synth python -c "from synth.miner.ensemble.builder import _make_sub_seed; print(_make_sub_seed(...))"`
