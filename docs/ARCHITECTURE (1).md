# Synth Miner — Hướng Dẫn Kiến Trúc Hệ Thống

> **Phiên bản**: v2.0 — Tái cấu trúc toàn diện  
> **Mục tiêu**: Chuyển từ cấu trúc monolithic hiện tại sang kiến trúc modular, tối ưu cho CRPS scoring trên Bittensor Subnet 50.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Luồng Dữ Liệu: Fetch → Store → Features](#2-luồng-dữ-liệu)
3. [Chiến Lược Theo Asset × Regime](#3-chiến-lược-theo-asset--regime)
4. [Pipeline Tuning Dựa Trên CRPS](#4-pipeline-tuning-dựa-trên-crps)
5. [Backtest Engine](#5-backtest-engine)
6. [Visualization Module](#6-visualization-module)
7. [Live Miner Request Flow](#7-live-miner-request-flow)
8. [Cấu Trúc Thư Mục Chi Tiết](#8-cấu-trúc-thư-mục-chi-tiết)
9. [Migration từ Codebase Hiện Tại](#9-migration-từ-codebase-hiện-tại)
10. [Hướng Dẫn Phát Triển](#10-hướng-dẫn-phát-triển)

---

## 1. Tổng Quan Kiến Trúc

Hệ thống được chia thành **5 layer chính**:

```mermaid
graph TD
    subgraph DATA["Data Layer"]
        F1[Pyth Oracle 1m poll] --> DB[(DuckDB time-series)]
        F2[Binance REST backup/history] --> DB
        DB --> AGG[OHLCV Aggregator 1m→5m/30m/1h/…]
        AGG --> FE[Feature Engineer returns · vol · skew · kurtosis]
    end

    subgraph STRAT["Strategy Layer"]
        FE --> REG[Regime Detector HMM + pattern scan]
        REG --> SW[Asset Switcher crypto / gold / equity]
        SW --> M1[Statistical GARCH family · SABR · ARIMA]
        SW --> M2[Stochastic Heston · Merton · OU]
        SW --> M3[ML / DL LSTM · Transformer]
        M1 & M2 & M3 --> ENS[Ensemble CRPS-weighted mix]
    end

    subgraph SIM["Simulation Engine"]
        ENS --> MC[Monte Carlo vectorized numpy]
        MC --> FMT[Formatter + Validator 8-digit · count · timeout]
    end

    subgraph OPS["Offline Ops"]
        BT[Backtest rolling window replay]
        TN[Tuner Optuna · PSO · GA]
        VIZ[Visualization Streamlit dashboard]
        BT <-->|CRPS feedback| TN
        TN --> VIZ
        BT --> VIZ
    end

    subgraph BT_IF["Bittensor Interface"]
        AX[Axon handler] --> LFT[LFT router 24h request]
        AX --> HFT[HFT router 1h request]
        LFT & HFT --> FMT
    end

    FMT --> AX
    DB --> BT
    DB --> TN
    TN -->|best params YAML| ENS
    VALIDATOR[Synth Validator external] -->|prompt| AX
    FMT -->|1000 paths| VALIDATOR
    VALIDATOR -->|live CRPS| ENS
```

### Mô tả các Layer

| Layer | Trách nhiệm | Module chính |
|-------|-------------|--------------|
| **Data** | Thu thập, lưu trữ, aggregate giá, tính features | `data/fetchers/`, `data/storage/`, `data/loaders/` |
| **Strategy** | Phát hiện regime, chọn model, ensemble | `strategies/regime/`, `strategies/statistical/`, `strategies/ensemble/` |
| **Simulation** | Chạy Monte Carlo, format output | `simulation/engine.py`, `simulation/formatter.py` |
| **Offline Ops** | Backtest, tuning hyperparams, visualization | `backtest/`, `training/`, `visualization/` |
| **Bittensor** | Axon handler, dispatch LFT/HFT | `neurons/miner.py` |

---

## 2. Luồng Dữ Liệu

```mermaid
flowchart LR
    PY[Pyth Oracle] -->|websocket / REST| FC[Fetcher ever 60s]
    BIN[Binance REST] -->|fallback| FC
    FC -->|raw tick asset · price · ts| CACHE[In-memory cache deque per asset]
    FC -->|batch flush| DB[(DuckDB raw_prices table)]
    CACHE -->|latest price| MC_LIVE[Live simulation start_price anchor]
    DB -->|SQL query date range| AGG
    AGG -->|step_size param 5m / 30m / 1h / 1d| OHLCV[OHLCV candles]
    OHLCV --> FE[Feature Engineer]
    FE --> FEAT["`**Features**
    • log returns
    • realized vol Nmin
    • rolling skew / kurtosis
    • RSI · ATR · Bollinger
    • intraday seasonality idx
    • overnight gap flag`"]
    FEAT --> STRAT[Strategy models]
```

### Chi tiết triển khai

**Fetchers** (`data/fetchers/`):
- `base_fetcher.py` — ABC định nghĩa `fetch()`, `start_polling()`, `stop()`
- `pyth_fetcher.py` — Nguồn chính, poll mỗi 60s qua REST/WebSocket
- `binance_fetcher.py` — Backup + bulk history download

**Storage** (`data/storage/`):
- `timeseries_db.py` — DuckDB wrapper: `insert_tick()`, `query_range(asset, start, end)`
  - Thay thế SQLite hiện tại (`synth/miner/data/price_data.db`) bằng DuckDB cho analytical queries nhanh hơn
- `market_cache.py` — `collections.deque` per asset, giữ N ticks gần nhất trong RAM

**Loaders** (`data/loaders/`):
- `ohlcv_aggregator.py` — Resample 1m raw → OHLCV(5m/30m/1h/1d) bằng pandas resample
- `feature_engineer.py` — Tính toán tất cả technical indicators + statistical features

---

## 3. Chiến Lược Theo Asset × Regime

```mermaid
flowchart TD
    REQ[Validator request asset · horizon · n_sim] --> HROUTER{Horizon?}
    HROUTER -->|24h| LFT[LFT pipeline]
    HROUTER -->|1h| HFT[HFT pipeline]

    LFT & HFT --> ATYPE{Asset type?}

    ATYPE -->|BTC · ETH · SOL| CR[Crypto regime HMM 3-state]
    ATYPE -->|XAU| GR[Gold regime OU-test + vol ratio]
    ATYPE -->|SPYX · NVDAX TSLAX · AAPLX · GOOGLX| ER[Equity regime gap model + market hours]

    CR -->|bull / trend| C1[GJR-GARCH + Merton jump]
    CR -->|high vol / fear| C2[Heston stoch vol + LSTM vol input]
    CR -->|ranging / low vol| C3[GARCH-1-1 + GBM baseline]

    GR -->|mean-reverting| G1[OU + GARCH-1-1]
    GR -->|trending / macro| G2[SABR model + XGBoost vol]

    ER -->|market open| E1[EGARCH + intraday seasonality]
    ER -->|overnight / gap| E2[Gap distribution + ARIMA-GARCH]
    ER -->|earnings window| E3[Inflated vol model + regime switch]

    C1 & C2 & C3 --> ENS_C[Ensemble Dirichlet weights]
    G1 & G2 --> ENS_G[Ensemble Dirichlet weights]
    E1 & E2 & E3 --> ENS_E[Ensemble Dirichlet weights]

    ENS_C & ENS_G & ENS_E --> MC[Monte Carlo 1000 paths]
```

### Mapping Asset → Regime → Strategy

| Asset Group | Regime Detector | States | Strategy Bank |
|------------|----------------|--------|--------------|
| **Crypto** (BTC, ETH, SOL) | HMM 3-state | bull / high-vol / ranging | GJR-GARCH+Merton, Heston+LSTM, GARCH+GBM |
| **Gold** (XAU) | OU-test + vol ratio | mean-reverting / trending | OU+GARCH, SABR+XGBoost |
| **Equity** (SPYX, NVDAX, TSLAX, AAPLX, GOOGLX) | Gap model + market hours | open / overnight / earnings | EGARCH+seasonal, Gap+ARIMA, Inflated vol |

### Registry Pattern

```python
# strategies/registry.py
class StrategyRegistry:
    _registry: dict[tuple[str, str], list[type[BaseStrategy]]] = {}

    @classmethod
    def register(cls, asset_type: str, regime: str, strategy_cls: type):
        key = (asset_type, regime)
        cls._registry.setdefault(key, []).append(strategy_cls)

    @classmethod
    def get(cls, asset_type: str, regime: str) -> list[BaseStrategy]:
        return [s() for s in cls._registry.get((asset_type, regime), [])]
```

---

## 4. Pipeline Tuning Dựa Trên CRPS

```mermaid
flowchart TD
    RAW[Raw historical data per asset] --> SPLIT[Walk-forward split no data leakage]
    SPLIT --> TR[Train set fit models]
    SPLIT --> VAL[Val set objective evaluation]
    SPLIT --> TEST[Test set final gate only]

    subgraph L1["Level 1 — Parameter Tuning  (per model)"]
        TR --> FIT[Fit model with trial params]
        FIT --> SIM1[Simulate n=50 paths at each val ts]
        SIM1 --> CRPS1[CRPS per timestamp]
        CRPS1 --> MEAN1[mean val_CRPS]
        MEAN1 -->|minimize| OPT[Optuna TPE 100–200 trials]
        OPT -->|next params| FIT
    end

    subgraph L2["Level 2 — Model Selection"]
        OPT -->|best params per model| RANK[Rank models by val_CRPS]
        RANK --> WTEST[Wilcoxon test significance check]
        WTEST --> BEST[Select top-K models]
    end

    subgraph L3["Level 3 — Ensemble Weight Optimization"]
        BEST --> CMTX[CRPS matrix n_models × n_val_ts]
        CMTX --> WOPT[scipy SLSQP Σw=1, w≥0.05]
        WOPT --> WSTAR[w* optimal weights per asset × regime]
    end

    WSTAR --> TESTEVAL[Test set evaluation leaderboard score sim]
    TESTEVAL -->|pass| DEPLOY[Deploy to strategies.yaml hot-reload miner]
    TESTEVAL -->|degraded| SPLIT

    subgraph ONLINE["Online Feedback Loop"]
        DEPLOY --> LIVE[Live miner]
        VAL_EXT -->|real CRPS| EMA[EMA weight update α=0.9 softmin]
        EMA --> WSTAR
        EMA -->|CRPS drift > threshold| SPLIT
    end
```

### 3 Levels Tuning

| Level | Mục tiêu | Tool | Output |
|-------|----------|------|--------|
| **L1** Parameter | Tối ưu params từng model | Optuna TPE (100-200 trials) | `best_params` per model |
| **L2** Selection | Chọn top-K models có ý nghĩa thống kê | Wilcoxon signed-rank test | Top-K model list |
| **L3** Ensemble | Tối ưu trọng số mix | scipy SLSQP (Σw=1, w≥0.05) | `w*` per asset×regime |

### Online Feedback Loop
- Validator trả CRPS thực sau 24h → EMA update ensemble weights
- Nếu CRPS drift > threshold → trigger retrain pipeline

---

## 5. Backtest Engine

```mermaid
flowchart TD
    CFG[Backtest config asset · date_range · strategies · n_sim] --> LOAD[Load historical prices from DuckDB]
    LOAD --> ROLL[Rolling window iterator step = time_increment]

    ROLL --> ANCHOR[Get anchor price at t0]
    ANCHOR --> STRAT_RUN[Run all strategies in parallel]
    STRAT_RUN --> PATHS[Simulated paths n_sim × n_timepoints]
    PATHS --> REALIZED[Get realized prices at t0+5m, t0+30m, t0+3h, t0+24h]
    REALIZED --> CRPS_CALC[CRPS per increment basis-point returns]
    CRPS_CALC --> SCORE[Prompt score Σ CRPS increments]

    SCORE --> STORE[Store results per strategy · per timestamp]
    STORE --> ROLL

    STORE --> METRICS[Aggregate metrics]
    METRICS --> M1[mean CRPS per strategy]
    METRICS --> M2[CRPS time series regime overlay]
    METRICS --> M3[Leaderboard score sim 10-day rolling avg]
    METRICS --> M4[Softmax emission rank estimate vs top miners]

    M1 & M2 & M3 & M4 --> VIZ[Visualization dashboard]
```

### Metrics quan trọng

| Metric | Mô tả | File |
|--------|-------|------|
| `mean_crps` | CRPS trung bình per strategy | `backtest/metrics.py` |
| `rolling_leaderboard` | Simulate điểm leaderboard rolling 10 ngày | `backtest/metrics.py` |
| `softmax_emission` | Ước lượng emission rank với β=-0.1 | `backtest/metrics.py` |
| `crps_ensemble` | CRPS tính bằng sort trick O(N log N) | `backtest/crps.py` |

---

## 6. Visualization Module

### Class Hierarchy

```mermaid
classDiagram
    class AbstractChart {
        <<interface>>
        +data: DataFrame
        +config: dict
        +render() Figure
        +save(path)
    }

    class PriceCandleChart {
        +overlays: list
        +add_ma(period)
        +add_bollinger()
        +add_regime_color()
    }

    class PathFanChart {
        +paths: ndarray
        +realized: Series
        +percentile_bands: list
    }

    class CRPSEvolutionChart {
        +strategies: dict
        +window: int
        +show_regime: bool
    }

    class DistributionCompareChart {
        +predicted_paths: ndarray
        +realized_returns: Series
        +render_kde() Figure
        +render_qq() Figure
    }

    class RegimeOverlayChart {
        +regime_labels: Series
        +state_colors: dict
    }

    class StrategyCompareChart {
        +crps_matrix: DataFrame
        +asset_weights: dict
        +render_heatmap() Figure
        +render_boxplot() Figure
    }

    AbstractChart <|-- PriceCandleChart
    AbstractChart <|-- PathFanChart
    AbstractChart <|-- CRPSEvolutionChart
    AbstractChart <|-- DistributionCompareChart
    AbstractChart <|-- RegimeOverlayChart
    AbstractChart <|-- StrategyCompareChart

    class Dashboard {
        +charts: list~AbstractChart~
        +add_chart(chart)
        +run_streamlit()
    }

    Dashboard o-- AbstractChart
```

### Dashboard Tabs (Streamlit)

| Tab | Nội dung |
|-----|---------|
| **Live** | Real-time price + fan chart paths đang chạy |
| **Backtest** | CRPS evolution, leaderboard sim, regime overlay |
| **Training** | Optuna study progress, best params table |
| **Compare** | Strategy heatmap, boxplot CRPS distribution |

---

## 7. Live Miner Request Flow

```mermaid
sequenceDiagram
    participant V as Synth Validator
    participant AX as Axon handler
    participant RT as LFT/HFT Router
    participant DC as Data Cache
    participant RD as Regime Detector
    participant ST as Strategy Ensemble
    participant MC as Monte Carlo Engine
    participant FMT as Formatter

    V->>AX: prompt(asset, start_time, horizon, n_sim, time_increment)
    AX->>RT: dispatch by horizon
    RT->>DC: get anchor price at start_time
    DC-->>RT: price (Pyth)
    RT->>RD: detect current regime(asset, features)
    RD-->>RT: regime state + confidence
    RT->>ST: select strategies(asset, regime, horizon)
    ST-->>RT: model ensemble + weights
    RT->>MC: simulate(params, n_sim, horizon, anchor)
    note over MC: vectorized numpy ~200ms for n=1000
    MC-->>RT: paths [n_sim × n_timepoints]
    RT->>FMT: format + validate
    note over FMT: check count, 8-digit, timeout guard
    FMT-->>AX: [ts, interval, path1, path2, …]
    AX-->>V: response (within start_time deadline)
    V-->>AX: real CRPS (async, 24h later)
    AX->>ST: update ensemble weights via EMA
```

### Timing Budget

| Bước | Thời gian ước tính |
|------|-------------------|
| Price fetch từ cache | < 1ms |
| Regime detection | ~50ms |
| Strategy selection | ~10ms |
| Monte Carlo (1000 paths) | ~200ms |
| Format + validate | ~20ms |
| **Tổng** | **< 300ms** (budget: đến start_time, ~60s) |

---

## 8. Cấu Trúc Thư Mục Chi Tiết

```
synth-miner/
│
├── config/
│   ├── assets.yaml                   # Per-asset: emission weight, type, model priors
│   ├── strategies.yaml               # Active strategy + best params per asset × regime
│   └── system.yaml                   # num_simulations, mode (test/prod), DB path, ports
│
├── data/
│   ├── __init__.py
│   ├── fetchers/
│   │   ├── __init__.py
│   │   ├── base_fetcher.py           # ABC: fetch(), start_polling(), stop()
│   │   ├── pyth_fetcher.py           # Pyth Oracle — primary source, 1m poll
│   │   └── binance_fetcher.py        # Binance REST — backup + bulk history
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── timeseries_db.py          # DuckDB wrapper: insert, query by range/asset
│   │   └── market_cache.py           # In-memory deque per asset (latest N ticks)
│   └── loaders/
│       ├── __init__.py
│       ├── ohlcv_aggregator.py       # Resample 1m ticks → OHLCV(step_size)
│       └── feature_engineer.py       # log_returns, vol, skew, kurt, RSI, ATR, BB
│
├── strategies/
│   ├── __init__.py
│   ├── base.py                       # BaseStrategy(ABC): fit(), simulate(), suggest_params()
│   ├── registry.py                   # StrategyRegistry: register/get per (asset, regime)
│   ├── statistical/
│   │   ├── gbm.py                    # Geometric Brownian Motion (baseline)
│   │   ├── garch.py                  # GARCH(p,q) — arch lib
│   │   ├── gjr_garch.py              # GJR-GARCH — leverage effect
│   │   ├── egarch.py                 # EGARCH — equity
│   │   ├── figarch.py                # FIGARCH — long memory vol
│   │   ├── arima_garch.py            # ARIMA mean + GARCH vol combo
│   │   └── sabr.py                   # SABR stochastic vol (XAU)
│   ├── stochastic/
│   │   ├── heston.py                 # Heston — Euler-Maruyama
│   │   ├── merton_jump.py            # Merton jump diffusion
│   │   ├── kou_jump.py               # Kou double-exponential jump
│   │   └── ornstein_uhlenbeck.py     # OU mean-reversion (XAU)
│   ├── ml/
│   │   ├── lstm_vol.py               # LSTM → vol prediction → GARCH input
│   │   ├── transformer_vol.py        # Attention-based vol (HFT)
│   │   ├── xgboost_regime.py         # XGBoost regime classifier
│   │   └── gap_model.py              # Overnight gap distribution (equity)
│   ├── regime/
│   │   ├── hmm_detector.py           # HMM 3-state — hmmlearn
│   │   ├── pattern_detector.py       # Rule-based: RSI, BB, volume spike
│   │   └── regime_switcher.py        # HMM + pattern → regime state + routing
│   └── ensemble/
│       ├── weighted_ensemble.py      # Mix N model path-sets with w*
│       └── meta_learner.py           # Online EMA weight update from live CRPS
│
├── simulation/
│   ├── __init__.py
│   ├── engine.py                     # run() → ndarray (n_sim × n_timepoints), Numba JIT
│   ├── formatter.py                  # to_synth_format(), round 8 significant digits
│   └── local_validator.py            # check_format() → (ok, error_msg)
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py                     # BacktestEngine.run(config) → BacktestResult
│   ├── crps.py                       # crps_ensemble() — sort trick O(N log N)
│   └── metrics.py                    # rolling_leaderboard, softmax_emission_rank
│
├── training/
│   ├── __init__.py
│   ├── data_split.py                 # walk_forward_split() — time-ordered, no leakage
│   ├── feature_store.py              # Precompute + cache features to DuckDB
│   ├── optimizer.py                  # CRPSObjective + optuna_tune / pso / ga
│   ├── tuner.py                      # AssetTuner: L1→L2→L3 orchestration
│   └── experiment_tracker.py         # WandB / MLflow logging
│
├── visualization/
│   ├── __init__.py
│   ├── base_chart.py                 # AbstractChart(ABC)
│   ├── charts/
│   │   ├── price_candle.py
│   │   ├── path_fan.py
│   │   ├── crps_evolution.py
│   │   ├── distribution_compare.py
│   │   ├── regime_overlay.py
│   │   └── strategy_compare.py
│   └── dashboard.py                  # Streamlit app — 4 tabs
│
├── neurons/
│   └── miner.py                      # Bittensor Axon — entry point (DO NOT RENAME)
│
├── tests/
│   ├── test_crps.py                  # unit: crps sort trick == naive
│   ├── test_formatter.py             # unit: format + validate
│   ├── test_ohlcv.py                 # unit: aggregation correctness
│   ├── test_strategies.py            # smoke: simulate() → correct shape
│   ├── test_backtest.py              # integration: BacktestEngine
│   └── test_tuner.py                 # integration: tune GARCH on 30d BTC
│
├── scripts/
│   ├── fetch_history.py              # Backfill via Binance REST
│   ├── run_backtest.py               # CLI backtest runner
│   ├── run_tuner.py                  # CLI tuner
│   └── run_dashboard.py              # streamlit run
│
├── miner.config.js                   # PM2 — mainnet
├── miner.test.config.js              # PM2 — testnet (netuid 247)
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 9. Migration từ Codebase Hiện Tại

### Mapping cũ → mới

| File/Module cũ | File/Module mới | Ghi chú |
|----------------|-----------------|---------|
| `synth/miner/data_handler.py` | `data/fetchers/pyth_fetcher.py` + `data/storage/` | Tách fetch logic khỏi storage |
| `synth/miner/fetch_daemon.py` | `data/fetchers/base_fetcher.py` | Polling loop → ABC pattern |
| `synth/miner/data/price_data.db` (SQLite) | `data/storage/timeseries_db.py` (DuckDB) | Upgrade DB engine |
| `synth/miner/strategies/base.py` | `strategies/base.py` | Giữ nguyên interface, thêm `suggest_params()` |
| `synth/miner/strategies/registry.py` | `strategies/registry.py` | Thêm regime-aware routing |
| `synth/miner/strategies/garch_v1.py` | `strategies/statistical/garch.py` | Consolidate v1/v2 |
| `synth/miner/strategies/gjr_garch.py` | `strategies/statistical/gjr_garch.py` | Di chuyển trực tiếp |
| `synth/miner/strategies/egarch.py` | `strategies/statistical/egarch.py` | Di chuyển trực tiếp |
| `synth/miner/strategies/jump_diffusion.py` | `strategies/stochastic/merton_jump.py` | Rename + refactor |
| `synth/miner/strategies/mean_reversion.py` | `strategies/stochastic/ornstein_uhlenbeck.py` | Formalize OU model |
| `synth/miner/strategies/regime_switching.py` | `strategies/regime/hmm_detector.py` | Tách detection khỏi strategy |
| `synth/miner/strategies/ensemble_weighted.py` | `strategies/ensemble/weighted_ensemble.py` | Thêm CRPS-based weighting |
| `synth/miner/core/regime_detection.py` | `strategies/regime/regime_switcher.py` | Nâng cấp multi-asset |
| `synth/miner/simulations.py` | `simulation/engine.py` | Vectorize + Numba JIT |
| `synth/miner/my_simulation.py` | `simulation/engine.py` | Merge vào engine chính |
| `synth/miner/backtest/runner.py` | `backtest/engine.py` | Rolling window pattern |
| `synth/miner/backtest/metrics.py` | `backtest/metrics.py` | Thêm leaderboard sim |
| `synth/miner/backtest/tuner.py` | `training/tuner.py` | Tách riêng module training |
| `synth/miner/backtest/duel.py` | `backtest/engine.py` + `training/optimizer.py` | Tách backtest vs tuning |
| `synth/miner/compute_score.py` | `backtest/crps.py` | Sort trick optimization |
| `neurons/miner.py` | `neurons/miner.py` | **KHÔNG RENAME** — thêm LFT/HFT router |

### Config Migration

| Config cũ | Config mới | Nội dung |
|-----------|-----------|---------|
| Hard-coded trong code | `config/assets.yaml` | Emission weights, asset types |
| Hard-coded trong code | `config/strategies.yaml` | Best params per asset×regime |
| `.env` + `synth/utils/config.py` | `config/system.yaml` + `.env` | System settings, DB path |

### Dependencies mới cần thêm

```txt
# requirements.txt — additions
duckdb>=0.10.0              # Thay thế SQLite cho time-series
hmmlearn>=0.3.0             # HMM regime detection
optuna>=3.5.0               # Hyperparameter tuning
torch>=2.0.0                # LSTM / Transformer
xgboost>=2.0.0              # XGBoost regime classifier
streamlit>=1.30.0           # Visualization dashboard
plotly>=5.18.0              # Interactive charts
```

---

## 10. Hướng Dẫn Phát Triển

### Thêm Strategy Mới

1. Tạo file trong `strategies/statistical/`, `stochastic/`, hoặc `ml/`
2. Kế thừa `BaseStrategy`:

```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def fit(self, data: pd.DataFrame) -> None:
        """Fit model trên historical data."""
        ...

    def simulate(self, n_sim: int, horizon: int,
                 anchor: float, step: int) -> np.ndarray:
        """Return shape (n_sim, n_timepoints)."""
        ...

    def suggest_params(self, trial) -> dict:
        """Optuna trial → hyperparameter dict."""
        return {
            "param_a": trial.suggest_float("param_a", 0.01, 0.5),
            "param_b": trial.suggest_int("param_b", 1, 10),
        }
```

3. Đăng ký trong `strategies/registry.py`:
```python
StrategyRegistry.register("crypto", "bull", MyStrategy)
```

4. Chạy smoke test:
```bash
pytest tests/test_strategies.py -k "my_strategy"
```

### Chạy Backtest

```bash
python scripts/run_backtest.py \
    --asset BTC \
    --start 2025-01-01 \
    --end 2025-03-01 \
    --strategies garch,gjr_garch,heston \
    --n_sim 100
```

### Chạy Tuner

```bash
python scripts/run_tuner.py \
    --asset all \
    --n_trials 200 \
    --output config/strategies.yaml
```

### Chạy Dashboard

```bash
streamlit run scripts/run_dashboard.py
```

### Coding Conventions

| Quy tắc | Chi tiết |
|---------|---------|
| **Type hints** | Bắt buộc cho tất cả public functions |
| **Docstrings** | Google style, bao gồm `Args`, `Returns`, `Raises` |
| **Testing** | Mỗi strategy mới phải có smoke test |
| **Config** | Không hard-code params — dùng YAML config |
| **Simulation output** | Luôn trả `np.ndarray` shape `(n_sim, n_timepoints)` |
| **CRPS calculation** | Dùng basis-point returns, không dùng raw price |

---

> **Lưu ý quan trọng**: File `neurons/miner.py` là entry point của Bittensor — **KHÔNG ĐƯỢC ĐỔI TÊN**. Mọi logic mới phải được import vào file này, không thay đổi signature của Axon handler.
