# simulations_new_v3 Strategies Explained

Tai lieu nay giai thich chi tiet cac strategy dang duoc su dung trong `synth/miner/simulations_new_v3.py`, bao gom:
- Cac tham so dau vao cua `generate_simulations`
- Co che ensemble + trimming + fallback
- Y nghia tung strategy va cac tham so quan trong

## 1) Tham so dau vao cua `generate_simulations`

Ham:
- `generate_simulations(simulation_input, asset, start_time, time_increment, time_length, num_simulations, seed, version)`

Y nghia:
- `asset`: ma tai san (`BTC`, `ETH`, `SOL`, `XAU`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`, `SPYX`)
- `start_time`: moc thoi gian bat dau du bao (ISO format)
- `time_increment`: buoc thoi gian moi diem du bao
  - `60` = 1m (high frequency)
  - `300` = 5m (low frequency)
- `time_length`: tong do dai du bao
  - `3600` = high prompt (1h)
  - `86400` = low prompt (24h)
- `num_simulations`: so path dau ra can lay
- `seed`: base seed cho toan bo request
- `simulation_input`: duoc dung de validate format response truoc khi tra ket qua

## 2) Co che chon strategy trong v3

### 2.1 Prompt type
- `time_length == 3600` -> `high`
- nguoc lai -> `low`

### 2.2 Strategy list theo asset/prompt
Trong `STRATEGY_LIST_FOR_ASSET`, moi cap `(asset, prompt)` map sang danh sach `(strategy_name, weight)`.

High:
- `BTC`: `garch_v2_2 (1.0)`
- `ETH`: `garch_v2_2 (1.0)`
- `XAU`: `garch_v4_2 (1.0)`
- `SOL`: `garch_v4 (1.0)`

Low:
- `BTC`: `garch_v4 (0.4) + garch_v2_2 (0.3) + gjr_garch (0.3)`
- `ETH`: `garch_v4 (0.4) + garch_v2_2 (0.3) + regime_switching (0.3)`
- `XAU`: `jump_diffusion (0.4) + garch_v4_1 (0.3) + weekly_regime_switching (0.3)`
- `SOL`: `garch_v2_2 (0.4) + garch_v4 (0.3) + regime_switching (0.3)`
- `NVDAX`: `arima_equity (0.4) + garch_v4_1 (0.3) + markov_garch_jump (0.3)`
- `TSLAX`: `weekly_garch_v4 (0.4) + garch_v4 (0.3) + regime_switching (0.3)`
- `AAPLX`: `markov_garch_jump (0.4) + regime_switching (0.3) + garch_v4_1 (0.3)`
- `GOOGLX`: `gjr_garch (0.4) + regime_switching (0.3) + garch_v4_1 (0.3)`
- `SPYX`: `weekly_regime_switching (0.4) + garch_v4 (0.3) + arima_equity (0.3)`

### 2.3 Bang tra nhanh lookback theo tung dong (ban active trong v3)

Luu y:
- `high` = 1m/1h (`time_increment=60`, `time_length=3600`)
- `low` = 5m/24h (`time_increment=300`, `time_length=86400`)
- 1 ngay 1m = 1440 nen; 1 ngay 5m = 288 nen

| Asset | Prompt | Strategy chinh | lookback_days thuc dung | So nen tuong ung |
|---|---|---|---:|---:|
| BTC | high | garch_v2_2 | 3.0 | 4320 (1m) |
| ETH | high | garch_v2_2 | 3.0 | 4320 (1m) |
| XAU | high | garch_v4_2 | Short=14, Long=60 | 20160 va 86400 (1m) |
| SOL | high | garch_v4 | 7 | 10080 (1m) |
| BTC | low | garch_v4 + garch_v2_2 + gjr_garch | 25 / 30 / 45(mac dinh) | 7200 / 8640 / 12960 (5m) |
| ETH | low | garch_v4 + garch_v2_2 + regime_switching | 20 / 30 / 30(mac dinh) | 5760 / 8640 / 8640 (5m) |
| XAU | low | jump_diffusion + garch_v4_1 + weekly_regime_switching | 30(mac dinh) / 25 / theo core weekly | 8640 / 7200 / tuy core (5m) |
| SOL | low | garch_v2_2 + garch_v4 + regime_switching | 20 / 15 / 30(mac dinh) | 5760 / 4320 / 8640 (5m) |
| NVDAX | low | arima_equity + garch_v4_1 + markov_garch_jump | market-hours ARIMA / 30 / 0 (hardcoded priors) | tuy strategy |
| TSLAX | low | weekly_garch_v4 + garch_v4 + regime_switching | weekly core / 15 / 30(mac dinh) | tuy strategy |
| AAPLX | low | markov_garch_jump + regime_switching + garch_v4_1 | 0 (hardcoded priors) / 30(mac dinh) / 40 | tuy strategy |
| GOOGLX | low | gjr_garch + regime_switching + garch_v4_1 | 45(mac dinh) / 30(mac dinh) / 40 | 12960 / 8640 / 11520 (5m) |
| SPYX | low | weekly_regime_switching + garch_v4 + arima_equity | weekly core / 45 / market-hours ARIMA | tuy strategy |

Ghi chu:
- `markov_garch_jump` voi stock (NVDAX/TSLAX/AAPLX/GOOGLX/SPYX) dang dung `lookback_days=0` theo `ASSET_PRIORS`, tuc la uu tien bo tham so hardcoded thay vi auto-fit lookback.
- `weekly_regime_switching`, `weekly_garch_v4`, `arima_equity` co logic gio giao dich va weekly profile trong core simulator, nen khong co 1 gia tri lookback_days co dinh de ghi 1 so duy nhat.

## 3) Co che ensemble (Phase 1)

File v3 dung cac nguyen tac sau:
- `_ENSEMBLE_TOP_N = 3`: chi lay toi da 3 strategy dau trong list
- Chuan hoa lai trong so (`weight / tong_weight`)
- Over-request `10%`:
  - `target_total_sims = int(num_simulations * 1.10)`
- Moi strategy duoc cap so path rieng theo trong so
- Moi strategy duoc tao `sub_seed` rieng tu `seed` + ten strategy
- Gom tat ca path bang `np.vstack`
- Trim outlier theo tong return path:
  - giu trong khoang percentile `1% -> 99%`
- Chon lai dung `num_simulations` path
- Validate format qua `validate_responses`; neu dung -> tra ket qua ngay

## 4) Co che fallback (Phase 2)

Neu ensemble fail (hoac fail validate), v3 chay lan luot theo:
- Danh sach strategy dang active cua asset/prompt
- Cong them `DEFAULT_FALLBACK_CHAIN` (neu chua co):
  - `garch_v2`, `garch_v4`, `garch_v4_1`, `garch_v4_2`, `egarch`, `garch_v2_1`, `seasonal_stock`, `garch_v1`, `har_rv`

Moi strategy fallback duoc chay tuan tu; strategy dau tien cho ket qua hop le se duoc dung.

## 5) Y nghia tham so quan trong (ap dung chung cho nhieu strategy)

- `lookback_days`: so ngay lich su dung de fit model
  - So diem du lieu thuc te = `lookback_days * (86400 // time_increment)`
- `mean_model`: cach model hoa drift (`Zero` hoac `Constant`)
- `dist`: phan phoi shock (`StudentsT`, `skewstudent`, ...)
- `min_nu`: san cho bac tu do cua Student-t (nu thap => duoi day hon)
- `vol_multiplier`: he so phong/thu hep bien do volatility du bao
- `p, q, o`: bac cua GARCH/GJR-GARCH (`o=1` bat asymmetry leverage)
- `scale`: he so scale return truoc khi fit de optimizer on dinh hon

## 6) Chi tiet tung strategy dang duoc su dung trong v3

Luu y: mot so strategy la wrapper va tham so chinh nam o core simulator.

### 6.1 `garch_v2_2`
- File strategy: `synth/miner/strategies/garch_v2_2.py`
- Core: `synth/miner/core/garch_simulator_v2_2.py`
- Mo ta:
  - GARCH(1,1) toi uu theo asset cho HFT, co tuning rieng cho tung tai san.
- Tham so/noi dung quan trong:
  - `lookback_days`, `min_nu`, `vol_multiplier`, `scale`, `dist`, `mean_model`
  - Co profile rieng cho BTC/ETH/SOL/XAU va stock tokenized.
- Khi hop:
  - Du lieu high-frequency can model nhanh, on dinh, bat duoc fat-tail.

Bang config theo asset trong core `garch_simulator_v2_2.py`:

| Asset | lookback_days (high/low) | min_nu | vol_multiplier | scale dac biet |
|---|---:|---:|---:|---:|
| BTC | 3.0 / 30.0 | 3.0 | 1.00 | 100 |
| ETH | 3.0 / 30.0 | 3.0 | 1.02 | 100 |
| SOL | 2.5 / 20.0 | 2.5 | 1.08 | 100 |
| XAU | 3.9 / 30.0 | 4.0 | 0.95 | 1000 |
| NVDAX | 20.0 / 20.0 | 3.0 | 1.10 | 100 |
| TSLAX | 20.0 / 20.0 | 3.0 | 1.10 | 100 |
| AAPLX | 30.0 / 30.0 | 4.5 | 0.95 | 100 |
| GOOGLX | 20.0 / 20.0 | 3.5 | 1.08 | 100 |
| SPYX | 15.0 / 15.0 | 4.0 | 0.95 | 100 |

### 6.2 `garch_v4`
- Wrapper: `synth/miner/strategies/garch_v4.py`
- Core: `synth/miner/strategies/grach_simulator_v4.py`
- Mo ta:
  - GJR-GARCH + Skew Student-t + FHS + regime drift.
- Dung khi:
  - Can bat asymmetry (tin xau/flash dump gay vol manh hon).

Bang lookback trong `grach_simulator_v4.py` (theo `get_optimal_config`):

| Asset | lookback_days high | lookback_days low | mean_model |
|---|---:|---:|---|
| XAU | 15 | 30 | Constant |
| BTC | 7 | 25 | Zero |
| ETH | 7 | 20 | Zero |
| SOL | 7 | 15 | Zero |
| GOOGLX | 7 | 15 | Constant |
| NVDAX | 7 | 25 | Constant |
| TSLAX | 7 | 15 | Constant |
| AAPLX | 7 | 45 | Constant |
| SPYX | 7 | 45 | Constant |

### 6.3 `garch_v4_1`
- Core: `synth/miner/strategies/grach_simulator_v4_1.py`
- Tham so noi bat:
  - `dist` mac dinh skewt
  - `mean_model` va `lookback_days` thay doi theo asset/frequency
- Dung khi:
  - Can bien the v4 on dinh hon cho mot so asset/khung thoi gian.

Bang lookback trong `grach_simulator_v4_1.py`:

| Nhom asset | lookback_days high | lookback_days low | mean_model |
|---|---:|---:|---|
| XAU/GOLD | 10 | 25 | Constant |
| Khac (crypto + stock) | 7 | 40 | Zero |

### 6.4 `garch_v4_2`
- Core: `synth/miner/strategies/grach_simulator_v4_2.py`
- Mo ta:
  - Ensemble noi bo 2 cau hinh:
    - `Short_Term`: `lookback_days` ngan, `mean_model=Constant`
    - `Long_Term`: `lookback_days` dai, `mean_model=Zero`
  - Dist uu tien `skewstudent`, retry sang `studentst` / `normal` khi fit kho.
- Dung khi:
  - Asset can pha tron memory ngan + dai de giu can bang responsiveness/robustness.

Chi tiet lookback noi bo cua `garch_v4_2`:

| Nhom asset | Config | lookback_days high | lookback_days low | mean_model |
|---|---|---:|---:|---|
| Crypto | Short_Term | 5 | 14 | Constant |
| Crypto | Long_Term | 30 | 60 | Zero |
| XAU/Gold | Short_Term | 14 | 14 | Constant |
| XAU/Gold | Long_Term | 60 | 60 | Zero |

Luu y:
- `garch_v4_2` la 2-model ensemble noi bo nen 1 request se fit 2 cua so lookback khac nhau (khong phai 1 so duy nhat).
- Ban da them fallback noi bo sang `garch_v2_2` khi ensemble noi bo fail.

### 6.5 `gjr_garch`
- File: `synth/miner/strategies/gjr_garch.py`
- Mo ta:
  - GJR-GARCH co leverage term `o=1`, phu hop du lieu bat doi xung.
- Tham so:
  - `lookback_days`, `mean_model`, `vol_multiplier`

### 6.6 `regime_switching`
- File: `synth/miner/strategies/regime_switching.py`
- Mo ta:
  - Chuyen regime dua tren trend/vol de dieu chinh profile mo phong.
- Tham so:
  - `lookback_days`
  - Muc `vol_multiplier` thay doi theo regime phat hien

### 6.7 `jump_diffusion`
- File: `synth/miner/strategies/jump_diffusion.py`
- Mo ta:
  - Merton Jump-Diffusion: GBM + Poisson jumps, bat discontinuity.
- Tham so:
  - `lookback_days`
  - `jump_intensity_override`, `jump_mean_override`, `jump_std_override`
- Dung khi:
  - Xuat hien jump events (tin tuc, gap, flash move).

### 6.8 `weekly_regime_switching`
- File: `synth/miner/strategies/weekly_regime_switching.py`
- Core: `synth/miner/core/stock_simulator_v4.py`
- Mo ta:
  - Regime switching + weekly trading-hours masking cho stock tokenized.
- Dung khi:
  - Low-frequency stock, can ton trong gio giao dich va seasonal weekly pattern.

### 6.9 `arima_equity`
- File: `synth/miner/strategies/arima_equity.py`
- Core: `synth/miner/core/arima_equity_simulator.py`
- Mo ta:
  - ARIMA equity simulator co rang buoc gio giao dich My (regular/extended hours tuy asset).
- Dung khi:
  - Stock tokenized can "exact market hours" behavior.

### 6.10 `markov_garch_jump`
- File: `synth/miner/strategies/markov_garch_jump.py`
- Mo ta:
  - Markov-switching 2 regimes + GARCH + jump process.
- Tham so quan trong:
  - Chuyen trang thai: `P00`, `P11`
  - Regime 0/1 GARCH: `omega_0/1`, `alpha_0/1`, `beta_0/1`
  - Jump: `lambda_0/1`, `mu_J`, `sigma_J`
  - Student-t: `nu`
  - `lookback_days` (co asset priors hardcoded cho nhieu ma)

### 6.11 `weekly_garch_v4`
- File: `synth/miner/strategies/weekly_garch_v4.py`
- Core: `synth/miner/core/stock_simulator_v3.py`
- Mo ta:
  - GARCH V4 + weekly empirical seasonality (stock hours aware).

## 7) Strategy co the xuat hien trong fallback chain cua v3

Ngoai cac strategy chinh tren, fallback chain con co:
- `garch_v2` (core `grach_simulator_v2.py`)
- `garch_v2_1` (core `grach_simulator_v2_1.py`, co them regime detection)
- `garch_v1` (core `garch_simulator.py`, baseline)
- `har_rv` (core `HAR_RV_simulatior.py`, HAR-RV + GARCH)
- `seasonal_stock` (core `stock_simulator.py`, intraday seasonality + GARCH)
- `egarch` (strategy `egarch.py`, bat asymmetry theo dang log-variance)

## 8) Luu y van hanh quan trong

- Trong v3, ten strategy trong `STRATEGY_LIST_FOR_ASSET` co the den tu:
  - `SIMULATOR_FUNCTIONS` hardcoded
  - hoac tu `StrategyRegistry.auto_discover()` (dynamic)
- Vi vay, neu doi ten strategy/doi file strategy ma quyen registry thay doi, can dong bo:
  1) ten trong `STRATEGY_LIST_FOR_ASSET`
  2) ten strategy class (`name = ...`)
  3) fallback chain neu muon uu tien model nao khi fail

