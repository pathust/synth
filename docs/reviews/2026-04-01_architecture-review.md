# Review Kien Truc Code Hien Tai

Ngay review: 2026-04-01  
Reviewer: Codex (code-level + runtime command verification)

## 1. Pham vi va cach review

- Pham vi: `synth/miner` va cac tai lieu lien quan (`docs/*`) voi focus vao runtime path hien tai.
- Cach review:
  - Doc code thuc thi (entrypoint, data flow, strategy/config/backtest orchestration).
  - Doi chieu tai lieu architecture hien co voi implementation that.
  - Chay mot so lenh xac thuc runtime trong moi truong `conda -n synth`.

## 2. Tong ket nhanh

Kien truc hien tai da co du huong module hoa (entry, strategy registry, backtest runner, deploy exporter), nhung dang gap 3 van de lon:

1. Tai lieu kien truc va implementation dang lech nhau ro rang (dac biet data layer/storage).
2. Quyet dinh frequency/data resolution dang gan theo `asset` thay vi prompt/request context.
3. Tai lieu + template strategy da cu, gay nguy co tao strategy/tooling khong tuong thich.

## 3. Finding chi tiet (uu tien theo muc do)

### [P1] Data architecture thuc te khac mo ta tai lieu

**Bang chung**

- Tai lieu mo ta DuckDB la storage trung tam, miner/tuner/backtest doc DuckDB, backtest ghi session DB rieng:
  - `docs/ARCHITECTURE.md:58-63`
  - `docs/ARCHITECTURE.md:69-72`
- Code runtime thuc te dang load/save gia qua MySQL (`DataHandler -> MySQLHandler`):
  - `synth/miner/data_handler.py:437-464`
  - `synth/miner/data_handler.py:466-502`
- Fetch daemon cung ghi MySQL truoc, DuckDB chi la mirror tuy chon:
  - `synth/miner/fetch_daemon.py:111-128`
  - `synth/miner/fetch_daemon.py:134-148`

**Tac dong**

- Team de hieu nham data source "source of truth".
- De lam sai quy trinh debug (nhin DuckDB trong khi runtime doc MySQL).
- Nguy co loi van hanh khi handover/onboarding.

**Khuyen nghi**

- Chot ro data source chinh (MySQL hay DuckDB) trong 1 RFC ngan.
- Cap nhat `docs/ARCHITECTURE.md` theo implementation hien tai (hoac refactor code theo docs neu docs la target).
- Neu giu dual-store, can ghi ro ownership: writer chinh, reader chinh, va SLA sync.

---

### [P1] Chon time resolution theo asset, khong theo request context

**Bang chung**

- `UnifiedDataLoader` uu tien `HIGH_FREQUENCY` neu asset co high label:
  - `synth/miner/data/dataloader.py:22-26`
- `simulate_crypto_price_paths` goi dataloader khong truyen `time_increment/time_length`:
  - `synth/miner/my_simulation.py:141-162`

**Tac dong**

- Cac asset co ca high/low (`BTC/ETH/SOL/XAU`) co the bi nap lich su 1m ngay ca khi prompt la low (5m).
- Lam lech calibration strategy va tao do vong giua ket qua backtest va ket qua live.

**Khuyen nghi**

- Dataloader API can nhan explicit `time_increment` (hoac `prompt_label`) tu caller.
- Cam default theo asset trong path runtime.

---

### [P2] Interface strategy trong docs/template da stale so voi code

**Bang chung**

- Base class that su dung:
  - `supported_asset_types`, `supported_regimes`
  - `synth/miner/strategies/base.py:67-69`
- Docs + template van huong dan:
  - `supported_assets`, `supported_frequencies`
  - `docs/STRATEGY_SPEC.md:10-11`
  - `docs/templates/strategy_template.py:14-15`
  - `docs/DEVELOPER_ONBOARDING_GUIDE.md:88-89`

**Tac dong**

- Dev moi de tao strategy metadata sai -> filter asset/regime khong dung y.
- Tooling co su dung field cu se crash (da gap trong duel CLI, xem tai lieu task 2).

**Khuyen nghi**

- Dong bo docs/template theo BaseStrategy hien tai.
- Them "compat warning" trong registry neu strategy co field cu.

---

### [P2] Ton tai 2 data loading abstraction song song, ownership chua ro

**Bang chung**

- Co `UnifiedDataLoader` (runtime dang dung):
  - `synth/miner/my_simulation.py:136-162`
- Co `pipeline/PriceLoader` va `DataProvider` stack rieng:
  - `synth/miner/pipeline/loader.py:19-37`
  - `synth/miner/pipeline/base.py:12-21`

**Tac dong**

- Tang chi phi maintain.
- De xay ra behavior drift giua path "duoc noi la moi" va path "dang duoc goi that".

**Khuyen nghi**

- Chot 1 path chinh.
- Neu giu `pipeline/`, can chuyen caller thuc te sang dung va bo duong legacy.

## 4. Ban do kien truc "as-is" (thuc thi)

1. `neurons/miner.py` nhan request va goi `synth/miner/entry.py`.
2. `entry.py` lay routing tu `StrategyStore` (`strategies.yaml`) + strategy registry auto-discovery.
3. `entry.py` goi `simulate_crypto_price_paths(...)`.
4. `simulate_crypto_price_paths` lay du lieu qua `UnifiedDataLoader` (hien dang map tf theo asset label).
5. Strategy sinh path -> validate output -> fallback chain neu can.
6. Backtest runner su dung DataHandler cho history, va `cal_reward` (legacy) de tinh CRPS.

## 5. De xuat roadmap kien truc

### Phase A (1-2 ngay)

- Dong bo docs strategy contract.
- Sua dataloader de nhan explicit frequency/time_increment.

### Phase B (3-5 ngay)

- Thong nhat data layer (MySQL-first hoac DuckDB-first) va cap nhat architecture doc.
- Bo sung integration test cho path low/high dung data resolution.

### Phase C (1-2 tuan)

- Don gian hoa data pipeline (bo abstraction thua/khong duoc goi).
- Dat logging va metric canary de phat hien drift giua backtest va runtime.

## 6. Muc do tin cay review

- Da verify bang runtime command (conda env) cho cac loi tooling/behavior chinh.
- Cac finding kien truc deu co line-level reference trong code va docs hien tai.
