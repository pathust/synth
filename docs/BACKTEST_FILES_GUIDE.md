# Backtest Files Guide

Tai lieu nay giai thich muc dich va cach dung 3 file:
- `synth/miner/backtest_strategies.py`
- `synth/miner/backtest_compare_simulations.py`
- `synth/miner/backtest_compare_from_db.py`

## 1) Nhanh gon: dung file nao khi nao?

- `backtest_strategies.py`
  - Dung khi muon benchmark **tung strategy** (garch_v4, jump_diffusion, ...).
  - Ket qua: bang CRPS theo strategy, tim strategy tot nhat cho moi asset.

- `backtest_compare_simulations.py`
  - Dung khi muon so sanh **2 module generate_simulations** (vd `simulations` vs `simulations_new_v3`) theo lich validator.
  - Ket qua: win/loss/tie theo slot + (tuy chon) rank so voi DB miners.

- `backtest_compare_from_db.py`
  - Dung khi muon danh gia rank tren **cac scored_time co that trong DB**.
  - Ket qua: top10/top50/top100/out>200 cho miner UID, theo high/low va theo asset.

---

## 2) `backtest_strategies.py`

### Muc dich
Chay backtest theo huong "model selection":
- Chon `asset`, `prompt` (`high`/`low`), `date` (hoac date range)
- Chay tat ca strategy (core + registry)
- Tinh CRPS theo cong thuc trong file (giong logic intervals cua validator)
- Xep hang strategy theo CRPS (lower is better)

### Dac diem quan trong
- Co the lay gia that tu:
  - DB (`use_db=True`, mac dinh)
  - Pyth API (`use_db=False`)
- Ho tro:
  - 1 asset hoac nhieu asset
  - 1 ngay hoac date range
- Prompt `high` co vong lap theo gio (24 slot), `low` danh gia theo ngay
- Luu JSON theo cau truc:
  - `result/backtest_strategies/{ASSET}/{YYYY_MM_DD}/backtest_strategies_{ASSET}_{prompt}_{YYYY_MM_DD}.json`

### Cach chay

Co 2 cach:

1) Chay truc tiep file (dang set san block cuoi file):
```bash
python -m synth.miner.backtest_strategies
```

2) Goi ham `main(...)` tu Python:
- phu hop khi ban muon script hoa batch test.

### Tham so chinh
- `asset`: `"BTC"` hoac `["BTC","ETH","XAU"]`
- `prompt_label`: `"high"` hoac `"low"`
- `date`: `"YYYY-MM-DD"` (neu khong dung range)
- `start_date`, `end_date`: chay date range (inclusive)
- `num_sims`: so simulation paths
- `seed`: seed ngau nhien
- `use_db`: `True/False`
- `strategies_to_test`: list strategy names de loc
- `output_dir`: thu muc output JSON

### Khi nao nen dung
- Toi uu chon strategy theo asset/prompt.
- Kiem tra strategy moi truoc khi dua vao `STRATEGY_LIST_FOR_ASSET`.

---

## 3) `backtest_compare_simulations.py`

### Muc dich
So sanh 2 module simulations tren cung request schedule cua validator:
- High: chu ky 12 phut (time_length=3600, inc=60)
- Low: chu ky 60 phut (time_length=86400, inc=300)

Moi slot:
- Chay `module_new.generate_simulations(...)`
- Chay `module_old.generate_simulations(...)`
- Tinh CRPS qua `cal_reward(...)`
- Ghi ket qua: ai thang, ai fail

### Dac diem quan trong
- Ho tro module:
  - `simulations`
  - `simulations_new`
  - `simulations_new_v2`
  - `simulations_new_v3`
  - `simulations_new_v4`
- Co cache output JSON (`reuse_if_exists`)
- Co the xep hang vs DB miners (`rank_vs_db=True`)
- Co the ve plot CRPS theo thoi gian (`plot_crps=True`)

### Cach chay nhanh (CLI)
```bash
python -m synth.miner.backtest_compare_simulations \
  --date 2026-03-20 \
  --module-new simulations_new_v3 \
  --module-old simulations \
  --asset BTC --asset ETH --asset XAU --asset SOL \
  --num-sims 1000 \
  --config-type both
```

### Tham so chinh
- `date`
- `assets` / `--asset` (lap lai duoc)
- `module_new`, `module_old`
- `config_type`: `both | high | low`
- `num_sims`, `seed`
- `max_high_slots`, `max_low_slots` (test nhanh)
- `rank_vs_db`, `refresh_db_cache`
- `plot_crps`

### Output
- JSON: `result/compare_simulations/compare_{module_new}_vs_{module_old}_{YYYY_MM_DD}.json`
- DB cache: `result/compare_simulations/db_crps_cache_{YYYY_MM_DD}.json`
- Plot (neu bat): trong `result/compare_simulations/...`

### Khi nao nen dung
- So sanh "A/B" giua 2 version simulations.
- Tra loi cau hoi: version moi co thang version cu theo lich validator khong?

---

## 4) `backtest_compare_from_db.py`

### Muc dich
Danh gia rank tren cac request **thuc te da co trong DB** (dua vao `scored_time`):
- Lay tat ca `(scored_time, time_length)` trong ngay tu MySQL
- Lay CRPS cua `miner_uid` trong DB
- Xep rank cua miner do so voi cac miner khac cung slot
- Thong ke top1-10, top11-50, top51-100, out>200, missing, no_db

### Dac diem quan trong
- Day la "benchmark theo du lieu thuc", khong phai schedule mo phong.
- Khong tin `time_increment` trong DB:
  - `time_length=3600` -> map `inc=60`
  - `time_length=86400` -> map `inc=300`
- Co cache JSON theo asset/ngay.
- Co ve bieu do rank theo thoi gian.

### Cach chay
File nay hien tai thiet ke theo kieu "chinh block cuoi file":
```bash
python synth/miner/backtest_compare_from_db.py
```

Sau do sua:
- `_DATE`
- `_ASSETS`
- `_MINER_UID`

### Output
- JSON:
  - `result/compare_from_db/{ASSET}/fromdb_{ASSET}_miner{UID}_rank_{YYYY_MM_DD}.json`
- Plot PNG:
  - `result/compare_from_db/{ASSET}/{YYYY_MM_DD}/fromdb_{ASSET}_miner{UID}_rank_{YYYY_MM_DD}.png`

### Khi nao nen dung
- Danh gia nang luc miner UID tren du lieu production.
- Theo doi rank theo thoi gian, theo high/low.

---

## 5) Quy trinh de nghi (practical workflow)

1. `backtest_strategies.py`
   - Tim strategy/phoi hop strategy tot nhat theo asset.

2. `backtest_compare_simulations.py`
   - Dong goi cac strategy vao module simulations moi, A/B voi baseline.

3. `backtest_compare_from_db.py`
   - Kiem chung ket qua tren scored_time/DB that, danh gia rank production.

---

## 6) Luu y quan trong

- CRPS cang thap cang tot.
- So sanh phai cung `num_sims`, `seed`, va cung data source (DB/Pyth) de cong bang.
- Neu ket qua dao dong manh:
  - tang date range
  - fix `seed`
  - su dung `max_high_slots/max_low_slots` chi de smoke test, khong de ket luan cuoi.
- `backtest_compare_simulations.py` co redirect stdout khi chay module -> log strategy ben trong co the khong hien het.

