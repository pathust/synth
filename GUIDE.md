# Server migration guide (Synth)

This guide captures how this repository expects **databases**, **PM2**, and optional **tmux** to be arranged, and how to move a working setup to a new machine with more CPU.

## 1. What runs where (inventory)

| Layer | Role | Technology |
|--------|------|------------|
| **Price / metagraph data** | Source of truth for OHLCV, validation scores, leaderboard tables | **MySQL/MariaDB** (see `synth/miner/mysql_handler.py`, `fetch_daemon.py`) |
| **Validator persistence** | Miner scores, predictions, metagraph history, etc. | **PostgreSQL** + SQLAlchemy (`synth/db/models.py`, `synth/validator/miner_data_handler.py`, Alembic) |
| **Optional analytics mirror** | Fast local reads (disabled by default in PM2) | **DuckDB** file, path `SYNTH_DUCKDB_PATH` (default `data/market_data.duckdb`) |
| **Long-running processes** | Miner, validator cycles, fetch daemon | **PM2** (`*.config.js`, `Makefile` `fetch` target) |
| **Local dev / localnet** | Detached shells for subtensor + miner/validator | **tmux** (`scripts/install_staging.sh` only — not required for production) |

**Docker:** `docker-compose.yaml` defines `mysql` (MariaDB 10.4), `postgres` (15.8), one-shot `migrations` (Alembic `upgrade head`), and `app` (validator image) with `POSTGRES_HOST=postgres` and `~/.bittensor` mounted.

---

## 2. Environment variables (copy to the new server)

Copy from `.env.example` and fill secrets on the new host. Minimum sets:

**PostgreSQL (validator + Alembic)**

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

**MySQL (miner fetch path)**

- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `MYSQL_ROOT_PASSWORD` if you administer the server with root

**Optional**

- `WANDB_*`, `VALIDATOR_*`, `NETWORK`, `NETUID`, `LOG_ID_PREFIX` as in `.env.example`
- `SYNTH_ENABLE_DUCKDB_WRITER` — `1` to mirror prices to DuckDB after fetch (default in code is `1`; `fetch.config.js` sets `0` to avoid a hard DuckDB dependency)
- `SYNTH_DUCKDB_PATH` — DuckDB file path if mirroring is enabled
- `SYNTH_ENTRY_IMPL` — miner entry implementation (`neurons/miner.py`)

**Hostname quirk:** If `HOSTNAME` is set and `MYSQL_HOST` is not, `MySQLHandler` uses host `synth-mysql` (Docker service name). For a bare-metal DB on localhost, set `MYSQL_HOST` explicitly (e.g. `127.0.0.1`).

---

## 3. Databases: backup on the old server

### 3.1 MySQL (`synth_prices` by default)

Logical dump (adjust user/host/password):

```bash
mysqldump -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
  --single-transaction --routines --triggers \
  "$MYSQL_DB" > synth_mysql_backup.sql
```

Restore on the new server (after DB and user exist):

```bash
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" < synth_mysql_backup.sql
```

Docker volume users can instead `docker exec` `mysqldump` / `mysql` against `synth-mysql`, or archive the volume if you prefer a physical copy (plan for same MariaDB/MySQL major version).

### 3.2 PostgreSQL (`synth` DB name in `.env.example`)

Custom format (recommended):

```bash
pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -Fc -f synth_pg_backup.dump "$POSTGRES_DB"
```

Restore:

```bash
pg_restore -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists synth_pg_backup.dump
```

On a **fresh** empty database you can use `pg_restore -C` to create the DB from the dump if your dump includes it; otherwise create the role/DB first to match `.env`.

### 3.3 Schema migrations (PostgreSQL)

After restore **or** on a new empty Postgres, align schema with the repo:

```bash
# From repo root, with .env loaded
alembic upgrade head
```

The Docker `migrations` service runs the same (`Dockerfile.migrations`).

### 3.4 Files beside SQL

| Path | Purpose |
|------|---------|
| `~/.bittensor/` | Coldkeys, hotkeys, subnet state — **must** move securely if you keep the same on-chain identity |
| `data/market_data.duckdb` | Optional DuckDB mirror — can rebuild from MySQL via fetch if omitted |
| `data/features/` | Parquet feature store (if used) |
| `result/` | Tuning/backtest outputs — optional |
| `config/strategies.yaml` (and related) | Live miner routing — copy if you rely on tuned strategies |

---

## 4. PM2: what the repo starts

Install: `sudo npm install pm2 -g` (see `docs/validator_guide.md` / `docs/miner_tutorial.md`).

| File | Apps / notes |
|------|----------------|
| `fetch.config.js` | `fetch-daemon` — `synth/miner/fetch_daemon.py`, interpreter `.venv/bin/python`, `PYTHONPATH=.`, args `--workers 2` (raise on a larger box if API limits allow) |
| `validator.config.js` | Three apps: `validator cycle low`, `validator cycle high`, `validator cycle scoring` — each uses `neuron.nprocs 8` in checked-in args |
| `miner.config.js` | Single `miner` — adjust `--wallet.*`, `--axon.port`, netuid to your deployment |
| `miner.dev.config.js` | Local dev miner |
| `Makefile` | `make fetch` → `pm2 start fetch.config.js`; `make validator` uses a different `pm2 start` pattern with explicit flags |

**Save and restore PM2 list on the old host:**

```bash
pm2 save
# copy ~/.pm2/dump.pm2 or use `pm2 cleardump` on new machine and re-add apps
```

On the **new** host, after repo clone, venv, and `.env`:

```bash
cd /path/to/synth
source .venv/bin/activate   # ensure deps installed per project README/requirements
pm2 start fetch.config.js    # if you run fetch here
pm2 start miner.config.js    # or validator.config.js — match your role
pm2 save
pm2 startup   # optional: systemd integration for reboot persistence
```

Edit `*.config.js` **before** `pm2 start` if wallet names, ports, netuid, or `interpreter` paths differ (the repo assumes `python3` on `PATH` for validators/miners and `.venv/bin/python` for fetch).

---

## 5. tmux (optional)

This repo does **not** use tmux for the main miner/validator/fetch path. `scripts/install_staging.sh` starts a **localnet** in a session named `localnet` and can split panes for miner/validator against `ws://127.0.0.1:9946`. Use that only for local chain development, not as a substitute for PM2 in production.

---

## 6. Using more CPU on the new server

1. **Validator** — In `validator.config.js`, each process passes `--neuron.nprocs 8`. Increase toward your core count (leave headroom for OS + fetch + DB). Replicate the same flag if you start validators via `Makefile` or raw `pm2 start ./neurons/validator.py -- ...`.
2. **Fetch daemon** — `fetch.config.js` uses `--workers 2`. Increase cautiously to avoid API rate limits; `MAX_WORKERS` inside `fetch_daemon.py` also caps internal parallelism.
3. **MySQL / PostgreSQL** — More cores help concurrent queries; tune `innodb_buffer_pool_size`, Postgres `shared_buffers`, and connection limits separately from this repo.
4. **Parallel tuning jobs** — Scripts such as `scripts/tune_crypto_low_pm2.sh` run heavy scans; you can start multiple PM2 jobs with different asset sets or use OS-level `taskset` only if you need hard CPU isolation.

---

## 7. Suggested migration checklist

1. **Stop** PM2 apps on the old server (`pm2 stop all` or per app) to quiesce writers; stop Docker stack if used (`docker compose down`).
2. **Backup** MySQL and PostgreSQL as above; copy `~/.bittensor` and any `data/`, `config/`, `result/` you need.
3. **Provision** the new server: OS deps, Python venv, Node+PM2, Docker (optional), firewall rules for axon/subtensor ports.
4. **Restore** databases; run `alembic upgrade head` if the Postgres restore is empty or behind.
5. **Clone** repo, install dependencies, copy `.env` (never commit secrets).
6. **Start** DBs (Docker or systemd), then PM2 apps; verify `pm2 logs` and DB connectivity.
7. **DNS / firewall** — Point validators/miners to the new public IP if applicable; update any allowlists.
8. **Decommission** the old server after burn-in.

---

## 8. Quick reference: files

- `docker-compose.yaml` — MySQL + Postgres + migrations + app
- `fetch.config.js`, `validator.config.js`, `miner.config.js` — PM2
- `synth/miner/mysql_handler.py` — MySQL env defaults
- `synth/db/models.py` — Postgres URL
- `alembic/` — PostgreSQL migrations
- `docs/ARCHITECTURE.md` — data flow (MySQL primary, optional DuckDB mirror)

If anything in this guide disagrees with your **actual** production commands, treat the live PM2 list (`pm2 list`, `pm2 show <name>`) and `.env` on the source host as the source of truth.
