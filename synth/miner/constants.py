"""
constants.py

Copied from sn50/synth/miner/constants.py.
Defines asset lists used by periodic data fetching and simulation routing.
"""

ASSETS_PERIODIC_FETCH_PRICE_DATA = ["XAU", "GOOGLX", "TSLAX", "AAPLX", "NVDAX", "SPYX"]

STOCK_ASSETS = ["GOOGLX", "TSLAX", "AAPLX", "NVDAX", "SPYX"]

# Calendar days of OHLC loaded strictly before start_time (UnifiedDataLoader /
# simulate_crypto_price_paths, entry regime context). Must be >= any strategy
# lookback_days in strategies.yaml and strategy param grids (e.g. weekly_garch_v4
# up to 120d; weekly_regime_switching / har_rv / gjr_garch up to 60–90d; BTC
# garch_v4 yaml 45d; core grach_simulator_v2 low XAU 90d).
HISTORY_WINDOW_DAYS = 120
