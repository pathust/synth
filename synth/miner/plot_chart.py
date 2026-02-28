import datetime
import numpy as np
import matplotlib.pyplot as plt


def plot_sim_paths(
    prices_sim: np.ndarray,
    real_prices: np.ndarray,
    times=None,
    max_paths: int = 50,
    save_path: str = "",
):
    """
    prices_sim: (n_sims, steps+1)
    real_prices: (steps+1,)
    """

    n_sims = prices_sim.shape[0]
    idx = np.random.choice(n_sims, min(max_paths, n_sims), replace=False)

    plt.figure(figsize=(12, 6))

    for i in idx:
        plt.plot(prices_sim[i], color="gray", alpha=0.15)

    plt.plot(real_prices, color="black", linewidth=2, label="Real price")

    plt.title("Simulation paths vs Real price")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)

def load_logs_from_dir(log_dir: str):
    records = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(log_dir, fname), "r") as f:
            records.append(json.load(f))
    dates = sorted([datetime.datetime.fromisoformat(r["date"]) for r in records])
    
    diffs = [(dates[i] - dates[i-1]).total_seconds() for i in range(1, len(dates))]
    time_increment = diffs[0]
    assert all(diff == time_increment for diff in diffs), "Time increment is not constant"

    return records, time_increment, dates
def plot_crps_over_time(records, real_prices, save_path: str = ""):
    dates = []
    scores = []

    for r in records:
        if "date" not in r or "score" not in r:
            continue
        dates.append(datetime.datetime.fromisoformat(r["date"]))
        scores.append(float(r["score"]))

    dates = np.array(dates)
    scores = np.array(scores)

    order = np.argsort(dates)
    dates = dates[order]
    scores = scores[order]
    
    fig, ax1 = plt.subplots(figsize=(12, 4))
    
    ax1.plot(dates, scores, color="blue", label="CRPS")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CRPS", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(dates, real_prices, color="black", linewidth=2, label="Real price")
    ax2.set_ylabel("Real Price", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    
    plt.title("CRPS (score) over time")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.savefig(save_path)

if __name__ == "__main__":
    import json, os
    from synth.miner.data_handler import  DataHandler, ValidatorRequest
    data_handler = DataHandler()

    # file_path = "synth/miner/logs/benchmark_test/BTC/v3/2025-12-13T00:00:00.json"
    # save_dir = file_path.replace("benchmark_test", "benchmark_test_plots").split("/")[:-1]
    # save_dir = "/".join(save_dir)
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = file_path.replace("benchmark_test", "benchmark_test_plots").replace(".json", ".png")
    # with open(file_path, "r") as f:
    #     log_dt = json.load(f)
    # plot_sim_paths(np.array(log_dt["prediction"][2]), np.array(log_dt["real_prices"]), save_path=save_path)



    asset = "BTC"
    log_dir  = f"synth/miner/logs/benchmark_test/{asset}/v3"
    save_dir = log_dir.replace("benchmark_test", "benchmark_test_plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "crps_over_time.png")

    
   

    records, time_increment, dates = load_logs_from_dir(log_dir)
    validator_request = ValidatorRequest(
        asset="BTC",
        start_time=dates[0],
        time_increment=time_increment,
        time_length=(dates[-1] - dates[0]).total_seconds(),
    )
    real_prices = data_handler.get_real_prices(**{"validator_request": validator_request})
    plot_crps_over_time(records, real_prices, save_path=save_path)