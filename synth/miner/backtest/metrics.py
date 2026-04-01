from abc import ABC, abstractmethod
import numpy as np
from synth.validator.crps_calculation import (
    get_interval_steps,
    calculate_price_changes_over_intervals,
    label_observed_blocks,
)

class BaseEvaluator(ABC):
    """
    Abstract metric evaluator for BacktestEngine.
    """
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate(self, predictions: np.ndarray, truth: np.ndarray, time_increment: int, scoring_intervals: dict) -> float:
        pass


class CRPSEvaluator(BaseEvaluator):
    """
    Calculates the Continuous Ranked Probability Score (CRPS).
    Lower is better.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, predictions: np.ndarray, truth: np.ndarray, time_increment: int, scoring_intervals: dict) -> float:
        """
        predictions: shape (N_sims, T)
        truth: shape (T,)
        """
        # Ensure truth is 1D
        truth = truth.ravel()
        if len(truth) == 0 or predictions.size == 0 or np.any(predictions == 0):
            return -1.0
            
        return self._calculate_crps_with_formula(predictions, truth, time_increment, scoring_intervals)
        
    def _crps_point_formula(self, x: float, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        N = y.size
        if N == 0:
            return np.nan
        term1 = np.mean(np.abs(y - x))
        term2 = np.sum(np.abs(y[:, None] - y[None, :])) / (2.0 * (N ** 2))
        return float(term1 - term2)

    def _calculate_crps_with_formula(self, simulation_runs: np.ndarray, real_price_path: np.ndarray, time_increment: int, scoring_intervals: dict) -> float:
        sum_all_scores = 0.0
        real_path = np.asarray(real_price_path, dtype=float).ravel()

        for interval_name, interval_seconds in scoring_intervals.items():
            interval_steps = get_interval_steps(interval_seconds, time_increment)
            absolute_price = interval_name.endswith("_abs")
            is_gap = interval_name.endswith("_gap")

            if absolute_price:
                while (
                    real_path[::interval_steps].shape[0] == 1
                    and interval_steps > 1
                ):
                    interval_steps -= 1

            simulated_changes = calculate_price_changes_over_intervals(
                simulation_runs,
                interval_steps,
                absolute_price,
                is_gap,
            )
            real_changes = calculate_price_changes_over_intervals(
                real_path.reshape(1, -1),
                interval_steps,
                absolute_price,
                is_gap,
            )
            data_blocks = label_observed_blocks(real_changes[0])
            if len(data_blocks) == 0:
                continue

            for block in np.unique(data_blocks):
                if block == -1:
                    continue
                mask = data_blocks == block
                sim_block = simulated_changes[:, mask]
                real_block = real_changes[0, mask]
                num_t = sim_block.shape[1]
                for t in range(num_t):
                    x = float(real_block[t])
                    y = sim_block[:, t]
                    crps_t = self._crps_point_formula(x, y)
                    if np.isfinite(crps_t):
                        if absolute_price:
                            crps_t = crps_t / real_path[-1] * 10_000
                        sum_all_scores += crps_t

        return float(sum_all_scores)
