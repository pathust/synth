from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for all trading and simulation strategies.
    Ensures a standardized interface for fitting models and generating price paths.
    """
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit(self, historical_data: np.ndarray) -> None:
        """
        Fit the model to historical data.
        
        Args:
            historical_data (np.ndarray): Historical price data. 
                                          Must strictly be out-of-sample relative to the simulation target.
        """
        pass

    @abstractmethod
    def simulate(self, asset: str, current_price: float, time_increment: int, time_length: int, num_simulations: int, seed: int) -> np.ndarray:
        """
        Generate multiple simulated price paths.
        
        Args:
            asset (str): The asset ticker (e.g., 'BTC').
            current_price (float): The starting price of the simulation.
            time_increment (int): Step size in seconds.
            time_length (int): Total length of the simulation in seconds.
            num_simulations (int): Number of independent paths to generate.
            seed (int): Random seed for reproducibility.
            
        Returns:
            np.ndarray: A 2D array of shape (num_simulations, num_steps) containing the simulated price paths.
        """
        pass
