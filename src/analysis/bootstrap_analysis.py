# src/analysis/bootstrap_analysis.py

import pandas as pd
import numpy as np

class BootstrapAnalyzer:
    """
    Performs bootstrap sampling to evaluate voting system stability.
    """

    def __init__(self, voting_class, data_path: str, num_bootstraps: int = 1000):
        """
        Args:
            voting_class: A class implementing the VotingSystem interface.
            data_path (str): Path to the original voting data CSV.
            num_bootstraps (int): Number of bootstrap samples.
        """
        self.voting_class = voting_class
        self.data_path = data_path
        self.num_bootstraps = num_bootstraps

    def run_bootstrap(self) -> pd.Series:
        """
        Run bootstrap sampling and return winning frequencies.
        """
        winner_counts = {}

        for i in range(self.num_bootstraps):
            data = pd.read_csv(self.data_path)

            # Resample voters with replacement
            bootstrap_sample = data.sample(n=len(data), replace=True, random_state=None)
            bootstrap_sample_path = "outputs/temp_bootstrap_sample.csv"
            bootstrap_sample.to_csv(bootstrap_sample_path, index=False)

            # Run election on bootstrap sample
            voting_system = self.voting_class(bootstrap_sample_path)
            results = voting_system.run_election()

            if isinstance(results, pd.Series):
                winner = results.idxmax()
            else:
                winner = results  # e.g., Condorcet can return single name directly

            if winner not in winner_counts:
                winner_counts[winner] = 0
            winner_counts[winner] += 1

        winner_series = pd.Series(winner_counts).sort_values(ascending=False)
        return winner_series / self.num_bootstraps
