# src/voting_methods/voting_system.py

import pandas as pd
import os

class VotingSystem:
    """
    Base class for voting systems.
    Handles loading and preprocessing of voter preference data.
    """

    def __init__(self, data_path: str):
        """
        Initialize the voting system with path to the voting data.
        Args:
            data_path (str): Path to the CSV file containing voting preferences.
        """
        self.data_path = data_path
        self.preferences = None
        self.candidates = []
        self._load_data()

    def _load_data(self):
        """
        Load voting preference data from CSV file.
        Assumes columns: Voter_ID, Rank_1, Rank_2, ..., Rank_n
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        self.preferences = df.drop(columns=["Voter_ID"])
        self.candidates = pd.unique(self.preferences.values.ravel())
        print(f"[INFO] Loaded {len(self.preferences)} voters and {len(self.candidates)} candidates.")

    def run_election(self):
        """
        Abstract method to run the election.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run_election().")
