# src/voting_methods/plurality.py

import pandas as pd
from .voting_system import VotingSystem

class PluralityVoting(VotingSystem):
    """
    Implements Plurality Voting System.
    """

    def run_election(self):
        """
        Run plurality voting.
        Each voter gives one vote to their top-ranked candidate.
        """
        # First column = voter's top preference
        first_choices = self.preferences.iloc[:, 0]
        results = first_choices.value_counts()
        winner = results.idxmax()

        print(f"[RESULT] Plurality Voting Winner: {winner}")
        return results
