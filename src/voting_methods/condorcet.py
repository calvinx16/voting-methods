# src/voting_methods/condorcet.py

import pandas as pd
from .voting_system import VotingSystem

class CondorcetVoting(VotingSystem):
    """
    Implements Condorcet Method.
    """

    def run_election(self):
        """
        Run Condorcet method: check pairwise victories.
        Winner must beat every other candidate in head-to-head comparisons.
        """
        pairwise_wins = {candidate: 0 for candidate in self.candidates}

        for candidate1 in self.candidates:
            for candidate2 in self.candidates:
                if candidate1 == candidate2:
                    continue
                winner = self._pairwise_winner(candidate1, candidate2)
                if winner == candidate1:
                    pairwise_wins[candidate1] += 1

        # Condorcet winner must beat all other candidates
        total_opponents = len(self.candidates) - 1
        winner = [c for c, wins in pairwise_wins.items() if wins == total_opponents]

        if winner:
            print(f"[RESULT] Condorcet Winner: {winner[0]}")
            return winner[0]
        else:
            print("[RESULT] No Condorcet winner (cycle detected).")
            return None

    def _pairwise_winner(self, candidate1, candidate2):
        """
        Determines the winner between two candidates by comparing voter rankings.
        """
        candidate1_wins = 0
        candidate2_wins = 0

        for _, voter in self.preferences.iterrows():
            ranks = {v: i for i, v in enumerate(voter)}
            if ranks[candidate1] < ranks[candidate2]:
                candidate1_wins += 1
            else:
                candidate2_wins += 1

        return candidate1 if candidate1_wins > candidate2_wins else candidate2
