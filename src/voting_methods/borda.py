# src/voting_methods/borda.py

import pandas as pd
from .voting_system import VotingSystem

class BordaCountVoting(VotingSystem):
    """
    Implements Borda Count Voting System.
    """

    def run_election(self):
        """
        Run Borda count voting.
        Candidates receive points inversely proportional to their rank.
        """
        num_candidates = self.preferences.shape[1]
        borda_scores = pd.Series(0, index=self.candidates)

        for rank in range(num_candidates):
            points = num_candidates - rank
            votes_for_rank = self.preferences.iloc[:, rank]
            for candidate in votes_for_rank:
                borda_scores[candidate] += points

        borda_scores = borda_scores.sort_values(ascending=False)
        winner = borda_scores.idxmax()

        print(f"[RESULT] Borda Count Winner: {winner}")
        return borda_scores
