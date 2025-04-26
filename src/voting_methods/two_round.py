# src/voting_methods/two_round.py

import pandas as pd
from .voting_system import VotingSystem

class TwoRoundVoting(VotingSystem):
    """
    Implements Two-Round Voting System.
    """

    def run_election(self):
        """
        First round: Plurality.
        Second round: Head-to-head between top two candidates.
        """
        # First round: Plurality
        first_choices = self.preferences.iloc[:, 0]
        first_round = first_choices.value_counts()
        top_two = first_round.nlargest(2).index.tolist()

        if len(top_two) < 2:
            print("[INFO] Only one candidate received votes.")
            return first_round

        # Second round: head-to-head
        top_two_votes = self.preferences.apply(lambda row: row[row.isin(top_two)].iloc[0], axis=1)
        final_round = top_two_votes.value_counts()
        winner = final_round.idxmax()

        print(f"[RESULT] Two-Round Voting Winner: {winner}")
        return final_round
