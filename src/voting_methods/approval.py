# src/voting_methods/approval.py

import pandas as pd
from .voting_system import VotingSystem

class ApprovalVoting(VotingSystem):
    """
    Implements Approval Voting System.
    """

    def __init__(self, data_path: str, approval_threshold: int = 3):
        """
        Args:
            approval_threshold (int): Candidates ranked within this threshold are considered "approved."
        """
        super().__init__(data_path)
        self.approval_threshold = approval_threshold

    def run_election(self):
        """
        Run approval voting.
        Each voter approves a set number of top candidates.
        """
        approvals = self.preferences.apply(lambda row: row[:self.approval_threshold], axis=1)
        flat_approvals = approvals.values.flatten()
        approval_counts = pd.Series(flat_approvals).value_counts()
        winner = approval_counts.idxmax()

        print(f"[RESULT] Approval Voting Winner: {winner}")
        return approval_counts
