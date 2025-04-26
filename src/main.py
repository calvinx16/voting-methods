# src/main.py

from voting_methods.plurality import PluralityVoting
from voting_methods.borda import BordaCountVoting
from voting_methods.two_round import TwoRoundVoting
from voting_methods.condorcet import CondorcetVoting
from voting_methods.approval import ApprovalVoting
from analysis.bootstrap_analysis import BootstrapAnalyzer

DATA_PATH = "data/sample_voting_data.csv"

def main():
    print("\n--- Running Voting Methods ---\n")

    # Plurality Voting
    plurality = PluralityVoting(DATA_PATH)
    plurality_results = plurality.run_election()

    # Borda Count Voting
    borda = BordaCountVoting(DATA_PATH)
    borda_results = borda.run_election()

    # Two-Round Voting
    two_round = TwoRoundVoting(DATA_PATH)
    two_round_results = two_round.run_election()

    # Condorcet Voting
    condorcet = CondorcetVoting(DATA_PATH)
    condorcet_results = condorcet.run_election()

    # Approval Voting
    approval = ApprovalVoting(DATA_PATH, approval_threshold=3)
    approval_results = approval.run_election()

    print("\n--- Running Bootstrap Analysis (Plurality Voting) ---\n")
    bootstrap = BootstrapAnalyzer(PluralityVoting, DATA_PATH, num_bootstraps=500)
    boot_results = bootstrap.run_bootstrap()
    print("\nBootstrap Win Probabilities (Plurality Voting):")
    print(boot_results)

if __name__ == "__main__":
    main()
