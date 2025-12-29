"""
Quick test to verify kwargs are passed correctly to Plotly layout
"""

import numpy as np

from penaltyblog.models import BayesianGoalModel

if __name__ == "__main__":
    # Create minimal test data
    np.random.seed(42)
    teams = ["Team A", "Team B", "Team C"]
    n_matches = 20

    teams_home = np.random.choice(teams, n_matches)
    teams_away = np.random.choice(teams, n_matches)
    goals_home = np.random.poisson(1.5, n_matches)
    goals_away = np.random.poisson(1.2, n_matches)

    # Fit model
    model = BayesianGoalModel(
        goals_home=goals_home,
        goals_away=goals_away,
        teams_home=teams_home,
        teams_away=teams_away,
    )

    print("Fitting model...")
    model.fit(n_samples=200, burn=100, n_chains=2, thin=2)

    # Test that kwargs work
    print("\nTesting kwargs...")
    fig = model.plot_trace(
        width=1200, height=800, title="Custom MCMC Trace Plot", showlegend=True
    )

    # Check that the layout was updated
    assert fig.layout.width == 1200, f"Width not set correctly: {fig.layout.width}"
    assert fig.layout.height == 800, f"Height not set correctly: {fig.layout.height}"
    assert (
        fig.layout.title.text == "Custom MCMC Trace Plot"
    ), f"Title not set correctly: {fig.layout.title.text}"
    assert (
        fig.layout.showlegend == True
    ), f"Showlegend not set correctly: {fig.layout.showlegend}"

    print("✓ All kwargs passed correctly to Plotly layout!")
    print(f"  - width: {fig.layout.width}")
    print(f"  - height: {fig.layout.height}")
    print(f"  - title: {fig.layout.title.text}")
    print(f"  - showlegend: {fig.layout.showlegend}")
