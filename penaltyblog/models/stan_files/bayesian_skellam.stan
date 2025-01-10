    data {
        int<lower=0> N;                   // Number of games
        int<lower=1> n_teams;             // Number of teams
        array[N] int goals_home;          // home goals scored
        array[N] int goals_away;          // away goals scored
        array[N] int<lower=1,upper=n_teams> home_team;  // home team indices
        array[N] int<lower=1,upper=n_teams> away_team;  // away team indices
        vector[N] weights;                // match weights
    }

    parameters {
        real home;            // Home advantage effect
        vector[n_teams] attack;       // Attack strength per team
        vector[n_teams] defence;      // Defense strength per team
    }

    model {
        // Priors
        home ~ normal(0, 1);
        attack ~ normal(0, 0.5);
        defence ~ normal(0, 0.5);

        // Likelihood
        for (i in 1:N) {
            real lambda_home = exp(home + attack[home_team[i]] - defence[away_team[i]]);
            real lambda_away = exp(attack[away_team[i]] - defence[home_team[i]]);

            target += weights[i] * poisson_lpmf(goals_home[i] | lambda_home);
            target += weights[i] * poisson_lpmf(goals_away[i] | lambda_away);
        }
    }

    generated quantities {
        vector[N] goal_difference;
        for (i in 1:N) {
            goal_difference[i] = goals_home[i] - goals_away[i];
        }
    }
