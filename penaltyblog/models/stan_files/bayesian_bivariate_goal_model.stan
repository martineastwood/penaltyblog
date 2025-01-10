    data {
        int<lower=0> N;                 // Number of matches
        int<lower=1> n_teams;           // Number of teams
        array[N] int goals_home;         // home goals scored
        array[N] int goals_away;         // away goals scored
        array[N] int<lower=1,upper=n_teams> home_team;  // home team indices
        array[N] int<lower=1,upper=n_teams> away_team;  // away team indices
        vector[N] weights;               // match weights
    }

    parameters {
        real home;
        vector[n_teams] attack;
        vector[n_teams] defence;
        real<lower=0,upper=1> rho;
    }

    transformed parameters {
        vector[N] lambda_home;
        vector[N] lambda_away;

        for (i in 1:N) {
            lambda_home[i] = exp(home + attack[home_team[i]] - defence[away_team[i]]);
            lambda_away[i] = exp(attack[away_team[i]] - defence[home_team[i]]);
        }
    }

    model {
        // Priors
        home ~ normal(0, 1);
        attack ~ normal(0, 1);
        defence ~ normal(0, 1);
        rho ~ beta(2, 2);

        // Likelihood
        for (i in 1:N) {
            target += weights[i] * (poisson_log_lpmf(goals_home[i] | log(lambda_home[i])) +
                      poisson_log_lpmf(goals_away[i] | log(lambda_away[i])));
        }
    }
