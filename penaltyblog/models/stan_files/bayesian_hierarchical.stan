data {
        int N;                           // number of matches
        int n_teams;                     // number of teams
        array[N] int goals_home;         // home goals scored
        array[N] int goals_away;         // away goals scored
        array[N] int<lower=1,upper=n_teams> home_team;  // home team indices
        array[N] int<lower=1,upper=n_teams> away_team;  // away team indices
        vector[N] weights;               // match weights
    }
    parameters {
        real home;                    // home advantage
        real intercept;               // global intercept
        real<lower=0.001, upper=100> tau_att;
        real<lower=0.001, upper=100> tau_def;
        vector[n_teams] atts_star;    // raw attack parameters
        vector[n_teams] def_star;     // raw defense parameters
    }
    transformed parameters {
        vector[n_teams] atts;         // centered attack parameters
        vector[n_teams] defs;         // centered defense parameters
        vector[N] home_theta;         // home scoring rates
        vector[N] away_theta;         // away scoring rates

        atts = atts_star - mean(atts_star);
        defs = def_star - mean(def_star);

        for (i in 1:N) {
            home_theta[i] = exp(intercept + home + atts[home_team[i]] + defs[away_team[i]]);
            away_theta[i] = exp(intercept + atts[away_team[i]] + defs[home_team[i]]);
        }
    }
    model {
        tau_att ~ gamma(0.1, 0.1);
        tau_def ~ gamma(0.1, 0.1);
        atts_star ~ normal(0, inv_sqrt(tau_att));
        def_star ~ normal(0, inv_sqrt(tau_def));

        for (i in 1:N) {
            target += weights[i] * poisson_lpmf(goals_home[i] | home_theta[i]);
            target += weights[i] * poisson_lpmf(goals_away[i] | away_theta[i]);
        }
    }
    generated quantities {
        vector[N] log_lik;
        for (i in 1:N) {
            log_lik[i] = poisson_lpmf(goals_home[i] | home_theta[i]) +
                        poisson_lpmf(goals_away[i] | away_theta[i]);
        }
    }
