| metric | jepa mean ± std | none mean ± std | random mean ± std |
|---|---|---|---|
| agents_alive | 264.8 ± 172 | 0 ± 0 | 300.4 ± 247 |
| survival_rate | 0.4102 ± 0.139 | 0 ± 0 | 0.4327 ± 0.216 |
| mean_energy | 48.67 ± 7.31 | nan ± nan | 53.36 ± 7.85 |
| mean_distance_km | 9776 ± 571 | 3156 ± 48.4 | 9107 ± 1.38e+03 |
| food_gathered | 4.5e+04 ± 1.77e+04 | 2309 ± 107 | 4.432e+04 ± 1.94e+04 |
| settlements | 20.1 ± 15.6 | 0 ± 0 | 19.5 ± 19.6 |
| mean_action_norm | 1.023 ± 0.0198 | 0 ± 0 | 1.016 ± 0.00145 |
| mean_intensity | 0.314 ± 0.019 | 0.1 ± 1.03e-17 | 0.3256 ± 0.00154 |
| train_steps | 50 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.9348 ± 0.051 | nan ± nan | nan ± nan |
| wall_clock_s | 15.64 ± 8.58 | 1.136 ± 0.0424 | 6.229 ± 4.05 |

Paired differences jepa-none (n seeds = 10)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|
| agents_alive | 264.8 | [167.2, 370.1] | 0.00195 |
| survival_rate | 0.4102 | [0.3289, 0.4904] | 0.00195 |
| mean_energy | nan | [nan, nan] | nan |
| mean_distance_km | 6620 | [6266, 6941] | 0.00195 |
| food_gathered | 4.269e+04 | [3.252e+04, 5.339e+04] | 0.00195 |
| settlements | 20.1 | [10.9, 29.4] | 0.00195 |
| mean_action_norm | 1.023 | [1.012, 1.035] | 0.00195 |
| mean_intensity | 0.214 | [0.2029, 0.2253] | 0.00195 |
| train_steps | 50 | [50, 50] | 0.00195 |
| final_train_loss | nan | [nan, nan] | nan |
| wall_clock_s | 14.51 | [9.578, 19.64] | 0.00195 |

Paired differences jepa-random (n seeds = 10)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|
| agents_alive | -35.6 | [-214, 120.6] | 1 |
| survival_rate | -0.02244 | [-0.1466, 0.1098] | 0.625 |
| mean_energy | -4.682 | [-9.649, 0.1881] | 0.16 |
| mean_distance_km | 669.6 | [47.81, 1426] | 0.131 |
| food_gathered | 678.3 | [-1.293e+04, 1.38e+04] | 0.922 |
| settlements | 0.6 | [-14.4, 13.3] | 0.625 |
| mean_action_norm | 0.007582 | [-0.003558, 0.01972] | 0.193 |
| mean_intensity | -0.01152 | [-0.023, -4.833e-05] | 0.16 |
| train_steps | 50 | [50, 50] | 0.00195 |
| final_train_loss | nan | [nan, nan] | nan |
| wall_clock_s | 9.413 | [4.349, 14.67] | 0.0137 |
