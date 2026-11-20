| metric | jepa mean ± std | none mean ± std | random mean ± std |
|---|---|---|---|
| agents_alive | 264.8 ± 172 | 0 ± 0 | 165.1 ± 123 |
| survival_rate | 0.4102 ± 0.139 | 0 ± 0 | 0.09789 ± 0.0655 |
| mean_energy | 48.67 ± 7.31 | nan ± nan | 40.89 ± 5.8 |
| mean_distance_km | 9776 ± 571 | 3156 ± 48.4 | 8869 ± 420 |
| food_gathered | 4.5e+04 ± 1.77e+04 | 2309 ± 107 | 1.008e+05 ± 3.03e+03 |
| settlements | 20.1 ± 15.6 | 0 ± 0 | 19.6 ± 12.6 |
| mean_action_norm | 1.023 ± 0.0198 | 0 ± 0 | 1.612 ± 0.00088 |
| mean_intensity | 0.314 ± 0.019 | 0.1 ± 1.03e-17 | 0.5078 ± 0.000849 |
| train_steps | 50 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.9348 ± 0.051 | nan ± nan | nan ± nan |
| wall_clock_s | 15.84 ± 8.62 | 1.153 ± 0.0607 | 22.98 ± 2.22 |

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
| wall_clock_s | 14.68 | [9.705, 19.87] | 0.00195 |

Paired differences jepa-random (n seeds = 10)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|
| agents_alive | 99.7 | [14.6, 183.3] | 0.105 |
| survival_rate | 0.3123 | [0.2351, 0.388] | 0.00195 |
| mean_energy | 7.783 | [1.915, 13.08] | 0.0371 |
| mean_distance_km | 907.2 | [461.6, 1288] | 0.00586 |
| food_gathered | -5.583e+04 | [-6.618e+04, -4.476e+04] | 0.00195 |
| settlements | 0.5 | [-8.6, 10.1] | 0.938 |
| mean_action_norm | -0.5892 | [-0.6005, -0.5769] | 0.00195 |
| mean_intensity | -0.1937 | [-0.2048, -0.1826] | 0.00195 |
| train_steps | 50 | [50, 50] | 0.00195 |
| final_train_loss | nan | [nan, nan] | nan |
| wall_clock_s | -7.147 | [-12.46, -1.486] | 0.0488 |
