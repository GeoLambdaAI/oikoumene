Config: 25 agents, 1000 ticks, train interval 2, seeds 0..29, random_scale 0.6026, const_intensity 0.4163

Note: Plan declared before any stage-2 data: modes jepa/random/const/none; random scale and const intensity matched to a 5-seed jepa pilot (seeds 1000-1004) per configuration; configs (i) 25 agents 1000 ticks 30 seeds, (iii) = (i) with train interval 2, (ii) 25 agents 3000 ticks 10 seeds without none. The plan was revised from the earlier 5000-tick/30-seed (ii) before any stage-2 data existed. A first run of (i) completed its simulations but crashed in the summary (MDE solver NaN) before writing any output; it was rerun unchanged after fixing the solver. Config (iii).

Pilot: {"seeds": [1000, 1001, 1002, 1003, 1004], "jepa_mean_action_norm": 0.9707382256434567, "jepa_mean_intensity": 0.41626923371916086, "expected_uniform_norm": 1.6110198104540456, "jepa_mean_abs_a2": 0.4101906126974016, "random_scale": 0.6025613213098021, "const_intensity": 0.41626923371916086}

| metric | jepa (JEPA planner) mean ± sd | none (no action vector) mean ± sd | random (random action (matched scale)) mean ± sd | const (no steering, constant intensity) mean ± sd |
|---|---|---|---|---|
| agents_alive | 323.7 ± 201 | 0 ± 0 | 263.7 ± 133 | 369.6 ± 142 |
| survival_rate | 0.2108 ± 0.163 | 0 ± 0 | 0.4598 ± 0.146 | 0.2734 ± 0.105 |
| mean_energy | 40.6 ± 6.14 | nan ± nan | 51.76 ± 4.37 | 38.88 ± 5.43 |
| mean_distance_km | 8773 ± 766 | 3139 ± 111 | 9531 ± 1.04e+03 | 9749 ± 625 |
| food_gathered | 9.207e+04 ± 1.57e+04 | 2325 ± 190 | 4.089e+04 ± 1.26e+04 | 8.831e+04 ± 1.3e+04 |
| settlements | 38.1 ± 21.6 | 0 ± 0 | 19.37 ± 11.6 | 47.67 ± 15.6 |
| mean_action_norm | 0.9867 ± 0.0147 | 0 ± 0 | 0.9716 ± 0.00127 | 0.4163 ± 7.71e-17 |
| mean_abs_a0 | 0.2656 ± 0.0162 | 0 ± 0 | 0.3008 ± 0.00171 | 0 ± 0 |
| mean_abs_a1 | 0.2609 ± 0.0158 | 0 ± 0 | 0.3012 ± 0.00165 | 0 ± 0 |
| mean_abs_a2 | 0.4737 ± 0.0494 | 0 ± 0 | 0.3042 ± 0.00147 | 0.4163 ± 7.71e-17 |
| mean_intensity | 0.4777 ± 0.0479 | 0.1 ± 2.49e-17 | 0.3123 ± 0.0013 | 0.4163 ± 7.71e-17 |
| train_steps | 500 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.1788 ± 0.0298 | nan ± nan | nan ± nan | nan ± nan |
| train_loss_last10 | 0.1816 ± 0.0294 | nan ± nan | nan ± nan | nan ± nan |
| wall_clock_s | 62.53 ± 16.2 | 1.409 ± 0.0629 | 6.976 ± 3.03 | 26.23 ± 6.61 |

Paired differences jepa-random [random action (matched scale)] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 60.07 | [-19.47, 142.9] | 0.314 | 0.314 | 123.4 | 157.7 |
| survival_rate | -0.249 | [-0.3207, -0.1704] | 5.14e-06 | 2.06e-05 | 0.1135 | 0.145 |
| mean_energy | -11.16 | [-13.85, -8.212] | 2.55e-07 | 1.28e-06 | 4.26 | 5.444 |
| mean_distance_km | -757.9 | [-1180, -337.2] | 0.00237 | 0.0063 | 637 | 814 |
| food_gathered | 5.118e+04 | [4.228e+04, 5.902e+04] | 1.3e-08 | 7.82e-08 | 1.279e+04 | 1.634e+04 |
| settlements | 18.73 | [9.5, 27.9] | 0.0021 | 0.0063 | 13.96 | 17.84 |
| mean_action_norm | 0.01509 | [0.009883, 0.0207] | 4.42e-06 | nan | 0.008059 | 0.0103 |
| mean_abs_a0 | -0.0352 | [-0.04046, -0.02892] | 2.61e-08 | nan | 0.008674 | 0.01108 |
| mean_abs_a1 | -0.0403 | [-0.04463, -0.03402] | 1.86e-08 | nan | 0.008258 | 0.01055 |
| mean_abs_a2 | 0.1695 | [0.1512, 0.186] | 1.86e-09 | nan | 0.02604 | 0.03327 |
| mean_intensity | 0.1654 | [0.1477, 0.1815] | 1.86e-09 | nan | 0.02528 | 0.0323 |
| train_steps | 500 | [500, 500] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 55.56 | [49.22, 61.58] | 1.86e-09 | nan | 9.412 | 12.03 |

Paired differences jepa-const [no steering, constant intensity] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | -45.83 | [-128.5, 40.2] | 0.258 | 0.516 | 127.6 | 163.1 |
| survival_rate | -0.06266 | [-0.1314, 0.01237] | 0.0449 | 0.225 | 0.1079 | 0.1379 |
| mean_energy | 1.72 | [-1.101, 4.563] | 0.105 | 0.315 | 4.236 | 5.413 |
| mean_distance_km | -976 | [-1366, -583.9] | 9.9e-05 | 0.000594 | 585.3 | 747.9 |
| food_gathered | 3761 | [-3639, 1.097e+04] | 0.262 | 0.516 | 1.113e+04 | 1.423e+04 |
| settlements | -9.567 | [-18.77, -0.06667] | 0.0612 | 0.245 | 14.19 | 18.13 |
| mean_action_norm | 0.5705 | [0.5654, 0.5758] | 1.86e-09 | nan | 0.007795 | 0.009962 |
| mean_abs_a0 | 0.2656 | [0.2604, 0.2718] | 1.86e-09 | nan | 0.008554 | 0.01093 |
| mean_abs_a1 | 0.2609 | [0.2566, 0.2673] | 1.86e-09 | nan | 0.008358 | 0.01068 |
| mean_abs_a2 | 0.05742 | [0.03922, 0.07415] | 7.99e-06 | nan | 0.02613 | 0.03339 |
| mean_intensity | 0.06148 | [0.04384, 0.07773] | 4.42e-06 | nan | 0.02533 | 0.03237 |
| train_steps | 500 | [500, 500] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 36.3 | [30.39, 41.99] | 3.73e-09 | nan | 8.834 | 11.29 |

Paired differences jepa-none [no action vector] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 323.7 | [256.4, 395.8] | 1.73e-06 | 3.46e-06 | 106.3 | 135.8 |
| survival_rate | 0.2108 | [0.1562, 0.2706] | 1.86e-09 | 9.31e-09 | 0.08619 | 0.1101 |
| mean_energy | nan | [nan, nan] | nan | nan | nan | nan |
| mean_distance_km | 5634 | [5361, 5904] | 1.86e-09 | 9.31e-09 | 405.5 | 518.1 |
| food_gathered | 8.975e+04 | [8.387e+04, 9.487e+04] | 1.86e-09 | 9.31e-09 | 8286 | 1.059e+04 |
| settlements | 38.1 | [30.87, 45.87] | 1.73e-06 | 3.46e-06 | 11.42 | 14.6 |
| mean_action_norm | 0.9867 | [0.9817, 0.9921] | 1.86e-09 | nan | 0.007795 | 0.009962 |
| mean_abs_a0 | 0.2656 | [0.2604, 0.2718] | 1.86e-09 | nan | 0.008554 | 0.01093 |
| mean_abs_a1 | 0.2609 | [0.2566, 0.2673] | 1.86e-09 | nan | 0.008358 | 0.01068 |
| mean_abs_a2 | 0.4737 | [0.4555, 0.4904] | 1.86e-09 | nan | 0.02613 | 0.03339 |
| mean_intensity | 0.3777 | [0.3601, 0.394] | 1.86e-09 | nan | 0.02533 | 0.03237 |
| train_steps | 500 | [500, 500] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 61.13 | [55.26, 66.67] | 1.86e-09 | nan | 8.581 | 10.97 |
