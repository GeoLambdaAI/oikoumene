Config: 25 agents, 3000 ticks, train interval 20, seeds 0..9, random_scale 0.5915, const_intensity 0.2898

Note: Plan declared before any stage-2 data: modes jepa/random/const/none; random scale and const intensity matched to a 5-seed jepa pilot (seeds 1000-1004) per configuration; configs (i) 25 agents 1000 ticks 30 seeds, (iii) = (i) with train interval 2, (ii) 25 agents 3000 ticks 10 seeds without none. The plan was revised from the earlier 5000-tick/30-seed (ii) before any stage-2 data existed. Config (ii).

Pilot: {"seeds": [1000, 1001, 1002, 1003, 1004], "jepa_mean_action_norm": 0.9529883366920794, "jepa_mean_intensity": 0.2898149995670011, "expected_uniform_norm": 1.6110198104540456, "jepa_mean_abs_a2": 0.27872386494559087, "random_scale": 0.5915435244855813, "const_intensity": 0.2898149995670011}

| metric | jepa (JEPA planner) mean ± sd | random (random action (matched scale)) mean ± sd | const (no steering, constant intensity) mean ± sd |
|---|---|---|---|
| agents_alive | 215.3 ± 135 | 346.1 ± 101 | 44.4 ± 34.7 |
| survival_rate | 0.1383 ± 0.0309 | 0.1661 ± 0.0241 | 0.07518 ± 0.0533 |
| mean_energy | 32.89 ± 2.7 | 32.32 ± 4.11 | 32.32 ± 3.83 |
| mean_distance_km | 7161 ± 1.26e+03 | 6913 ± 409 | 6395 ± 477 |
| food_gathered | 1.232e+05 ± 6.36e+04 | 1.735e+05 ± 4.57e+04 | 3.979e+04 ± 1.91e+04 |
| settlements | 53.2 ± 30.9 | 87.3 ± 22.5 | 13.1 ± 11.3 |
| mean_action_norm | 0.9617 ± 0.0485 | 0.9532 ± 0.000313 | 0.2898 ± 4.53e-17 |
| mean_abs_a0 | 0.2915 ± 0.028 | 0.2956 ± 0.000256 | 0 ± 0 |
| mean_abs_a1 | 0.2747 ± 0.0146 | 0.2957 ± 0.00027 | 0 ± 0 |
| mean_abs_a2 | 0.2814 ± 0.0258 | 0.2961 ± 0.000196 | 0.2898 ± 4.53e-17 |
| mean_intensity | 0.2923 ± 0.0246 | 0.3045 ± 0.000194 | 0.2898 ± 4.53e-17 |
| train_steps | 150 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.6158 ± 0.0137 | nan ± nan | nan ± nan |
| train_loss_last10 | 0.6264 ± 0.0124 | nan ± nan | nan ± nan |
| wall_clock_s | 408.2 ± 279 | 396.6 ± 167 | 42.07 ± 37.9 |

Paired differences jepa-random [random action (matched scale)] (n seeds = 10; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | -130.8 | [-225.9, -30] | 0.0488 | 0.195 | 168.6 | 232.1 |
| survival_rate | -0.02782 | [-0.05162, -0.008177] | 0.0273 | 0.152 | 0.03709 | 0.05108 |
| mean_energy | 0.5709 | [-2.733, 3.657] | 0.695 | 1 | 5.401 | 7.437 |
| mean_distance_km | 248.2 | [-675.4, 1173] | 0.695 | 1 | 1569 | 2161 |
| food_gathered | -5.028e+04 | [-9.475e+04, -3506] | 0.0488 | 0.195 | 7.938e+04 | 1.093e+05 |
| settlements | -34.1 | [-56, -11.9] | 0.0254 | 0.152 | 37.98 | 52.3 |
| mean_action_norm | 0.008514 | [-0.0191, 0.03704] | 0.695 | nan | 0.04822 | 0.06639 |
| mean_abs_a0 | -0.004137 | [-0.02015, 0.01187] | 0.625 | nan | 0.0278 | 0.03828 |
| mean_abs_a1 | -0.021 | [-0.02852, -0.01167] | 0.00586 | nan | 0.01444 | 0.01989 |
| mean_abs_a2 | -0.01472 | [-0.02853, 0.001597] | 0.16 | nan | 0.0257 | 0.03538 |
| mean_intensity | -0.01226 | [-0.0254, 0.003307] | 0.232 | nan | 0.0245 | 0.03373 |
| train_steps | 150 | [150, 150] | 0.00195 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 11.6 | [-167.1, 188.1] | 0.922 | nan | 307.1 | 422.8 |

Paired differences jepa-const [no steering, constant intensity] (n seeds = 10; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 170.9 | [87.8, 254.9] | 0.00391 | 0.0234 | 145.4 | 200.2 |
| survival_rate | 0.06311 | [0.02429, 0.1069] | 0.00977 | 0.0391 | 0.07226 | 0.0995 |
| mean_energy | 0.1149 | [-3.434, 3.647] | 0.938 | 0.938 | 6.583 | 9.665 |
| mean_distance_km | 766.3 | [63.92, 1519] | 0.16 | 0.32 | 1252 | 1724 |
| food_gathered | 8.339e+04 | [4.419e+04, 1.221e+05] | 0.00977 | 0.0391 | 6.802e+04 | 9.366e+04 |
| settlements | 40.1 | [20.7, 59.8] | 0.00391 | 0.0234 | 34.1 | 46.96 |
| mean_action_norm | 0.6719 | [0.6441, 0.7004] | 0.00195 | nan | 0.04833 | 0.06655 |
| mean_abs_a0 | 0.2915 | [0.2754, 0.3075] | 0.00195 | nan | 0.02787 | 0.03837 |
| mean_abs_a1 | 0.2747 | [0.2672, 0.2841] | 0.00195 | nan | 0.01452 | 0.01999 |
| mean_abs_a2 | -0.008443 | [-0.02229, 0.007885] | 0.275 | nan | 0.02574 | 0.03545 |
| mean_intensity | 0.002458 | [-0.01073, 0.01804] | 0.846 | nan | 0.02453 | 0.03377 |
| train_steps | 150 | [150, 150] | 0.00195 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 366.1 | [206.1, 531.8] | 0.00195 | nan | 281.2 | 387.2 |
