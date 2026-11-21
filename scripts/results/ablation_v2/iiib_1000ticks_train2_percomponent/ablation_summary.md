Config: 25 agents, 1000 ticks, train interval 2, seeds 0..29, random_scale [0.5311, 0.5218, 0.9449, 0.0, 0.0, 0.0, 0.0, 0.0], const_intensity 0.4777

Note: POST HOC, added after seeing (i) and (iii). Config (iii-b): same seeds (0-29) and settings as (iii) (25 agents, 1000 ticks, train interval 2). jepa rows and loss steps reused unchanged from scripts/results/ablation_v2/iii_1000ticks_train2 (not rerun). Controls matched per component to those 30 jepa evaluation runs: const intensity = jepa mean intensity; random a0/a1 ~ U(-2m,2m) matching jepa mean |a0|, |a1| (the movement-component norm was not recorded in (iii), so mean absolute value per component is matched instead), a2 scale chosen so E clip(|a2|,0.1,1) = jepa mean intensity, a3..a7 = 0. jepa wall_clock_s comes from the (iii) run (different CPU contention).

Pilot: {"source": "jepa evaluation rows (reused)", "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "jepa_mean_abs_a0": 0.2655700808666431, "jepa_mean_abs_a1": 0.26091340513718875, "jepa_mean_intensity": 0.4777499596823714, "random_scale": [0.5311401617332862, 0.5218268102743775, 0.944916979041088, 0.0, 0.0, 0.0, 0.0, 0.0], "const_intensity": 0.4777499596823714}

| metric | jepa (JEPA planner) mean ± sd | random (random action (matched scale)) mean ± sd | const (no steering, constant intensity) mean ± sd |
|---|---|---|---|
| agents_alive | 323.7 ± 201 | 155.1 ± 83.1 | 168.6 ± 62.1 |
| survival_rate | 0.2108 ± 0.163 | 0.09574 ± 0.0481 | 0.09927 ± 0.0366 |
| mean_energy | 40.6 ± 6.14 | 41.63 ± 5.09 | 41.87 ± 5.24 |
| mean_distance_km | 8773 ± 766 | 8866 ± 360 | 8835 ± 418 |
| food_gathered | 9.207e+04 ± 1.57e+04 | 9.707e+04 ± 8.06e+03 | 9.938e+04 ± 5.99e+03 |
| settlements | 38.1 ± 21.6 | 20.53 ± 10.1 | 22.77 ± 8.79 |
| mean_action_norm | 0.9867 ± 0.0147 | 0.6607 ± 0.000952 | 0.4777 ± 1.36e-16 |
| mean_abs_a0 | 0.2656 ± 0.0162 | 0.2653 ± 0.000568 | 0 ± 0 |
| mean_abs_a1 | 0.2609 ± 0.0158 | 0.2609 ± 0.00079 | 0 ± 0 |
| mean_abs_a2 | 0.4737 ± 0.0494 | 0.4755 ± 0.00107 | 0.4777 ± 1.36e-16 |
| mean_intensity | 0.4777 ± 0.0479 | 0.4807 ± 0.00104 | 0.4777 ± 1.36e-16 |
| train_steps | 500 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.1788 ± 0.0298 | nan ± nan | nan ± nan |
| train_loss_last10 | 0.1816 ± 0.0294 | nan ± nan | nan ± nan |
| wall_clock_s | 62.53 ± 16.2 | 26.38 ± 4.56 | 28.3 ± 3.5 |

Paired differences jepa-random [random action (matched scale)] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 168.6 | [96.37, 244.2] | 0.000189 | 0.00113 | 112.3 | 143.6 |
| survival_rate | 0.115 | [0.05957, 0.1764] | 0.000667 | 0.00267 | 0.08801 | 0.1125 |
| mean_energy | -1.029 | [-3.555, 1.653] | 0.393 | 0.698 | 3.926 | 5.017 |
| mean_distance_km | -93.29 | [-374.8, 193.1] | 0.349 | 0.698 | 423.3 | 541 |
| food_gathered | -5001 | [-1.147e+04, 776.3] | 0.0636 | 0.191 | 9147 | 1.169e+04 |
| settlements | 17.57 | [10.13, 25.47] | 0.000217 | 0.00113 | 11.61 | 14.84 |
| mean_action_norm | 0.326 | [0.321, 0.3313] | 1.86e-09 | nan | 0.00769 | 0.009827 |
| mean_abs_a0 | 0.0002689 | [-0.004925, 0.006466] | 0.529 | nan | 0.008533 | 0.0109 |
| mean_abs_a1 | 3.389e-06 | [-0.004312, 0.006384] | 0.0879 | nan | 0.008316 | 0.01063 |
| mean_abs_a2 | -0.001805 | [-0.01995, 0.0148] | 0.7 | nan | 0.02596 | 0.03318 |
| mean_intensity | -0.002919 | [-0.02049, 0.0132] | 0.792 | nan | 0.02518 | 0.03217 |
| train_steps | 500 | [500, 500] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 36.15 | [30.31, 41.36] | 9.31e-09 | nan | 8.383 | 10.71 |

Paired differences jepa-const [no steering, constant intensity] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 155.1 | [84.33, 229.5] | 0.000528 | 0.00264 | 109.9 | 140.4 |
| survival_rate | 0.1115 | [0.05565, 0.173] | 0.000418 | 0.00251 | 0.08854 | 0.1132 |
| mean_energy | -1.271 | [-3.854, 1.315] | 0.452 | 0.904 | 3.843 | 4.91 |
| mean_distance_km | -62.69 | [-372.1, 247.3] | 0.67 | 0.904 | 466.9 | 596.7 |
| food_gathered | -7308 | [-1.395e+04, -1296] | 0.135 | 0.404 | 9446 | 1.207e+04 |
| settlements | 15.33 | [7.633, 23.53] | 0.00171 | 0.00685 | 12.01 | 15.34 |
| mean_action_norm | 0.509 | [0.5039, 0.5143] | 1.86e-09 | nan | 0.007795 | 0.009962 |
| mean_abs_a0 | 0.2656 | [0.2604, 0.2718] | 1.86e-09 | nan | 0.008554 | 0.01093 |
| mean_abs_a1 | 0.2609 | [0.2566, 0.2673] | 1.86e-09 | nan | 0.008358 | 0.01068 |
| mean_abs_a2 | -0.00406 | [-0.02226, 0.01267] | 0.824 | nan | 0.02613 | 0.03339 |
| mean_intensity | 1.166e-16 | [-0.01765, 0.01625] | 0.626 | nan | 0.02533 | 0.03237 |
| train_steps | 500 | [500, 500] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 34.23 | [28.46, 39.63] | 5.59e-09 | nan | 8.443 | 10.79 |
