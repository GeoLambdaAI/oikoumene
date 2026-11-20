Config: 25 agents, 1000 ticks, train interval 20, seeds 0..29, random_scale 0.6135, const_intensity 0.3271

Note: Plan declared before any stage-2 data: modes jepa/random/const/none; random scale and const intensity matched to a 5-seed jepa pilot (seeds 1000-1004) per configuration; configs (i) 25 agents 1000 ticks 30 seeds, (iii) = (i) with train interval 2, (ii) 25 agents 3000 ticks 10 seeds without none. The plan was revised from the earlier 5000-tick/30-seed (ii) before any stage-2 data existed. A first run of (i) completed its simulations but crashed in the summary (MDE solver NaN) before writing any output; it was rerun unchanged after fixing the solver. Config (i).

Pilot: {"seeds": [1000, 1001, 1002, 1003, 1004], "jepa_mean_action_norm": 0.9883063690889593, "jepa_mean_intensity": 0.32705982870568306, "expected_uniform_norm": 1.6110198104540456, "jepa_mean_abs_a2": 0.3178598292791153, "random_scale": 0.6134663041855566, "const_intensity": 0.32705982870568306}

| metric | jepa (JEPA planner) mean ± sd | none (no action vector) mean ± sd | random (random action (matched scale)) mean ± sd | const (no steering, constant intensity) mean ± sd |
|---|---|---|---|---|
| agents_alive | 178.5 ± 158 | 0 ± 0 | 285.6 ± 190 | 259.4 ± 151 |
| survival_rate | 0.305 ± 0.182 | 0 ± 0 | 0.4436 ± 0.184 | 0.4576 ± 0.164 |
| mean_energy | 49.16 ± 7.24 | nan ± nan | 50 ± 5.91 | 51.78 ± 5.2 |
| mean_distance_km | 9211 ± 1.62e+03 | 3139 ± 111 | 9199 ± 1.26e+03 | 9457 ± 881 |
| food_gathered | 3.803e+04 ± 2.23e+04 | 2325 ± 190 | 4.163e+04 ± 1.58e+04 | 4.08e+04 ± 1.26e+04 |
| settlements | 15.57 ± 16 | 0 ± 0 | 22.8 ± 16.1 | 18 ± 11.8 |
| mean_action_norm | 0.9911 ± 0.0382 | 0 ± 0 | 0.9898 ± 0.00177 | 0.3271 ± 4.12e-17 |
| mean_abs_a0 | 0.2972 ± 0.0304 | 0 ± 0 | 0.3064 ± 0.00194 | 0 ± 0 |
| mean_abs_a1 | 0.286 ± 0.0239 | 0 ± 0 | 0.3071 ± 0.00169 | 0 ± 0 |
| mean_abs_a2 | 0.2926 ± 0.021 | 0 ± 0 | 0.3099 ± 0.00153 | 0.3271 ± 4.12e-17 |
| mean_intensity | 0.303 ± 0.02 | 0.1 ± 2.49e-17 | 0.3178 ± 0.00142 | 0.3271 ± 4.12e-17 |
| train_steps | 50 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| final_train_loss | 0.9042 ± 0.00537 | nan ± nan | nan ± nan | nan ± nan |
| train_loss_last10 | 0.9153 ± 0.00498 | nan ± nan | nan ± nan | nan ± nan |
| wall_clock_s | 15.4 ± 12.5 | 1.386 ± 0.074 | 7.475 ± 3.93 | 6.254 ± 2.72 |

Paired differences jepa-random [random action (matched scale)] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | -107 | [-193, -20.86] | 0.0277 | 0.139 | 127.3 | 162.7 |
| survival_rate | -0.1386 | [-0.2249, -0.0499] | 0.00619 | 0.0372 | 0.1307 | 0.167 |
| mean_energy | -0.7521 | [-3.555, 2.021] | 0.733 | 1 | 4.187 | 5.357 |
| mean_distance_km | 12.2 | [-629.1, 682.4] | 0.984 | 1 | 975.9 | 1247 |
| food_gathered | -3595 | [-1.309e+04, 6276] | 0.318 | 0.955 | 1.446e+04 | 1.848e+04 |
| settlements | -7.233 | [-14.77, 0.6333] | 0.0693 | 0.277 | 11.49 | 14.69 |
| mean_action_norm | 0.001379 | [-0.01185, 0.01521] | 0.903 | nan | 0.02056 | 0.02628 |
| mean_abs_a0 | -0.009133 | [-0.01958, 0.002361] | 0.0427 | nan | 0.01646 | 0.02103 |
| mean_abs_a1 | -0.02114 | [-0.02918, -0.01247] | 9.9e-05 | nan | 0.01279 | 0.01634 |
| mean_abs_a2 | -0.01727 | [-0.02471, -0.009838] | 0.000111 | nan | 0.01111 | 0.0142 |
| mean_intensity | -0.01487 | [-0.02193, -0.007762] | 0.000283 | nan | 0.01058 | 0.01352 |
| train_steps | 50 | [50, 50] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 7.923 | [3.547, 12.85] | 0.00322 | nan | 6.917 | 8.839 |

Paired differences jepa-const [no steering, constant intensity] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | -80.87 | [-155.8, -5.565] | 0.0879 | 0.352 | 112.2 | 143.4 |
| survival_rate | -0.1526 | [-0.237, -0.06739] | 0.00501 | 0.0301 | 0.1271 | 0.1624 |
| mean_energy | -2.519 | [-5.933, 1.041] | 0.0623 | 0.311 | 5.317 | 6.802 |
| mean_distance_km | -246.2 | [-967.4, 494.7] | 0.612 | 1 | 1095 | 1399 |
| food_gathered | -2772 | [-1.175e+04, 7028] | 0.428 | 1 | 1.399e+04 | 1.788e+04 |
| settlements | -2.433 | [-8.9, 4.301] | 0.471 | 1 | 9.876 | 12.62 |
| mean_action_norm | 0.6641 | [0.6511, 0.6777] | 1.86e-09 | nan | 0.02021 | 0.02583 |
| mean_abs_a0 | 0.2972 | [0.287, 0.3085] | 1.86e-09 | nan | 0.01609 | 0.02057 |
| mean_abs_a1 | 0.286 | [0.2781, 0.2946] | 1.86e-09 | nan | 0.01265 | 0.01616 |
| mean_abs_a2 | -0.03447 | [-0.04198, -0.02701] | 2.61e-08 | nan | 0.01112 | 0.01422 |
| mean_intensity | -0.0241 | [-0.03122, -0.01701] | 1.42e-06 | nan | 0.01058 | 0.01352 |
| train_steps | 50 | [50, 50] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 9.144 | [4.955, 14.02] | 0.000209 | nan | 6.786 | 8.673 |

Paired differences jepa-none [no action vector] (n seeds = 30; Holm across the 6 outcome metrics)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | Holm p | MDE80 (α=.05) | MDE80 (Holm) |
|---|---|---|---|---|---|---|
| agents_alive | 178.5 | [124.8, 236.4] | 2.56e-06 | 7.69e-06 | 83.52 | 106.7 |
| survival_rate | 0.305 | [0.2416, 0.3691] | 2.56e-06 | 7.69e-06 | 0.09611 | 0.1228 |
| mean_energy | nan | [nan, nan] | nan | nan | nan | nan |
| mean_distance_km | 6072 | [5498, 6647] | 1.86e-09 | 9.31e-09 | 853.5 | 1091 |
| food_gathered | 3.571e+04 | [2.817e+04, 4.401e+04] | 1.86e-09 | 9.31e-09 | 1.182e+04 | 1.511e+04 |
| settlements | 15.57 | [10.1, 21.53] | 1.22e-05 | 1.22e-05 | 8.457 | 10.81 |
| mean_action_norm | 0.9911 | [0.9782, 1.005] | 1.86e-09 | nan | 0.02021 | 0.02583 |
| mean_abs_a0 | 0.2972 | [0.287, 0.3085] | 1.86e-09 | nan | 0.01609 | 0.02057 |
| mean_abs_a1 | 0.286 | [0.2781, 0.2946] | 1.86e-09 | nan | 0.01265 | 0.01616 |
| mean_abs_a2 | 0.2926 | [0.2851, 0.3] | 1.86e-09 | nan | 0.01112 | 0.01422 |
| mean_intensity | 0.203 | [0.1958, 0.2101] | 1.86e-09 | nan | 0.01058 | 0.01352 |
| train_steps | 50 | [50, 50] | 4.32e-08 | nan | nan | nan |
| final_train_loss | nan | [nan, nan] | nan | nan | nan | nan |
| train_loss_last10 | nan | [nan, nan] | nan | nan | nan | nan |
| wall_clock_s | 14.01 | [9.938, 18.81] | 1.86e-09 | nan | 6.619 | 8.458 |
