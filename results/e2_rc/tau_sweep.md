# E2-ext: sweep τ_abs (de-tilt + T_choice) — logits LOAO

## UNSW (media sobre 9 folds)

| τ | abstención % | falsa-atrib. ataque % | absorbido→normal % |
|---|---|---|---|
| 0.05 | 0.00 | 91.16 | 8.84 |
| 0.10 | 0.00 | 91.16 | 8.84 |
| 0.15 | 0.00 | 91.16 | 8.84 |
| 0.20 | 0.00 | 91.16 | 8.84 |
| 0.25 | 0.03 | 91.13 | 8.84 |
| 0.30 | 0.42 | 90.81 | 8.77 |
| 0.35 | 2.85 | 88.62 | 8.53 |
| 0.40 | 7.15 | 84.52 | 8.33 |
| 0.45 | 23.15 | 68.88 | 7.97 |
| 0.50 | 37.54 | 54.97 | 7.49 |
| 0.55 | 47.45 | 45.47 | 7.08 |
| 0.60 | 55.32 | 38.01 | 6.67 |
| 0.65 | 63.10 | 30.57 | 6.33 |
| 0.70 | 68.98 | 24.97 | 6.05 |
| 0.75 | 72.02 | 22.30 | 5.68 |
| 0.80 | 76.72 | 17.94 | 5.34 |
| 0.85 | 81.62 | 13.36 | 5.02 |
| 0.90 | 89.80 | 5.72 | 4.48 |
| 0.95 | 93.75 | 2.55 | 3.70 |
| 0.99 | 97.33 | 0.66 | 2.01 |

AURC (familias vistas): 0.0574

## CIC (media sobre 7 folds)

| τ | abstención % | falsa-atrib. ataque % | absorbido→normal % |
|---|---|---|---|
| 0.05 | 0.00 | 5.53 | 94.47 |
| 0.10 | 0.00 | 5.53 | 94.47 |
| 0.15 | 0.00 | 5.53 | 94.47 |
| 0.20 | 0.00 | 5.53 | 94.47 |
| 0.25 | 0.00 | 5.53 | 94.47 |
| 0.30 | 0.00 | 5.53 | 94.47 |
| 0.35 | 0.00 | 5.53 | 94.47 |
| 0.40 | 0.00 | 5.53 | 94.47 |
| 0.45 | 0.00 | 5.53 | 94.47 |
| 0.50 | 0.01 | 5.53 | 94.46 |
| 0.55 | 0.24 | 5.44 | 94.32 |
| 0.60 | 0.46 | 5.34 | 94.20 |
| 0.65 | 0.79 | 5.17 | 94.04 |
| 0.70 | 0.99 | 5.09 | 93.91 |
| 0.75 | 1.20 | 5.00 | 93.79 |
| 0.80 | 1.76 | 4.99 | 93.24 |
| 0.85 | 7.63 | 4.99 | 87.38 |
| 0.90 | 13.77 | 4.69 | 81.54 |
| 0.95 | 15.06 | 4.16 | 80.78 |
| 0.99 | 17.79 | 2.83 | 79.39 |

AURC (familias vistas): 0.0012

## Estadístico de confianza y margen en la familia excluida

| fold | excl | conf p50 | conf p90 | margin p50 | margin p90 |
|---|---|---|---|---|---|
| cic_small_strat_loao_botnet | botnet | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| cic_small_strat_loao_bruteforce | bruteforce | 0.9996 | 1.0000 | 0.9992 | 0.9999 |
| cic_small_strat_loao_dos_ddos | dos_ddos | 0.9989 | 0.9998 | 0.9978 | 0.9997 |
| cic_small_strat_loao_exploitation | exploitation | 0.9994 | 0.9994 | 0.9989 | 0.9989 |
| cic_small_strat_loao_infiltration | infiltration | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| cic_small_strat_loao_recon | recon | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| cic_small_strat_loao_webattack | webattack | 0.8523 | 0.9479 | 0.7703 | 0.9127 |
| unsw_small_oficial_loao_analysis | analysis | 0.4402 | 0.4768 | 0.0631 | 0.1091 |
| unsw_small_oficial_loao_backdoor | backdoor | 0.4800 | 0.5501 | 0.1592 | 0.2688 |
| unsw_small_oficial_loao_dos | dos | 0.6416 | 0.9552 | 0.5307 | 0.9290 |
| unsw_small_oficial_loao_exploitation | exploitation | 0.7521 | 0.9541 | 0.5949 | 0.9298 |
| unsw_small_oficial_loao_fuzzing | fuzzing | 0.7102 | 0.9927 | 0.5220 | 0.9891 |
| unsw_small_oficial_loao_generic | generic | 0.5001 | 0.6984 | 0.2602 | 0.5789 |
| unsw_small_oficial_loao_recon | recon | 0.6580 | 0.9042 | 0.4660 | 0.8503 |
| unsw_small_oficial_loao_shellcode | shellcode | 0.5394 | 0.8601 | 0.2887 | 0.7703 |
| unsw_small_oficial_loao_worms | worms | 0.7765 | 0.8808 | 0.6640 | 0.8339 |
