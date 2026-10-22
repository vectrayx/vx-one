# E9 (per-window retrain) vs corrida estática — mismos evaluadores E8

| bloque | n | AUROC | ECE15 | Brier | FPR@95 | macroF1 | ECE-ch | sevMAE |
|---|---|---|---|---|---|---|---|---|
| day1_ddos static | 138145 | 0.9931 | 0.1273 | 0.1254 | 0.0 | 0.2971 | 0.1071 | 0.2543 |
| day2_portscan static | 286467 | 0.56 | 0.5531 | 0.539 | 0.0 | 0.1024 | 0.5426 | 0.5609 |
| day1_ddos[asrun_val] (e9) | 138145 | 1.0 | 0.0006 | 0.0009 | 0.0 | 0.333 | 0.0141 | 0.0542 |
| day2_portscan[asrun_val] (e9) | 286467 | 0.8 | 0.5539 | 0.5464 | 0.0 | 0.1024 | 0.5474 | 0.5578 |
| day1_ddos[calib_valday] (e9) | 138145 | 1.0 | 0.0005 | 0.0008 | 0.0 | 0.333 | 0.0116 | 0.0542 |
| day2_portscan[calib_valday] (e9) | 286467 | 0.8 | 0.5538 | 0.5489 | 0.0 | 0.1024 | 0.549 | 0.5578 |
