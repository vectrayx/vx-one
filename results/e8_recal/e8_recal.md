| model | window | cal-src | AUROC | ECE15 | Brier | FPR@95(cal) | T_noul | ECE15-ch | macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| static | Friday-WorkingHours-Afte | val | 0.9931 | 0.1273 | 0.1254 | 0.0 | 2.914 | 0.1071 | 0.2971 |
| oracle | Friday-WorkingHours-Afte | oracle(day-self) | 0.9931 | 0.0287 | 0.0267 | 0.0 | a=0.75,b=6.71 | 0.1048 | 0.2971 |
| static | Friday-WorkingHours-Afte | val | 0.56 | 0.5531 | 0.539 | 0.0 | 2.914 | 0.5426 | 0.1024 |
| walk | Friday-WorkingHours-Afte | day:Friday-WorkingHours-Afternoon-DDos.pcap_ISCX | 0.56 | 0.5531 | 0.5037 | 0.0 | 4.255 | 0.528 | 0.1024 |
| walkplatt | Friday-WorkingHours-Afte | day:Friday-WorkingHours-Afternoon-DDos.pcap_ISCX [platt] | 0.56 | 0.5255 | 0.508 | 0.0 | a=0.75,b=6.71 | 0.528 | 0.1024 |
| walkacc | Friday-WorkingHours-Afte | val+day | 0.56 | 0.5531 | 0.5315 | 0.0 | 3.262 | 0.5385 | 0.1024 |
| oracle | Friday-WorkingHours-Afte | oracle(day-self) | 0.56 | 0.4452 | 0.2462 | 0.1481 | a=0.06,b=0.98 | 0.0564 | 0.1024 |
