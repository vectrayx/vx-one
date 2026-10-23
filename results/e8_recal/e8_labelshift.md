# E8-LS — label shift explícito (PortScan day2)

q1=0.5573  q2_true=0.5548
| mapa base | estimador | q̂2 | ECE15 | Brier | AUROC |
|---|---|---|---|---|---|
| temp(day1) sin corrección | none | — | 0.5057 | 0.5037 | 0.56 |
| temp(day1) prior-corregido[bbse_hard] (q̂=0.000) |  | — | 0.5548 | 0.5548 | 0.56 |
| temp(day1) prior-corregido[bbse_soft] (q̂=0.000) |  | — | 0.5548 | 0.5548 | 0.56 |
| temp(day1) prior-corregido[em] (q̂=0.000) |  | — | 0.5548 | 0.5548 | 0.56 |
| raw (sin calibrar) |  | — | 0.5538 | 0.5537 | 0.56 |
| platt(day1) sin corrección | none | — | 0.5223 | 0.5239 | 0.56 |
| platt(day1) prior-corregido[bbse_hard] (q̂=0.004) |  | — | 0.5529 | 0.5532 | 0.56 |
| platt(day1) prior-corregido[bbse_soft] (q̂=0.006) |  | — | 0.5525 | 0.553 | 0.56 |
| platt(day1) prior-corregido[em] (q̂=0.015) |  | — | 0.5508 | 0.5521 | 0.56 |
| raw (sin calibrar) |  | — | 0.5538 | 0.5537 | 0.56 |
| iso(day1) sin corrección | none | — | 0.5543 | 0.5552 | 0.5065 |
| iso(day1) prior-corregido[bbse_hard] (q̂=0.041) |  | — | 0.5451 | 0.5484 | 0.5065 |
| iso(day1) prior-corregido[bbse_soft] (q̂=0.004) |  | — | 0.5527 | 0.553 | 0.5065 |
| iso(day1) prior-corregido[em] (q̂=0.001) |  | — | 0.5543 | 0.5543 | 0.5065 |
| raw (sin calibrar) |  | — | 0.5538 | 0.5537 | 0.56 |
