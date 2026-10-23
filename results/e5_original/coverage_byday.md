# E5-orig — Fase 1 con payload real (pcaps originales)

Cobertura por 5-tupla del día (protocolo anti desfase-ts; ver docstring).

| día | alertas(día/total) | attack resolved | benign flagged | P | R |
|---|---|---|---|---|---|
| 2017-07-03 | 9,425/9,425 | 0.0 | 0.0478 | 0.0 | 0.0 |
| 2017-07-04 | 2,644/2,644 | 0.0205 | 0.02379 | 0.0269 | 0.0205 |
| 2017-07-05 | 2,799/2,799 | 0.0219 | 0.02364 | 0.3463 | 0.0219 |
| 2017-07-06 | 26,652/26,652 | 0.0208 | 0.02225 | 0.0045 | 0.0208 |
| 2017-07-07 | 2,745/12,170 | 0.0023 | 0.02151 | 0.0702 | 0.0023 |

## Por familia (resolved share)
### 2017-07-03
- normal: n=529,450 resolved=0.0478
### 2017-07-04
- bruteforce: n=13,832 resolved=0.0205
- normal: n=431,809 resolved=0.0238
### 2017-07-05
- dos_ddos: n=251,711 resolved=0.0218
- exploitation: n=11 resolved=1.0
- normal: n=439,668 resolved=0.0236
### 2017-07-06
- infiltration: n=36 resolved=0.9722
- normal: n=456,267 resolved=0.0222
- webattack: n=2,180 resolved=0.005
### 2017-07-07
- botnet: n=1,956 resolved=0.0
- dos_ddos: n=128,025 resolved=0.0045
- normal: n=413,928 resolved=0.0215
- recon: n=158,804 resolved=0.0006
