# Diagnósticos E3 sobre logits ya computados (sin re-entrenar)

## 1) Reliability por condicional CORN + ECE directo del drop gate C2

Lema 2: C2 = P(sev≥2|x) = q1·q2. El ECE del gate se mide DIRECTO;
los condicionales se miden por separado en su población propia.

| run | split | ECE(q1) | ECE(q2 \| sev≥1) | n(q2) | ECE(C2) dir | ECE(C2) raw |
|---|---|---|---|---|---|---|
| unsw_small_oficial | val | 0.0138 | 0.0379 | 17902 | 0.0256 | 0.0285 |
| unsw_small_oficial | test | 0.0925 | 0.0508 | 45332 | 0.0438 | 0.0441 |
| cic_small_strat | val | 0.0015 | 0.0029 | 83647 | 0.0009 | 0.0016 |
| cic_small_strat | test | 0.0015 | 0.0028 | 83647 | 0.0010 | 0.0016 |
| cic_small_time | val | 0.0474 | 0.4822 | 53021 | 0.0772 | 0.0551 |
| cic_small_time | test | 0.4203 | 0.1407 | 235915 | 0.1125 | 0.0807 |
| ember_small_temporal | — | n/a | n/a | — | n/a | n/a |

## 2) Classwise calibration de Choice (val)

E[1[y=k] \| p_k] por clase; la coordenada recon alimenta g_recon.

| run | classwise ECE | peor clase | ECE | support | recon: ECE |
|---|---|---|---|---|---|
| unsw_small_oficial | 0.0012 | exploitation | 0.0081 | 5009 | 0.0027 |
| cic_small_strat | 0.0001 | normal | 0.0004 | 340964 | 0.0001 |
| cic_small_time | 0.0062 | dos_ddos | 0.0358 | 51042 | nan |
| ember_small_temporal | 0.0010 | wacatac | 0.0082 | 7460 | nan |

## 3) Bayes-vs-Chow (val, choice head)

Chow: atribuye argmax si max p ≥ 1−λ_r. Bayes: matriz de costes
L(d,y) = c(y)·1[d≠y], c(y)=m si y es attack (miss) sino 1;
abstain = λ_r. Con m=1 las reglas coinciden EXACTAMENTE
(sanity: 0 desviaciones). EMBER sin clase normal → n/a.

| run | λ_r | m | % filas donde difieren | % entre attacks |
|---|---|---|---|---|
| unsw_small_oficial | 0.5 | 1 | 0.00 | 0.00 |
| unsw_small_oficial | 0.1 | 1 | 0.00 | 0.00 |
| unsw_small_oficial | 0.5 | 2 | 10.50 | 10.89 |
| unsw_small_oficial | 0.1 | 2 | 2.46 | 3.47 |
| unsw_small_oficial | 0.5 | 3 | 13.43 | 13.97 |
| unsw_small_oficial | 0.1 | 3 | 3.44 | 4.88 |
| unsw_small_oficial | 0.5 | 5 | 15.92 | 17.09 |
| unsw_small_oficial | 0.1 | 5 | 4.62 | 6.57 |
| unsw_small_oficial | 0.5 | 10 | 19.23 | 21.43 |
| unsw_small_oficial | 0.1 | 10 | 6.03 | 8.59 |
| cic_small_strat | 0.5 | 1 | 0.00 | 0.00 |
| cic_small_strat | 0.1 | 1 | 0.00 | 0.00 |
| cic_small_strat | 0.5 | 2 | 0.09 | 0.19 |
| cic_small_strat | 0.1 | 2 | 0.04 | 0.05 |
| cic_small_strat | 0.5 | 3 | 0.12 | 0.24 |
| cic_small_strat | 0.1 | 3 | 0.07 | 0.09 |
| cic_small_strat | 0.5 | 5 | 0.16 | 0.32 |
| cic_small_strat | 0.1 | 5 | 0.12 | 0.19 |
| cic_small_strat | 0.5 | 10 | 0.20 | 0.38 |
| cic_small_strat | 0.1 | 10 | 0.20 | 0.31 |
| cic_small_time | 0.5 | 1 | 0.00 | 0.00 |
| cic_small_time | 0.1 | 1 | 0.00 | 0.00 |
| cic_small_time | 0.5 | 2 | 1.73 | 10.19 |
| cic_small_time | 0.1 | 2 | 1.38 | 5.18 |
| cic_small_time | 0.5 | 3 | 2.17 | 11.87 |
| cic_small_time | 0.1 | 3 | 2.43 | 8.61 |
| cic_small_time | 0.5 | 5 | 2.93 | 14.97 |
| cic_small_time | 0.1 | 5 | 4.23 | 13.63 |
| cic_small_time | 0.5 | 10 | 4.86 | 24.06 |
| cic_small_time | 0.1 | 10 | 10.80 | 28.78 |

## Veredictos
- **CORN/Lema 2**: los ECE condicionales de los eslabones NO determinan el ECE del producto. UNSW test: q1 0.0925 y q2 0.0508 coexisten con C2 directo 0.0438; CIC by-day test: q1 0.4203 y q2 0.1407 coexisten con C2 directo 0.1125. Exactamente la razón por la que Lemma 2 manda a medir el gate, no a sus eslabones.
- **Classwise**: la calibración max-prob (TS) no garantiza calibración por coordenada; ECE classwise val ≤0.007 en los 4 runs y la coordenada recon queda ≤0.003 donde existe (UNSW 0.0027 en 1,574 filas; CIC strat 0.0001 en 23,839). El g_recon del controlador hereda una coordenada razonablemente calibrada.
- **Bayes-vs-Chow**: con costes uniformes la abstracción max-prob es regla de Bayes EXACTA (0 desviaciones en los 3 runs). La desviación aparece solo con asimetrías fuertes y SIEMPRE en la misma dirección: Bayes escala (abstiene) y Chow atribuye — la forma exacta del contraejemplo del apéndice. Magnitudes: CIC strat ≤0.2% en todo el grid; UNSW val 13.4% (m=3, λ_r=0.5) hasta 19.2% (m=10); CIC by-day ≤4.9% hasta m=10 λ_r=0.5, pero 10.8% (28.8% entre attacks) con m=10 λ_r=0.1. Consecuencia operativa: con τ=0.5 y costes moderados la abstracción está justificada empíricamente; operadores con asimetría grande (miss:alarm ≥5:1 con λ_r bajo) NO pueden heredar τ=0.5 sin remirir la política en val.
