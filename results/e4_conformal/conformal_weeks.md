# Conformal sobre Noul, EMBER (advisory §2(d))

α=0.1; calibración split-conformal en w46–51 (n=120,000, seed 42); Q̂=0.0855; **coverage en calibración 0.9000** (referencia de intercambiabilidad nombrada).

| sem | cov unp. (IC95) | cov unp. mal | cov unp. ben | d_TV ≥ 1−2BER | BER held-out | cov wt | cov wt mal | cov wt ben |
|---|---|---|---|---|---|---|---|---|
| 52 | 0.9300 [0.9280,0.9321] | 0.9207 | 0.9393 | 0.2048 | 0.3976 | 0.9309 | 0.9212 | 0.9407 |
| 53 | 0.9316 [0.9296,0.9336] | 0.9254 | 0.9379 | 0.2393 | 0.3803 | 0.9230 | 0.9189 | 0.9271 |
| 54 | 0.9298 [0.9277,0.9318] | 0.9229 | 0.9366 | 0.1517 | 0.4241 | 0.9079 | 0.9049 | 0.9109 |
| 55 | 0.9272 [0.9251,0.9293] | 0.9304 | 0.9241 | 0.1974 | 0.4013 | 0.8995 | 0.9104 | 0.8885 |
| 56 | 0.9353 [0.9333,0.9373] | 0.9333 | 0.9373 | 0.1721 | 0.4139 | 0.9277 | 0.9271 | 0.9282 |
| 57 | 0.9170 [0.9147,0.9192] | 0.9054 | 0.9285 | 0.1816 | 0.4092 | 0.9100 | 0.8983 | 0.9217 |
| 58 | 0.9265 [0.9244,0.9286] | 0.9238 | 0.9292 | 0.2126 | 0.3937 | 0.9240 | 0.9216 | 0.9265 |
| 59 | 0.9167 [0.9145,0.9189] | 0.9059 | 0.9276 | 0.2055 | 0.3973 | 0.9384 | 0.9269 | 0.9499 |
| 60 | 0.9086 [0.9063,0.9109] | 0.8835 | 0.9337 | 0.2294 | 0.3853 | 0.9272 | 0.9049 | 0.9495 |
| 61 | 0.9083 [0.9059,0.9106] | 0.8954 | 0.9211 | 0.1864 | 0.4068 | 0.9274 | 0.9156 | 0.9392 |
| 62 | 0.9098 [0.9074,0.9120] | 0.9043 | 0.9152 | 0.1919 | 0.4041 | 0.9115 | 0.9057 | 0.9172 |
| 63 | 0.8947 [0.8923,0.8972] | 0.8841 | 0.9053 | 0.1770 | 0.4115 | 0.8978 | 0.8865 | 0.9091 |

Lectura (scoped del advisory): coverage EMPÍRICO con IC Clopper–Pearson por semana — NO una garantía distribution-free. La cota de drift certifica magnitud (d_TV ≥ 1−2·BER, BER medido held-out); el anclaje citable es Barber et al. 2023 (gap ≈ 2·d_TV con pesos conocidos) y la variante weighted es mitigación empírica, nunca garantía exacta. Challenge set: fuera de esta tabla (estrés sin benignos, se reporta separado).

## 4) Challenge set (estrés; fuera de toda promesa)

| fila | n | % unknown | coverage positivos | set vacío | mediana p | % p ≥ 0.5 |
|---|---|---|---|---|---|---|
| challenge checkpoints | 6315 | 98.7% | **0.267** | 49.4% | 0.655 | 58.2% |

El coverage de positivos colapsa de ~0.92 (semanas test) a 0.267 — como
declara el advisory Q2: sin intercambiabilidad (evasivo, 98.8% unknown)
NINGÚN wrapper conformal promete algo; se reporta como estrés. Semántica:
el detector sigue votando "ataque" (58% de las filas con p ≥ 0.5, mediana
0.65) pero sin la confianza del conjunto conformal — honestidad a dos
niveles, no una cobertura que no tiene.
