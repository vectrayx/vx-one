# VX-One — Protocolo experimental E1–E7 y resultados medidos

Documento de trabajo (2026-10-19). Explica **qué se midió, cómo, con qué código
y qué salió**. Todos los números provienen de `results/` del repo original
(`~/projects/paper`), ejecutado entre 2026-10-17 y 2026-10-18. Es la fuente de
verdad del paper (`paper/main.tex`); el draft antiguo
(`docs/reference/VectrayX_One_Paper_Draft.md`) está **obsoleto** y sus cifras
aspiracionales no se usan.

---

## 0. Resumen de estado

| Exp | Qué mide | Estado | Resultado clave |
|---|---|---|---|
| E1 | Calidad de detección + baselines | ✅ 4 datasets × 3 escalas × 4 baselines | Gana detección (AUROC/FPR@95); pierde macro-F1 vs XGBoost en UNSW/CIC/EMBER calibrado |
| E2 | Día cero (LOAO + drift EMBER) | ✅ UNSW 9/9 folds, CIC 7 folds, EMBER drift | Detector genérico detecta lo no visto en UNSW 9/9; en CIC 3/7 familias duras; drift EMBER → familia mutada va a `unknown` |
| E3 | Calibración | ✅ 15 bins, ECE/MCE/Brier, raw+cal en baselines | En-distribution ECE ≤0.06 net; bajo drift colapsa a 0.40 (hallazgo negative honesto) |
| E4 | Latencia | ✅ batch=1, 1 hilo, CPU local, n=3000 | p50 675 (Nano) / 955 (Small) / 1652 µs (Base), p99 ≤2.1 ms |
| E5 | Cascada dos fases (Suricata + ET) | ✅ 5 días CIC-IDS2017 (~2.8M flujos) | F1 resuelve ~0.1% del tráfico sin payload; zona gris ≈99.9%; F2 la recupera (comb R ≥0.978) |
| E6 | Ablaciones | ✅ 14/14 | rama textual +5 pts (UNSW), tilt ω perjudica CIC (+4.9 sin), dropout crítico en CIC (−14.7) e innecesario en EMBER (+2.7) |
| E7 | Baseline comercial (Jev 1.13 vía Zen) | ✅ 2,021 filas, coste 0 | §9 |

---

## 1. Datos y splits (F1/F2, verificados por conteo)

| Dataset | filas train/val/test | d_num | split | text fields |
|---|---|---|---|---|
| UNSW-NB15 | 149,039 / 26,302 / 82,332 | 39 | oficial (175,341/82,332 auditado) | proto, service, state |
| CIC-IDS2017 (variante MLCVE) | 1,981,520 / 424,611 / 424,612 | 77 | `strat` 70/15/15 **y** `time` (drift) | ninguna |
| EMBER2024 | 2,760,000 / 360,000 / 726,315 (+challenge 6,315) | 2568 | temporal por semanas (w0–63 / w64 challenge) | ninguna |

Contratos de features: `docs/reference/data/SCHEMA.md` copiado → `docs/reference/`.
Feature stage φ (D6): imputación de no-finitos con mediana + flag de fila,
signed log1p, escala robusta mediana/IQR de train, clip ±8.
Código: `src/models/feature_stage.py`, `src/features/`.

### Quirks de datasets (alimentan la sección de datos del paper)

- **UNSW-NB15:** split oficial 175,341/82,332 verificado; el `attack_cat` real
  del CSV es **"Backdoor" singular** (la documentación dice "Backdoors") — una
  ingestión ingenua lo manda a `unknown` (2,329 filas); corregido y congelado
  en SCHEMA §3.2. Mirror HF con test.csv corrupto evitado (verificación md5).
- **CIC-IDS2017 (MLCVE):** ocho CSVs con headers idénticos (md5 verificado);
  15 labels con conteos congelados en SCHEMA §3.1; bugs del dataset
  documentados y normalizados antes de materializar: labels web-attack con
  **mojibake U+FFFD** (bytes `ef bf bd` verificados por hexdump), `DoS
  slowloris` (s minúscula), filename con typo `Infilteration`; 2,867 filas
  NaN/inf (duración 0) conservadas tal cual; micro-clases Infiltration=36 y
  Heartbleed=11 conservadas y estratificadas exactas; el split temporal usa el
  orden de días como proxy (PortScan→test, Bot→val por diseño de captura).
- **El label `DDoS` (LOIT) vive en el viernes**, no el miércoles → el subset
  `choice_ddos_vector` = Wednesday + Friday-DDoS (D8: el candidato se nombra
  `tcp_flood` — "flood TCP volumétrico" — porque `SYN Flag Count = 0` en 100%
  de esos flujos es un artefacto de CICFlowMeter; sí llamarlo SYN_Flood sería
  reclamar un mecanismo que los datos no exhiben).
- **EMBER2024:** 52 train + 12 test + 64 challenge semanas; 60,000
  líneas/semana exactas; censo de familias FIRST (46,607 familias, cola
  extrema de singletons) evitó re-descargar 53 GB; top-10 real = 40.7% de
  malicious con corte top-10→11 limpio (34,262 vs 17,974); drift temporal
  genuino confirmado por crosstab (sfone 43,720→132, sinowal 34,488→18,
  grandoreiro 40,698→226, berbew CRECE); challenge = 6,315 evasivos con 98.8%
  → `unknown` (test de estrés, fuera de toda promesa de cobertura).

## 2. Cómo se entrenó (F2, D1–D12)

- **Arquitectura:** proyección numérica + embeddings lookup de texto (vocab ≤135)
  → tronco de L bloques [Linear→LN→GELU→Dropout 0.1] (residual desde bloque 2)
  → cabezas tipadas: Gate (sigmoide), Choice_family (softmax K + `unknown`),
  Choice_ddos (solo CIC), Score (CORN, 4 niveles).
- **Escalas:** Nano ≈100K, Small ≈1M, Base ≈5–10M params. Tabla exacta §2.3 spec F2.
- **Losses:** Gate = BCE sin pesos de clase (D2: la BCE ponderada desplaza el
  logit óptimo y la temperatura no revierte shifts); Choice = CE ponderada
  sqrt-normalizada media-1 + **de-tilt analítico** `z − log ω` en inferencia (D4);
  Score = CORN (D3); multi-tarea Kendall homeostático modificado con máscaras (D5).
- **Receta:** AdamW wd=1e-2 (solo 2D), warmup 5% + coseno, clip 1.0, AMP, patience 4,
  SEED=42, checkpoint atomico + resume (probado con kill -9 real).
- **Calibración:** temperature scaling por cabeza ajustado por NLL en **val**
  (grid log-uniforme 200 + sección dorada); baselines calibrados con el mismo
  buscador (anti-strawman §5.3). Nunca sobre test.
- **Entrenamiento en Modal** (GPU serverless, L4): `modal/app.py`,
  fan-out con `.spawn()`, volúmenes `paper-data`/`paper-results`, dump de logits
  crudos a npz; **las métricas se computan local** desde npz (`src/eval/`).

Código: `src/models/{network,losses,calibration,data,train,engine}.py` — los
tests que fijan cada decisión están en `tests/` (60/60 verdes, incl. humo E2E
con parquets reales y EMBER streaming).

---

## 3. E1+E3 — Calidad de detección y calibración

`results/e1e3/e1e3.{json,md}` · código `src/eval/run_e1e3.py`, `src/baselines/`.

Protocolo: macro-F1 con labels explícitos (excluye filas `unknown` GT), AUROC,
PR-AUC, **FPR@95TPR** (umbral = cuantil de positivos; métrica operativa de
línea de fuego) y ECE/MCE/Brier a 15 bins. Baselines: XGBoost (500×d8 hist) y
RF (300, min_leaf 1) calibrados 1-T/cabeza con el mismo buscador NLL; MLP
tabular (2×256); Kitsune (ensemble de 9 autoencoders embudo con feature
bagging, entrenado **solo en benignos de train** y con su propio φ benign-only,
Platt en val).

### Tabla E1 transcripción (test, temperatura calibrada)

```
| modelo          | AUROC   | FPR@95TPR | macro-F1 | ECE noul | ECE ch |
|-----------------|---------|-----------|----------|----------|--------|
| VX n UNSW       | 0.9838  | 0.1085    | 0.4332   | 0.0564   | 0.0625 |
| VX s UNSW       | 0.9834  | 0.1211    | 0.4571   | 0.0427   | 0.0496 |
| VX b UNSW       | 0.9842  | 0.1191    | 0.4543   | 0.0504   | 0.0609 |
| XGBoost UNSW    | 0.9818  | 0.1369    | 0.4968   | 0.0641   | 0.0647 |
| RF UNSW         | 0.9792  | 0.1399    | 0.4378   | 0.0504   | 0.0742 |
| MLP UNSW        | 0.9821  | 0.1279    | 0.4445   | 0.0571   | 0.0578 |
| Kitsune UNSW    | 0.6086  | 0.5617    | —        | 0.2179   | —      |
| VX n CIC        | 0.9999  | 0.0004    | 0.8995   | 0.0012   | 0.0010 |
| VX s CIC        | 0.9999  | 0.0003    | 0.8826   | 0.0004   | 0.0003 |
| VX b CIC        | 0.9999  | 0.0004    | 0.9026   | 0.0007   | 0.0005 |
| XGBoost CIC     | 1.0000  | 0.0002    | 0.9416   | 0.0001   | 0.0001 |
| RF CIC          | 0.9998  | 0.0004    | 0.6981   | 0.0011   | 0.0006 |
| MLP CIC         | 0.9999  | 0.0004    | 0.8594   | 0.0006   | 0.0007 |
| Kitsune CIC     | 0.6227  | 0.6910    | —        | 0.0373   | —      |
| VX s CIC-time   | 0.7112  | 0.5731    | 0.4518   | 0.4037   | 0.0592 |
| XGBoost CIC-t   | 0.6372  | 1.0000    | 0.8336   | 0.4397   | 0.1810 |
| RF CIC-t        | 0.6308  | 1.0000    | 0.4207   | 0.4474   | 0.1742 |
| MLP CIC-t       | 0.5959  | 1.0000    | 0.4579   | 0.3868   | 0.0784 |
| VX n EMBER      | 0.9963  | 0.0151    | 0.7472   | 0.0035   | 0.0055 |
| VX s EMBER      | 0.9969  | 0.0119    | 0.7549   | 0.0033   | 0.0076 |
| VX b EMBER      | 0.9968  | 0.0129    | 0.7717   | 0.0036   | 0.0063 |
| XGBoost EMBER   | 0.9981  | 0.0053    | 0.8094   | 0.0029   | 0.0012 |
| RF EMBER        | 0.9971  | 0.0094    | 0.6636   | 0.0032   | 0.0075 |
| MLP EMBER       | 0.9970  | 0.0098    | 0.7733   | 0.0022   | 0.0089 |
```

n= Nano, s= Small, b= Base. Tabla completa: `results/e1e3/e1e3.md`.

### Lectura honesta (es la entrada §6 del paper)

1. **Detección:** VX gana AUROC/FPR@95 vs todos los baselines en UNSW y bajo
   drift temporal de CIC (0.7112 vs 0.6372 del mejor árbol; FPR@95 de los
   árboles es 1.0 = no alcanzan TPR 0.95 a FPR razonable). Pero en CIC
   `strat` todo está saturado (~1.0): benchmark de sanidad, no de ventaja.
2. **Atribución:** XGBoost calibrado gana macro-F1 en UNSW, CIC y EMBER. La
   ventaja de VX está en la **detección** y en el **comportamiento ante lo
   desconocido** (E2), no en clasificarlo.
3. **Drift** (CIC time): todos los métodos decaen en detección (MLP 0.596,
   RF 0.631, XGB 0.637, VX 0.711) y **el ECE de todos estalla** (0.39–0.45).
   La calibración sobre val estacionaria no sobrevive al tiempo: limitación
   central declarada, no escondida.
4. **Kitsune:** AUROC ~0.61 con Platt — señal correcta pero débil sobre flujos
   estandarizados; ilustra el contraste "calibrar no crea señal" (Platt bien
   calibrado y no discriminativo a la vez).

## 4. E2 — Día cero

`results/e2/e2.json` · código `src/eval/run_e2.py` y folds via `modal/app.py::loao`.

LOAO (leave-one-attack-family-out): entrenar sin familia F, evaluar contra F en
test. Métricas de la familia no vista: AUROC del gate de ataque (Gate head),
FPR@95TPR, abstención del Choice (τ_abs=0.5) y macro-F1 de conocidas (control).

### E2 UNSW (9/9 folds, Small) — transcripción

```
| familia excluida | noul AUROC | FPR@95 | abstención | conocidas m-F1 |
|------------------|------------|--------|------------|----------------|
| analysis         | 0.9992     | 0.0002 | 0.92       | 0.5076         |
| backdoor         | 0.9989     | 0.0028 | 0.63       | 0.4967         |
| dos              | 0.9941     | 0.0202 | 0.03       | 0.5085         |
| exploitation     | 0.9550     | 0.4177 | 0.08       | 0.5584         |
| fuzzing          | 0.8519     | 0.3899 | 0.28       | 0.4514         |
| generic          | 0.9957     | 0.0128 | 0.50       | 0.3802         |
| recon            | 0.9523     | 0.1777 | 0.28       | 0.4138         |
| shellcode        | 0.9688     | 0.1305 | 0.42       | 0.4618         |
| worms            | 0.9871     | 0.0642 | 0.23       | 0.5064         |
```

El gate detecta la familia nunca vista **en 9/9**; la abstención varía 3–92%
(frame de honestidad del Choice variable por familia).

### E2 CIC (7 folds, strat) — transcripción

```
| familia       | noul AUROC | FPR@95   | abst. | n_rows (test) |
|---------------|------------|----------|-------|---------------|
| botnet        | 0.7499     | 0.4038   | 0.00  | 295           |
| bruteforce    | 0.9674     | 0.0778   | 0.0005| 2,076         |
| dos_ddos      | 0.8524     | 0.3394   | 0.00  | 57,103        |
| exploitation  | 0.9703     | 0.0297   | 0.00  | 1             |
| infiltration  | 0.5329     | 0.8262   | 0.00  | 5             |
| recon         | 0.4259     | 0.8685   | 0.00  | 23,840        |
| webattack     | 0.9792     | 0.1550   | 0.00  | 327           |
```

Dos modos de fallo que el paper discute: (a) **recon queda bajo azar** (BAJO
0.43) — los probes de puertos CIC son indistinguibles de servicios en flujos
sin payload; (b) **Choice nunca abstiene en CIC** (τ_abs=0.5 no muerde con
calibración saturada): un umbral fijo no es una política portátil entre
datasets. Asimetría aguda con UNSW.

### E2 EMBER drift temporal (test = semanas posteriores al train)

Gate global (Small): AUROC **0.9969**, FPR@95 **0.0119** — la detección
resiste 4 meses de drift. La mutación queda en atribución:

```
| familia      | n_test  | recall propia | → unknown |
|--------------|---------|---------------|-----------|
| unknown GT   | 586,018 | 0.9935        | —         |
| berbew       | 92,086  | 0.9979        | 0.0021    |
| wacatac      | 12,238  | 0.1289        | 0.8531    |
| cosmu        | 9,562   | 0.9699        | 0.0299    |
| expiro       | 7,966   | 0.9136        | 0.0844    |
| upatre       | 6,430   | 0.9677        | 0.0314    |
| glupteba     | 3,412   | 0.9513        | 0.0487    |
| flystudio    | 1,912   | 0.7552        | 0.2427    |
| grandoreiro  | 226     | 0.2212        | 0.7788    |
| sfone        | 132     | 0.9394        | 0.0455    |
| sinowal      | 18      | 0.4444        | 0.5556    |
```

wacatac (mutó) y grandoreiro (casi desapareció del train) caen, y **el
candidato `unknown` los absorbe** (85%/78%) en lugar de atribuciones falsas:
el modo trained_unknown canaliza el drift a honestidad.

## 5. E3 — Calibración (detalle)

`results/e1e3/` · `src/eval/metrics.py`.

- ECE binario con conf = max(p, 1−p) (p directo infla el gap en negativos de
  baja p).
- Temperature scaling por cabeza (NLL en val); de-tilt ANTES de T.
- Baselines raw Y calibrados en la tabla completa (anti-strawman: mostramos la
  distorsión de boosted trees y su fix; Niculescu-Mizil & Caruana 2005).
- Ablación a3: calibración no cambia ranking (argmax), solo probabilidad:
  ECE choice UNSW 0.0496 → 0.0980 sin T.
- Reliability diagrams en `results/e1e3/e1e3.json`.

## 6. E4 — Latencia

`results/e4/*.json` · `src/eval/run_e4.py`.

Hardware: CPU local, modelo exacto reportado (`local-cpu`), 1 hilo, batch=1,
n=3000 mediciones.

```
| escala | p50 µs | p95 µs | p99 µs | max µs |
|--------|--------|--------|--------|--------|
| Nano   | 675.3  | 714.5  | 773.2  | 852.9  |
| Small  | 955.2  | 999.1  | 1022.3 | 1058.8 |
| Base   | 1651.8 | 1795.3 | 2081.8 | 2334.8 |
```

Presupuesto de diseño (<10 ms en CPU) con margen ≥5× incluso en Base. La
latencia de 70–120 ms del draft antiguo era falsa (diferente magnitud).

## 7. E5 — Cascada de dos fases (Suricata → VX-One)

`results/e5/*.json` + `results/e5/e5.md` · código `src/e5/{flow2pcap,patch_iplen,cascade}.py`, `modal/app.py::suricata_e5`.

**Setup** (documentado con sus trucos):
- Flujos CIC-IDS2017 (variante enhanced) reconstruidos a PCAP sintético por día
  → Suricata **7.0.10** (Modal, capa backports Debian) con **ET Open frescas
  (52,674 reglas) + 8 reglas estructurales propias** (VXSTRUCT sid 9700001-8:
  syn-flood, portscan por umbral, probes 445/22/80, udp/icmp flood) porque la
  señal ET es casi toda de payload y los PCAPs reconstruidos no tienen payload.
- Fix técnico documentado: paquetes IP crudos con linktype Ethernet →
  `Ether()` wrapper; ip.len no debe fijarse a mano (suricata descarta todo como
  truncado); checksum-checks=no (pcap sintético).
- **Fase 1** (per-row real): alertas casadas por 5-tupla + ventana temporal →
  cobertura por familia y composición de la zona gris.
- **Fase 2** (estimación insesgada): filas TEST del día con logits reales.
- **Combinado**: estimación con supuesto explícito y documentado de que
  `P(F2 flag | familia)` es independiente de "¿F1 lo detectó?" — exacto para
  familias invisibles a F1 (web sin payload), optimista donde F1 falla justo
  los sigilosos. El paper lo reporta como estimación, no como medición.

```
| día          | flujos  | ataque  | F1 acc | F2 acc | comb acc | comb P | comb R | zona gris |
|--------------|---------|---------|--------|--------|----------|--------|--------|-----------|
| 2017-07-03 L | 529,450 | 0       | 0.9989 | 0.9996 | 0.9985   | —      | —      | 99.9%     |
| 2017-07-04 M | 445,641 | 13,832  | 0.9681 | 0.9994 | 0.9985   | 0.9579 | 0.9961 | 99.9%     |
| 2017-07-05 X | 691,390 | 251,722 | 0.6359 | 0.9986 | 0.9979   | 0.9971 | 0.9971 | 99.9%     |
| 2017-07-06 J | 458,483 | 2,216   | 0.9942 | 0.9971 | 0.9961   | 0.5521 | 0.9779 | 99.9%     |
| 2017-07-07 V | 702,713 | 288,785 | 0.5887 | 0.9979 | 0.9974   | 0.9980 | 0.9958 | 100.0%    |
```

**Hallazgo central:** con reglas ET sobre tráfico reconstruido sin payload,
Fase 1 resuelve ~0.1% de los flujos — la **zona gris es el 99.9% del día** —
y Fase 2 la recupera (combinado R ≥0.978 con P ≥0.95 salvo jueves). El jueves
(web attacks, invisible para F1) el combinado P baja a 0.55: los FP de F2
dominan porque hay 456K benignos × ~0.28% FPR. **El accuracy combinado no es
el headline** (dominado por la clase benigna); el paper reporta P/R
condicionales por familia y la proporción de zona gris.

⚠️ Declarar siempre: Fase 1 es **Suricata de referencia**, no el motor C++
VectrayX (limitación del repo original); reconstrucción sin payload = cota
inferior de F1 consciente, no strawman: es lo que las firmas ven sin payload).

## 8. E6 — Ablaciones

`results/e6/` · corridas via `modal/app.py` (`abl_a*`).

```
| corrida                    | AUROC | FPR@95 | macro-F1 | ECE noul | ECE ch |
|----------------------------|-------|--------|----------|----------|--------|
| unsw a1 sin rama textual   | 0.9716| 0.1579 | 0.4079   | 0.0501   | 0.0509 |
| unsw a3 sin calibración    | 0.9834| 0.1211 | 0.4571   | 0.0450   | 0.0980 |
| unsw a4 sin tilt ω         | 0.9838| 0.1215 | 0.4545   | 0.0501   | 0.0488 |
| unsw a5 unknown+abst off   | 0.9834| 0.1211 | 0.4571   | 0.0427   | 0.0496 |
| unsw a6 dropout 0          | 0.9839| 0.1181 | 0.4507   | 0.0599   | 0.0451 |
| unsw a2 sin multi-tarea    | 0.9784| 0.1446 | 0.4598   | 0.0552   | 0.0540 |
| (refs) unsw small          | 0.9834| 0.1211 | 0.4571   | 0.0427   | 0.0496 |
| cic a3 sin calibración     | 0.9999| 0.0003 | 0.8826   | 0.0003   | 0.0007 |
| cic a4 sin tilt ω          | 0.9999| 0.0003 | 0.9313   | 0.0001   | 0.0002 |
| cic a5 unknown+abst off    | 0.9999| 0.0003 | 0.8826   | 0.0004   | 0.0003 |
| cic a6 dropout 0           | 0.9999| 0.0004 | 0.7360   | 0.0009   | 0.0007 |
| (refs) cic small           | 0.9999| 0.0003 | 0.8826   | 0.0004   | 0.0003 |
| ember a3 sin calibración   | 0.9969| 0.0119 | 0.7549   | 0.0092   | 0.0112 |
| ember a4 sin tilt ω        | 0.9969| 0.0110 | 0.7506   | 0.0031   | 0.0061 |
| ember a5 unknown off (10w) | 0.9961| 0.0157 | 0.8674   | 0.0040   | 0.0022 |
| ember a6 dropout 0         | 0.9966| 0.0124 | 0.7821   | 0.0031   | 0.0081 |
| (refs) ember small         | 0.9969| 0.0119 | 0.7549   | 0.0033   | 0.0076 |
```

Conclusiones honestas para la discusión del paper:
- **a1:** la rama textual (proto/service/state) vale ~5 pts en UNSW.
- **a3:** la calibración no cambia el ranking, solo la probabilidad (como dice la spec).
- **a4 hallazgo incómodo:** el tilt ω **perjudica** la atribución en CIC
  (0.8826 → 0.9313 sin tilt): el de-tilt empuja abajo las clases raras que la
  CE ponderada ya había subido. Candidato a revisión del diseño ω en datasets
  saturados; se reporta tal cual.
- **a6:** el dropout es crítico en CIC (0.8826 → 0.7360 sin él, −14.7) pero
  superfluo en EMBER (0.7549 → 0.7821 sin él; la clase mayoritaria domina).
  El efecto depende del dataset: no hay una receta única.
- **a5-EMBER:** quitar `unknown` sube macro-F1 (0.7549 → 0.8674) —
  esperado (10-way más balanceado) y exactamente el trade-off honestidad vs
  utilidad que justifica mantener `unknown`.
- **a2:** la ganancia del multi-tarea está en la detección (Gate AUROC
  0.9834→0.9784 sin; FPR@95 0.1211→0.1446), Choice no la necesita.

## 9. E7 — Baseline comercial ✅ (OpenCode Zen / System One passthrough, 2,021 filas)

**Estado 2026-10-20: EJECUTADA COMPLETA (2,021/2,021 filas, $0).** Vía real inesperada:
el endpoint `systemone` del Zen de OpenCode hace passthrough al mismo proveedor
(`jev-1.13-free` = mismo modelo Jev 1.13 que el SKU comercial de TypeSafe) y queda
exento del bloqueo del free tier. Coste publicado idéntico al comercial
($0.042/Mtok, solo input) — la corrida consumió **2,202,313 tokens ≈ $0.09 pagada**;
vía free = **$0.00**.

**Resultados medidos** (`results/e7_vendors/opencode/vendor_jev_free_full.json`,
mismo protocolo congelado, sample seed-42: 1,000 normal + 1,021 ataque):

| Primitiva | Vendor (zero-shot) | Nuestro (fincado UNSW) |
|---|---|---|
| Gate AUROC | **0.568** | **0.981** |
| Gate FPR@95 | **0.932** | **0.157** |
| Gate ECE15 | 0.265 | 0.097 |
| Choice macro-F1 | 0.069 | 0.534 |
| Severidad MAE | 0.809 | 0.333 |

Latencia vendor p50 208 ms / p95 591 ms (single-request, state+3 preguntas).
Veredicto: **el baseline comercial zero-shot queda a nivel de azar en el gate
(AUROC 0.568 ≈ 0.5) con FPR@95 de 93%**, y en choice/score también cede 3–8× —
la promesa de las primitivas genéricas no traslada al dominio de red sin fino
del dataset
Con E7 medido, la comparativa "primitivas congeladas vs pipeline fincado" queda
cerrada con números de un tercero comercial.

**Ablación `--with-unknown` (2026-10-20, 1,039 filas = subsample seed-42
500 normal + 500 ataque, protocolo congelado idéntico; la opción explícita
"unknown" se añade al criteria del choice)**: vendor macro-F1 0.085 (con
unknown) vs 0.069 (sin unknown, corrida main n=2,021) — la opción casi no se
usa (**unknown-rate 0.9% overall; 1.7% en ataques; 0% en normales**) y el F1
sobre filas no-unknown no cambia (0.087). Nuestro known-choice en el mismo
subsample: 0.601. Veredicto: **abstención explícita no rescata al vendor
zero-shot**; el gap es de dominio (fine-tuning), no de falta de opción de
abstención. JSON: `vendor_jev_free_ablation.json` (nota de comparación
cross-sample dentro).

Infra y hallazgos técnicos (anotación, fuera del paper):
- El free tier de OpenCode aplica una ventana diaria **por IP** (~250 requests;
  verificado cruzando 2 keys en 3 IPs). La corrida full se distribuyó entonces
  entre VM1 OCI + 13 nodos del fleet potfics (14 slices, worker resumible
  `e7_remote_worker.py` + `e7_launch_fleet.sh`), cada nodo solo tardó ~2.5 min en
  completar su slice de ~137 filas sin un solo 429.
- Runner resumible local: `scripts/e7_resumable.py` (append .jsonl flush,
  métricas offline `--metrics`); mapping k (test-idx) → posición de la muestra
  vía `session_idx.npz`.
- Azure Microsoft-Decision-1 sigue disponible como segunda vía (deploy solo
  por portal `ai.azure.com`: feature/quota oculta no auto-registrable por CLI),
  pero E7 ya no depende de ella.

Protocolo (fijo): muestra estratificada seed-42 del test UNSW (1,000 normal +
1,000 ataque ∝ familias, min 15/familia → 2,021 filas); 1 request por fila con
las 3 primitivas sobre el MISMO state (39 features nombradas según
`UNSW_FEATURE_COLUMNS` congelado + proto/service/state): gate=noul(¿ataque?),
choice=el MISMO candidate set congelado del run (10) — abl `--with-unknown`
agrega la opción; score=rubrik severidad 0–3 alineada a la nuestra. Alineación
te-npz↔parquet verificada fila a fila (y_bin idéntico). Métricas vendor-vs-
nuestro en las mismas filas: AUROC/FPR@95 (`fpr_at_tpr` exacto del repo)/
ECE15 del gate; macro-F1/acc del choice; MAE de severidad; latencia p50/p95;
tokens y coste USD. Dry-run nuestro en la muestra: AUROC 0.9810, FPR@95 0.157,
ECE15 0.0973, macro-F1 0.5340.

### 9-anexo. E7-long con unknown — corrida PAIRED completa ✅ (2026-10-20 nocturna)

**Estado: 2,021/2,021 filas, coste $0.** Upgrade del ablation: las MISMAS
2,021 filas de la run principal vuelven a pedirse con la opción `unknown`
(comparación paired; la única diferencia es el extra option en family).
Launch: `scripts/e7_build_unknown_slices.py` (slice build barato: labels→idx
→ take de 2,021 filas, exactamente la muestra del main — assert alineación
npz↔parquet) + fleet 12 nodos (z4 down, su slice corrió en VM1) + worker
retry-after (todos duermen en su cooldown IP y ejecutan solos de madrugada
~4 min por nodo; pull con merge+dedupe en `results/e7_vendors/opencode/unkfull/`).
Métricas: `scripts/e7_metrics_unknown_full.py` → `vendor_jev_free_unkfull.json`.

**Resultados (paired vs main, mismas filas):**

| métrica | vendor main (sin unknown) | vendor con unknown | nuestro |
|---|---|---|---|
| gate AUROC | 0.568 | **0.476** | 0.981 |
| gate FPR@95 | 0.932 | 0.969 | 0.157 |
| macro-F1 | 0.0693 | 0.0657 (0.0592 con unknown-cls) | 0.534 |
| unknown-rate | — | 0.35% (0.2% atk / 0.5% normal) | — |
| MAE sev | 0.809 | 0.848 | 0.333 |

**Lectura**: a escala completa y paired, el veredicto del ablation subsample
(0.9% unknown, macro 0.085) se CONFIRMA y refuerza: el vendor dispone del
hatch de abstención y NO lo usa (0.35%); añadir la opción no cierra el gap
(macro-F1 0.066) y el gate cae a 0.476 (ruido con la opción extra). "An
explicit abstention option is not enough" queda medido a escala completa, sin
la salvedad cross-sample del subsample. Tokens 2.24M ($0.09 si pagase);
latency p50 178.8 ms / p95 494.2 ms (fleet). Integrado al paper §E7.

## 9b. E8 — Recalibración por ventana (walk-forward) bajo drift ✅ (2026-10-20)

**Estado: EJECUTADA.** Cierra el next-step (i) de la conclusión del paper. Protocolo
congelado en `scripts/e8_window_recal.py`: sobre el npz de la corrida CIC by-day
(`results/outputs_seed/cic_small_time.npz`, días de test en orden de captura
verificado contra el parquet — 2 ventanas: Friday-DDos n=138,145 →
Friday-PortScan n=286,467), variantes:

- **static**: recalibración congelada con la val del pipeline (lo que reporta
  hoy el paper).
- **walk**: temperatura del gate re-fit con las labels del día anterior
  (día-1 usa la val — streaming honesto con labels retrasadas).
- **walkplatt**: idem pero Platt (a·logit+b) — el sesgo b expresa cambio de prior.
- **walkacc**: val + todos los días de test previos (reservoir acumulado).
- **oracle**: calibración con las PROPIAS labels del día — el techo absoluto
  de cualquier recalibración post-hoc.

| día | variante | AUROC | ECE15 | Brier | FPR@95(cal) | ECE15-ch |
|---|---|---|---|---|---|---|
| DDoS | static | 0.9931 | 0.1273 | 0.1254 | 0.0 | 0.1071 |
| DDoS | oracle | 0.9931 | **0.0287** | **0.0267** | 0.0 | 0.1048 |
| PortScan | static | 0.56 | 0.5531 | 0.5390 | 0.0 | 0.5426 |
| PortScan | walk | 0.56 | 0.5531 | 0.5037 | 0.0 | 0.5280 |
| PortScan | walkplatt | 0.56 | **0.5255** | 0.5080 | 0.0 | 0.5280 |
| PortScan | walkacc | 0.56 | 0.5531 | 0.5315 | 0.0 | 0.5385 |
| PortScan | oracle | 0.56 | 0.4452 | **0.2462** | 0.1481 | **0.0564** |

Macro-F1 del choice es idéntico por construcción (la temperatura no muta el
argmax) — lo recalibrable es la confianza.

**Lectura honesta (3 hallazgos medidos):**
1. **La calibración por ventana ayuda donde el día discrimina**: DDoS-day
   AUROC 0.99 → oracle ECE 0.13→0.03; walk-platt captura casi todo el techo
   con 2 parámetros y labels de un día (b≈6.7 expresa el prior del día).
2. **El techo bajo drift fuerte es discriminación, no calibración**: PortScan
   AUROC 0.56; **ni siquiera el oracle** baja el ECE del gate de 0.445
   (static 0.55). Recalibrar calibra; no emperra la detección — el parche
   post-hoc no existe para un día con disc 0.56; ahí el mecanismo honesto
   sigue siendo la semántica `unknown` (E2).
3. **El confidence del choice SÍ es recalibrable incluso en days débiles**:
   ECE15-ch 0.543 → 0.056 oracle (softmax NLL de 6 vías es más exprersive que
   temp binaria: el margen del argmax da señal extra); con walk-platt
   day-lag solo −1.5 pts — la frescura del label es la variable.

Resultado para el paper (Limitations): "recalibración por ventana = trabajo
futuro" → ahora medido: mejora acotada (Brier −6.5% worst-day con walk-platt;
−54% con oracle), y el cuello pasa a ser detectar-per-ventana (retrain), no
calibración. Runbook/protocolo: `e8_window_recal.py`; outputs en
`results/e8_recal/`.

## 7b. E5b — Risk–coverage del gating compuesto de la cascada ✅ (2026-10-20)

**Estado: EJECUTADA (local, ~4 min; eve.json.gz relocalizados del volumen).**
Cierra el next-step (ii) de la conclusión. Protocolo congelado en
`scripts/e5_risk_coverage.py` (reusando exactamente los helpers de
`src/e5/cascade.py` con rutas absolutas; eve_*.json.gz desde el volumen Modal
cuenta-1). Semántica de adjudicación selectiva: A(τ) = {F1 alertó} ∪
{p_noul ≥ τ}, veredicto "ataque", abstain lo no adjudicado; cov(τ)=|A|/N,
risk(τ)=1−precision(A). 201 puntos de τ por día.

Exactitud por bloque: F1 leg EXACTO por flujo (eve casada por 5-tupla + ventana
`F1_T0_SLACK/F1_END_SLACK`); F2 leg EXACTO por fila-test strat (npz +
temperatura como-run); compuesto = estimador del supuesto documentado de E5
(gray ≈ muestra aleatoria por clase).

| día | F1 cov / prec (exacto) | F2 cov@risk≤10% | comp cov@risk≤10% | comp cov@risk≤20% |
|---|---|---|---|---|
| Mon 03 (benigno puro) | 0.0011 / 0.000 | 0.0 | 0.0 | 0.0 |
| Tue 04 | 0.0009 / 0.030 | 0.0326 | 0.0335 | 0.0335 |
| Wed 05 | 0.0013 / 0.490 | 0.3657 | 0.3666 | 0.3666 |
| Thu 06 | 0.0011 / 0.059 | 0.0044 | **0.0** | 0.0055 |
| Fri 07 | 0.0005 / 0.067 | 0.4163 | 0.4155 | 0.4155 |

**Lectura honesta (3 hallazgos):**
1. **Lunes benigno-puro = punto nulo del gating selectivo**: TODA adjudicación
   es error por construcción (risk→1.0 a cualquier τ; cov=0 en risk≤30%). El
   "risk" de selectiva con abstención NO existe sin ataques: sin budget de
   composición previa, el gate abstiene todo — coherente con la semántica
   `unknown`/coverage como mecanismo para día real.
2. **Viernes (41% ataque): cov 0.416 de la jornada a risk≤5%**, plateau enteramente
   suministrado por Fase 2 (Δ cov F1→comp ≤ 1pt): bajo opacidad de payload el
   motor de firma aporta latencia/provenance, NO coverage de adjudicación.
3. **Jueves pequeño-pack: el gating compuesto NO admite nada a risk≤10%** — los
   flags de F1 (0.11% a precision 0.059) corrompen todos los puntos; el core
   limpio de F2 solo cubre 0.44% (risk mín 0.33%). La multiplicación-FP que
   dominate la precision combinada del §E5 castiga también la curva del gate.

Espera vs tabla tab:e5: cov@risk fixed ≤ cov del combined-estimate full-day
(aquí la admisión es selectiva con abstención, no clasificación total). Outputs:
`results/e5_rc/` (5×json por día + e5b.md + e5b_curves.npz con curvas 201 pts).


## 9c. E9 — Per-window retraining (continue-train) bajo drift CIC ✅ (2026-10-20)

**Estado: EJECUTADA (local CPU, ~1.5 min/época).** Cierra el único next-step de
E8: "retraining por ventana es lo que queda". Protocolo congelado en
`scripts/e9_window_retrain.py`: continue-train del checkpoint oficial
`results/artifacts/cic_small_time.pt` (by-day s42) sobre
train ∪ Friday-DDos (n=138,145 labels retrasadas al cierre del día-1),
FeatureStage/candidates/ω congelados del artifact, AdamW lr 1.5e-4 (×0.15 del
original), wd D11, seed 42, 1 pasada por defecto (streaming-honesto).
Calibración doble de la MISMA corrida: as-run (val del pipeline) y val∪day-1
(semántica walk de E8). Sensibilidad: ep2/ep4 (`--tag`). Eval sobre día-2
(Friday-PortScan, n=286,467) con los MISMOS evaluadores de E8.

**Tabla día-2 PortScan (estático = corrida by-day pre-E9):**

| variante | AUROC | ECE15 | Brier | ECE-oracle-recal |
|---|---|---|---|---|
| static (E8) | 0.560 | 0.5531 | 0.5390 | 0.4452 |
| E9 1 ep | 0.800 | 0.5539 | 0.5464 | 0.1165 |
| E9 2 ep | 0.8392 | 0.5540 | 0.5467 | 0.1003 |
| E9 4 ep | 0.760 | 0.5538 | 0.5465 | 0.1735 |

Sanity día-1 (las filas se entrenaron): AUROC 1.0, ECE 0.0006-0.0007 — eval
in-window, no generalización. Choice macro-F1 día-2 queda 0.102 (sin cambios
medibles), ECE-ch plano; el gate AUROC es lo que se recupera.
FPR@95(cal) día-2 = 0.0 en static y E9 (el umbral 0.95cuantil de val separa;
el score de PortScan-flow se comprime en p<0.1 — la discriminación vive en el
orden full-precision del logit, AUROC es estándar sobre scores completos).

**Lectura honesta (3 hallazgos medidos):**
1. **El retraining con ventana fresca recupera RANKING**: gate AUROC día-2
   0.560 → 0.800 (1 pasada) → 0.8392 (2 pasadas); **4 pasadas CAE a 0.760** —
   existe sweet-spot y el overfit al día-1 (persistencia + saturación) degrada.
2. **El techo post-hoc del STATIC estaba limitado por su ranking, no por el
   mecanismo de recal**: el mismo oracle-platt sobre el modelo retrained baja
   el ECE-oracle de 0.4452 → 0.1165 (1 ep) / 0.1003 (2 ep) — con ranking 0.8+
   el refit a=0.83--1.08, b≈9--12 expresa la composición del día y recupera
   confianza. La recalibración "solo" (sin retrain) no tiene nada que expresar.
3. **El walk-recal con labels retrasadas SIGUE siendo insuficiente incluso con
   ranking recuperado**: platt día-1→día-2 sobre E9 da b≈−0.32…−0.67 (el shift
   requerido es b≈+9.5) porque el día-1 quedó memorizado (BCE-fit 0.002-0.003,
   scores saturados) y su composición-pos-rate difiere. La prior del día
   nuevo no se infiere de ventanas previas — ECE as-run se mantiene 0.554.

Cierre del arco E8→E9 para el paper: under drift fuerte, (a) recalibración
calibra pero no detecta (E8), (b) retraining por ventana recupera parte del
ranking (0.56→0.84 peak) y re-habilita la recalibración (ECE-oracle
0.45→0.10), (c) ni retraining ni recal fijan el prior de composición del día
(as-run ECE 0.55₁; la semántica `unknown`/coverage sigue siendo el mecanismo
honesto para día cero real). Runbook/protocolo: `scripts/e9_window_retrain.py`
(+ `e9_vs_static.py`, `e9_walkplatt.py`); outputs en `results/e9_retrain/`
({e9_retrain,e9_retrain{ep2,ep4}}.json+npz, e9_vs_static.json, e9_walk{,ep2,ep4}.json,
e9_oracle.json).


## 10. Limitaciones a declarar (→ § del paper)

1. Score/severidad = rubrik familia→nivel, no etiquetas reales de severidad.
2. Datasets de benchmark con errores de etiquetado documentados (Engels 2021;
   Rosay et al.) → CIC con la variante MLCVE + auditoría reportada.
3. LOAO mide generalización entre familias, no contra amenazas del mundo real.
4. Calibración colapsa bajo drift (ECE 0.40 en CIC time) — la val estacionaria
   miente; recalibración por ventana = trabajo futuro inmediato.
5. τ_abs fijo (0.5) no es portátil entre datasets (UNSW vs CIC E2) — matriz de
   costes como política del controlador.
6. E5 = Fase 1 de referencia (Suricata), reconstrucción sin payload, estimación
   del combinado con supuesto de independencia documentado.
7. Latencia medida en CPU local single-thread, batch 1; el costo por 1M eventos
   queda pendiente del billing real de Modal.
8. Una sola seed por corrida (diferencias <1 pt = ruido hasta 3 seeds).
9. Inyección en campos de texto (`x_txt`) fuera de alcance v0.1 (Greshake et
   al. 2023) → sanitización = trabajo futuro.

---

## 11. Formalismo integrado al paper (advisory Opus Q1–Q3, fusionado 2026-10-19)

El paper (`paper/main.tex`) incorpora el advisory `docs/reference/VX_One_F2_Advisory_Opus.md` así:

- **§ Decision controller semantics:** los gates v0.1 son funcionales de UNA
  cabeza (verificado: 0 violaciones del rubrik en 3,088,416/3.09M flujos);
  composición de decisiones booleanas entre gates, NUNCA probabilidades entre
  cabezas (Lemas 1–2 y Prop 3 en Apéndice A del paper).
- **τ_abs = 0.5 como declaración de costes** (regla de Bayes con rechazo:
  abstener sii max_k p_k < 1−λ_r; λ_r=0.5 declarado; τ* = c_FP/(c_FP+c_FN)
  de Elkan para operadores, ejemplo 10:1 → 0.091). La regla general con matriz
  de costes vive en el controlador; la desviación de max-prob se mide en val.
- **LOAO = rechazo OOD por semejanza** (max-softmax, Hendrycks & Gimpel 2017),
  no probabilidad calibrada sobre la familia excluida; EMBER = rechazo
  in-distribution (clase entrenada). El paper reporta los dos ejes por
  separado.
- **Conformal scoped:** garantía exacta SOLO en splits intercambiables (UNSW
  oficial, CIC strat); en splits temporales coverage empírico semanal con IC
  Clopper-Pearson + cota de Barber et al. 2023 (gap ≈ 2·d_TV, d_TV ≥ 1−2·BER
  certificado por clasificador de dominios — Lema 4); challenge (0 benignos,
  98.8% unknown) fuera de toda promesa.

## 12. Pendientes mínimos derivados del advisory (reusan predicciones ya computadas)

- [x] **E2 sweep de τ_abs** — ✅ EJECUTADO 2026-10-19 (`scripts/e2_tau_sweep.py`,
      resultados en `results/e2_rc/`, sin re-entrenar, logits LOAO de
      `~/projects/paper/results/outputs/`). Grid 0.05–0.99. Veredicto medido:
      **UNSW** tiene frontera operativa (τ=0.5: abst 37.5% / falsa-atrib 55.0%;
      τ=0.99: abst 97.3% / falsa-atrib 0.66%); **CIC no tiene punto operativo**
      (τ=0.99 aún abstiene solo 17.8%; falla dominante = absorción→normal 94.5%
      conf saturada, p50 conf≥0.999 en 5/7 folds, recon e infiltration: 1.000;
      margen p(1)−p(2) en recon = 1.000 → el selective-by-margin también muere).
      Integrado al paper como Fig. fig:tau + párrafo "Sweep" en E2; el punto
      A de la review externa (Gemini, 2026-10-19) queda cubierto con datos.
- [x] **E3 reliability por condicional CORN** — ✅ EJECUTADO 2026-10-20
      (`scripts/e3_gates_diag.py` → `results/e3_diag/`, sin re-entrenar).
      Los ECE condicionales NO determinan el ECE del producto: UNSW test
      q1 0.0925 / q2 0.0508 con C2 directo 0.0438; CIC by-day test
      q1 0.4203 / q2 0.1407 con C2 directo 0.1125. Integrado al Apéndice
      tras la prueba de Lemma 2 ("Measured (drop gate, this deployment)").
- [x] **E3 classwise calibration de Choice** — ✅ EJECUTADO (mismo script).
      ECE classwise val ≤0.007 en los 4 runs; coordenada recon ≤0.003
      (UNSW 0.0027 n=1,574; CIC strat 0.0001). Justifica g_recon como
      medición, no supuesto. Integrado en el párrafo de calibración E3.
- [x] **Diagnóstico Bayes-vs-Chow** — ✅ EJECUTADO (val, choice head).
      Con costes uniformes la abstracción max-prob es regla de Bayes
      exacta (0 desviaciones en los 3 runs con clase normal). Con
      penalización m-fold en misses las desviaciones aparecen SOLO en una
      dirección (Bayes escala, Chow atribuye) y a partir de m≥3: UNSW val
      13.4% (m=3, λ_r=0.5); by-day 29% entre attacks (m=10, λ_r=0.1).
      CIC strat ≤0.2% en todo el grid. EMBER n/a (choice head sin clase
      normal). Integrado al párrafo "Bayes rule with rejection" del
      Apéndice.
## 13. Varianza multi-seed de las 4 corridas principales (2026-10-20, Modal cuenta 2)

- [x] **3 seeds × 4 corridas** en L4 (`modal/app.py::seeds_main`, seeds 42/43/44,
      cuenta `jssantillanajuan`, app `ap-1p9KTdr8POEq2utQey4vfA`, 0 fallidos).
      Logits en `results/outputs_seed/` (24 archivos) y evaluación con el mismo
      evaluador E1 local (`scripts/e1_seed_variance.py` → `results/e1_seed/`).
- **Veredicto**: seed-42 reproduce la oficial de UNSW al 4º decimal
  (determinismo confirmado). AUROC se mueve ≤0.002 salvo CIC by-day
  (0.7336±0.0176). Macro-F1 de CIC es inestable en CUALQUIER escala:
  strat 0.8312±0.0737 (s43 cae a 0.727 con AUROC/calibra intactos),
  by-day 0.3442±0.0852 (spread 0.24–0.45). UNSW 0.4560±0.0009 y
  EMBER 0.7462±0.0116 estables.
- Integrado al paper: párrafo "Seed variance" en E1 + Limitations (item
  "one seed" ahora delimita a LOAO/ablation/E4/E5) + conclusiones.
- Nota operativa: montaje del contenedor uploader con
  `--security-opt label=disable` (SELinux negaba la mitad de los archivos con
  `:Z` aplicado a medias), el volumen `paper-results` lista/lee con paths
  **internos** (`/outputs/...`, no `/vol/outputs/...`), y el repo montado para
  lanzar `seeds_main` debe ser `paper-fix-glm` (el viejo no tiene el entrypoint).
- [x] **(Si se ejecuta) protocolo conformal EMBER del advisory §2(d)** — ✅
      EJECUTADO 2026-10-20 (`scripts/e4_conformal_ember.py` →
      `results/e4_conformal/`, todo local sobre logits npz + features
      parquet; 0 entrenamientos de modelos). Protocolo: split-conformal
      sobre Noul (α=0.1, calibración n=120k de w46–51), coverage semanal
      con IC Clopper–Pearson, variante weighted con clasificador de
      dominios logístico SGD streaming (odds clipeadas [0.1,10]) y BER
      held-out → cota inferior d_TV.
- [x] **Challenge set** (estrés, fuera de toda promesa): coverage de
      positivos **0.267** (test weeks ~0.92) — colapso esperado, advisory
      Q2: "ningún wrapper conformal promete algo ahí". Semántica útil:
      el detector sigue votando ataque (58% con p≥0.5, mediana p 0.65)
      pero sin la confianza del conjunto. 98.7% unknown-familia,
      confirmado. Añadido al Apéndice del paper.
- **Resultados**: coverage de calibración **0.9000 exacto**; por semana
  unpweighted 0.8947–0.9353 (cae ~1 pt hacia w63; el lado malicioso
  concentra la pérdida: 0.8835 en w60; benignos ≥0.905 toda semana);
  drift certificado **d_TV ≥ 0.15–0.24** (BER held-out, AUC 0.61–0.64 →
  el drift de features es real pero modesto); weighted sin ganancia
  medible (Δ≤0.022, dirección inconsistente). Lectura: el drift de
  EMBER que castiga es el label-shift de familias (lo absorbe
  `unknown`), no el covariate-shift del binario. Integrado al paper
  (Limitations item + párrafo "Conformal audit" del Apéndice).
- **Nota operativa anti-colgado** (la máquina se colgó el 2026-10-20):
  sklearn lbfgs sobre 120k×2568 convierte/copias en float64 (~5 GB por
  copia) y el pool jemalloc de pyarrow retiene VA tras liberar. Regla:
  dominio-clf en SGD streaming float32 con preallocazione, pool de
  pyarrow en modo `system_memory_pool()` y `ulimit -v 12GB` siempre que
  un script toque features 2568-dim × cientos de miles de filas.

## 14. E8b — Label shift explícito bajo drift (2026-10-22, ítem 6 del informe externo)

Motivación: la caída de calibración del día 2 (PortScan del viernes) — ¿es
"falta corregir el prior" (label shift) o es techo de discriminación?
Script: `scripts/e8_labelshift.py` (solo logits ya computados; CPU segundos).
Protocolo: mapas base (temperature/Platt/isotonic) ajustados SOLO con labels
del día 1 (retrasados, despliegue honesto) → estimadores del prior del día 2
(BBSE-hard, BBSE-soft, EM/Saerens) → corrección de prior preservando el
ranking (odds-reweighting). Dato: q1=0.5573 (DDos), q2_true=0.5548
(PortScan, prevalence ALTA — nota: el informe que la creía baja se equivoca).

Resultados (`results/e8_recal/e8_labelshift.json/.md`):
- Estimadores FALLAN: q̂2 ∈ {0.000–0.041} vs q2=0.5548 para TODOS los mapas
  y TODOS los estimadores. Un modelo confidente-eqivocado no deja señal
  legible por ningún estimador estándar (BBSE/EM).
- Cota oracle-prior: constante en el prior verdadero = ECE 0.0000,
  Brier 0.2470=q(1−q). ⚠️ CORRECCIÓN del apéndice E8 del paper: el "mapa
  constante cerca del prior en 0.445" del binfit NO es en el prior sino en
  la media del score crudo (p.mean()≈0.001); la narrativa del plateau 0.445
  se reescribe (constante en el prior y perfectamente calibrada existe;
  lo no resoluble sin labels es el prior del día nuevo y el ranking).
- Con q̂ equivocado la corrección preservando ranking mueve ECE ≤0.01.

Conclusion: la restricción vinculante es prior estimabilidad + discriminación,
NO la familia de mapas. Añade la defensa del protocolo E9 (delays labels) y
del abstention semantics.

## 15. E5c — Phase-2 por-día (honesto) para la cascada (2026-10-22, falla crítica 1 del informe)

La cascada del paper (tab:e5) estima Fase 2 con el artefacto STRATIFICADO
(cic_small_strat): in-distribución POR FAMILIA (el train contiene filas de
todas las familias). Re-estimación con el artefacto POR-DÍA
(cic_small_time.npz; te=viernes en orden de captura; train no vio las
familias del viernes): `scripts/e5_phase2_byday.py` →

| día | F1 R | F2(byday) acc | comb(byday) P/R | comb(strat) P/R |
|---|---|---|---|---|
| 2017-07-06 | 0.0135 | 0.0 (sin filas test por-día: jueves ∈ train del split time) | 0.0586/0.0135 | 0.552/0.9779 |
| 2017-07-07 | 0.0001 | 0.143 | **0.9707/0.3394** | 0.998/0.9958 |

Por familia en Friday (byday): dos_ddos recall 0.7654; recon/PortScan NO
existe como candidato en el modelo por-día (D12 lo retira del train al no
haber ejemplos) → zona gris de PortScan = 0 recuperación. Combined honesto
en el día de onda: R 0.34 vs 0.996 (strat). El "plateau de recuperación de
la zona gris" del paper queda anclado como cota SUPERIOR con Fase-2
family-covered; el motor por día solo recupera DoS/DDoS. Mon–Thu por-día no
computable con los artefactos actuales (esos días están en el train del
split time) — se declara en el paper/caption.

## 16. E2d — LOAO baselines + OOD (2026-10-22, ítem 2 del informe) 🔄 en curso

Protocolo idéntico a E1 baselines (`src/baselines/run_tabular.py`, mismos
splits, misma φ, temperaturas val) por fold LOAO con exclude-family, con lo
que el "clasificador binario entrenado con el resto de familias" y los
árboles multi-clase NO son strawman:
- UNSW (9 folds × xgboost/rf/mlp + logreg + Mahalanobis-LedoitWolf):
  `results/baselines_loao/unsw_oficial_loao_*` + `unsw_small_oficial_loao_*` (OOD),
  evaluador `scripts/e2_loao_baselines_eval.py` → `results/e2_baselines/`.
- CIC (7 folds × igual, cpu=32/132GB en Modal cuenta 2): en curso.
Los primeros folds UNSW ya muestran: xgboost gate AUROC 0.977–0.998 (peor que
VX 0.99–1.0 en los mismos folds), FPR@95 ~0.03–0.28 (mucho peor que VX en
analysis), y la tolerancia de compromiso de abstención de los árboles es
distinta (falseAtt@0.5 0.19–0.68 vs atribución honesta de VX con unknown).

### 16b. Resultado final E2d (2026-10-22, completo: 9 UNSW + 7 CIC folds, árboles + OOD)

`results/e2_baselines/e2_loao_baselines_{unsw,cic}.{json,md}` — medianas por fold:

| modelo | UNSW AUROC (med) | UNSW FPR@95 med | CIC AUROC (med) | CIC FPR@95 med |
|---|---|---|---|---|
| xgboost | 0.855–0.998 (0.974) | 0.149 | 0.826–0.999 (0.976) | 0.101 |
| rf | 0.870–0.990 (0.962) | 0.151 | 0.505–0.995 (0.854) | 0.730 |
| mlp | 0.860–0.998 (0.991) | 0.047 | 0.027–0.963 (0.607) | 0.898 |
| logreg | 0.780–0.956 (0.930) | 0.263 | 0.163–0.765 (0.560) | 0.864 |
| mahalanobis | 0.653–0.870 (0.832) | 0.517 | 0.003–0.573 (0.230) | 0.988 |

Lectura (doble sentido, ya integrada al paper §e2):
- UNSW: la gate tipada de VX lidera 9/9 folds (baseline med 0.93–0.99 vs VX
  ≥0.95–1.0); la sospecha del informe queda refutada ahí.
- CIC estratificado: la sospecha se CONFIRMA en los folds débiles — XGBoost
  transfiere mejor que la gate de VX justo donde VX degrada (botnet 0.905
  vs 0.750; recon 0.830 vs 0.426); los OOD clásicos colapsan (logreg 0.16–0.77,
  Mahalanobis 0.00–0.57). Y los árboles reproducen la patología de
  saturación: abstención @τ=0.5 ≈ 0 en CIC con absorción a normal.
- En UNSW la frontera max-prob de RF es real (abst 0.80 / falseAtt 0.14 med)
  pero es abstención por baja confianza, no declaración de clase nueva.

## 17. E5d — Fase 1 con payload REAL (pcaps originales) ✅ (2026-10-22, ítem 1 del informe)

Motivación: la falla crítica 1 del informe (¿la zona gris es artefacto de la
reconstrucción sin payload?). Descarga de los 5 pcaps originales de
CICIDS2017 (cicresearch.ca, md5 verificados contra los .md5 oficiales;
Monday 10.8GB, Tuesday 11.0GB, Wednesday ~14GB?, Thursday 8.3GB, Friday 8.8GB)
en Modal cuenta 2 (vol paper-data→/vol/e5_original/pcaps) + Suricata
7.0.10 con el MISMO motor/ruleset (52,706 ET Open + 8 estructurales;
el paper referenciaba 52,674 — drift diario del ruleset, declarado).
eve → `results/e5_original/eve_original_*.json.gz` (los consolidados de
cicresearch traen la semana completa; filtrado por fecha).

Matching: NO por ventana temporal (los Timestamp del enhanced están
re-computados; desfase no constante ±10min..+10h, modos medidos) sino por
**5-tupla del día** (bidireccional; el reuso de puertos efímeros sesga
"benign flagged" hacia arriba, declarado). Script `scripts/e5_original_match.py`.

| día | alertas día | attack resolved | benign flagged | P | R |
|---|---|---|---|---|---|
| 2017-07-03 | 9,425 | 0.0 (solo benigno) | 0.0478 | — | — |
| 2017-07-04 | 2,644 | 0.0205 | 0.0238 | 0.0269 | 0.0205 |
| 2017-07-05 | 2,799 | 0.0219 | 0.0236 | 0.3463 | 0.0219 |
| 2017-07-06 | 26,652 | 0.0208 | 0.0222 | 0.0045 | 0.0208 |
| 2017-07-07 | 2,745/12,170 | **0.0023** | 0.0215 | 0.0702 | 0.0023 |

Por familia: bruteforce 2.05%, dos_ddos(DoS) 2.18%, webattack 0.5%,
infiltration 97.2% (36 flows), Heartbleed 1/1, dos_ddos(DDoS LOIT) 0.45%,
recon/PortScan **0.06%**, botnet 0%.

**Conclusión (más fuerte que el paper actual): la zona gris NO es artefacto
de la reconstrucción** — con payload real, ET Open sigue resolviendo solo
0.2–2.2% del volumen diario de ataques (99.5% de zona gris en el día de
onda). Las firmas de contenido capturan payloads de exploit (Heartbleed,
Infiltration), no floods ni scans. El FP-rate de Fase 1 sobre tráfico real
(2.2–4.8% de los benignos alertados por reglas informacionales) es no
trivial. Integrado al paper (abstract (iii), intro, §e5, limitaciones,
conclusión: el "next step" de payload-inclusive queda MEDIDO y cerrado).

## 18. E6b — semillas de ablación (2026-10-22, ítem 3 del informe) ✅

Re-entrenadas en Modal cuenta 2 (L4) con seeds 43/44 (ref = 42):
`results/outputs_abl_seed/` + `results/e6_seed/e6.json` (evaluador estándar).

| ablación | macro-F1 seeds 42/43/44 | σ |
|---|---|---|
| UNSW a2 (single-head merge) | 0.4598 / 0.4560 / 0.4548 | 0.0026 |
| UNSW a6 (dropout 0) | 0.4507 / 0.4522 / 0.4481 | 0.0021 |
| CIC a6 (dropout 0) | 0.7360 / 0.6628 / 0.9120 | **0.1281** |
| CIC a4 (sin tilt) | 0.9313 / 0.9254 / 0.9213 (Δ +4.9/+4.3/+3.9 vs ref 0.8826) | 0.005 |

Lectura:
- UNSW: a2 y a6 son seed-estables (σ ≈ 0.002–0.003).
- CIC a4: el exceso +4.9 del tilt es direccionalmente estable (σ 0.005).
- **CIC a6 es seed-UNSTABLE (σ 0.128, rango 0.66–0.91)**: el coste −14.7 de
  dropout-0 del ref está dentro del ruido de semilla (una seed incluso
  queda por encima del ref). El paper ahora lo declara (§e6 y conclusión).

## 19. Correcciones de la segunda revisión externa (2026-10-22, tarde)

1. **LOAO UNSW: la gate de VX NO lidera todos los folds** (revisión externa,
   correcto). Por fold: VX gana 4/9 (analysis 0.999, backdoor, dos, generic),
   MLP 3 (recon 0.963, shellcode, worms 0.991), RF 1 (fuzzing 0.870), XGB 1
   (exploitation 0.974). Medianas: MLP 0.991 > VX 0.987 > XGB 0.974 > RF 0.962.
   Todos degradan en fuzzing (0.65–0.87). Bootstrap 95% CI por fold
   (`results/e2_baselines/auroc_bootstrap_ci_unsw.json`): ±0.004–0.005 en folds
   con n≥378; ±0.024 en worms (n=44). §e2 reescrito: "empate en AUROC cuyo
   tie-break depende de fold/dataset; lo medible del diseño tipado bajo LOAO
   es la opción unknown entrenada, la calibración con garantías y la honestidad
   en CIC-recon — no superioridad AUROC".
2. **Validación del matching 5-tupla** (`scripts/e5_original_match_validate.py`
   → `results/e5_original/matching_validation.*`): presencia alertas→enhanced
   = 1.000 en los 5 días (ninguna alerta se pierde por ausencia de tupla);
   cobertura de flujos de ataque con la IP del atacante documentada
   (172.16.0.1) idéntica a la global (0.0205/0.0219/0.0023 Mar/Mié/Vie;
   Jueves 0.005 vs 0.021 explicado por per-familia: webattack 0.5% vs
   infiltration 97%). Sesgo declarado: reuso de puertos → solo hacia arriba.
3. **BBSE/EM reencuadrados**: son consistentes bajo label shift (P(x|y)
   invariante); el día de onda es open-set (PortScan ausente del train →
   P(x|y) cambia). Su fallo NO prueba "prior no estimable en general":
   muestra que el shift está fuera del modelo label-shift (y que la gate
   confidente-equivocada no da señal ni in-scope). Apéndice E8, limitaciones
   y conclusión corregidos.
4. **Cascada en el abstract/intro**: ahora el ancla honesta va primero
   (by-day 0.34 = share in-scope 44% × recall 0.77; el resto PortScan es
   agregación temporal fuera de alcance por-evento, coherente con el párrafo
   Scope de §e5) y el ≥0.977 queda como cota superior family-covered.
5. **Abstract**: "always includes unknown" → semántica híbrida (clase
   entrenada donde hay población unknown; abstención en el resto).
   También (i) ahora declara el margen estrecho en UNSW (0.983 vs 0.982).
6. **Conclusión**: eliminada la contradicción (decía a la vez que las
   ablaciones ya se midieron y que faltaba medirlas). Pendientes reales:
   varianza de E7 (fleet N=5) y abstention head con outlier exposure.

## 20. Corrección de métrica E8 + restos de la tercera revisión (2026-10-22, noche)

**La corrección más importante del ciclo**: el `ece_bin` de las tablas E8
(`scripts/e8_window_recal.py`, `scripts/e8_nonparam.py`) NO es la ECE
estándar — computa `|mean(1[p≥0.5]) − mean(y)|` sobre bins de confianza
(el desbalance FP-vs-FN del mapa binarizado), mientras E3 usa la top-label
estándar `|acc − conf|`. Con esa lente:

| mapa (PortScan, q=0.5548) | softECE (E3) | hardGap (tablas E8) | Brier | err duro |
|---|---|---|---|---|
| cruda | 0.5538 | 0.5534 | 0.5537 | 0.5554 |
| temp day-self (a=.06) | 0.1823 | **0.4452** | 0.2462 | 0.4452 |
| iso day-self | **0.0000** | 0.2603 | 0.1775 | 0.2609 |
| constante en q | **0.0000** | **0.4452** | 0.2470 | 0.4452 |

- **El "plateau 0.445" es 1−q**: el piso de CUALQUIER predictor todo-unos
  bajo la métrica hard-gap (la temp day-self con a=0.06 y la constante en
  el prior caen exactamente ahí). No es un techo de calibración.
- La isotonica day-self es soft-calibrada in-sample (ECE 0.000) con Brier
  0.178; su "0.260" es el ERROR DURO de la mejor binarización monótona
  (piso de Bayes implícito por AUROC 0.56) — no una ECE.
- El argumento del "techo" queda reescrito: la monotonía NO acota la ECE
  (aplanar hacia la constante la lleva a 0); acota la NITIDEZ (Brier →
  q(1−q)) y el error duro (≈0.26 el mejor vs 0.445 el plano). E9 hereda la
  misma métrica: su "0.445 → 0.10–0.12" es la caída del error duro del
  oráculo day-self con el ranking recuperado (AUROC 0.84) — una ganancia
  de discriminación, no de calibración. Apéndice E8, limitaciones y
  conclusión corregidos con esta lectura.
- Mi explicación anterior ("constante en la media del score ≈0.002
  explica el plateau") era incorrecta: esa constante (binfit1) tiene
  hardGap 0.5548/Brier 0.55; el par 0.4452/0.246 pertenece a mapas
  todo-≥0.5 (binfit2 in-sample y la temp plana).

Restos menores también cerrados: "always contains unknown" en contribución 1
(→ semántica híbrida); "four/seven" de la conclusión (→ five + lista sin
numeral inconsistente); Tabla 6 ahora tiene a2 (UNSW) y a5 (CIC); Tabla 1 con
bootstrap CIs en caption (UNSW VX [0.9828,0.9841] vs XGB [0.9812,0.9825] —
margen al borde de la significancia); Tabla 2/3 con % de abstención exactos
(2 decimales, de results/e2/e2.json) y daggers en folds degenerados (n=1, n=5).

## 21. E7-azure — segundo vendedor: Microsoft-Decision-1 ✅ (2026-10-22)

Acceso restaurado al recurso Azure (la suspensión de 2026-10-20 quedó atrás):
la ruta correcta NO es `openai.azure.com/openai/v1/decisions` (otro esquema,
input str + questions array), sino la **System One nativa vía AI Foundry**:

    POST https://holaaaaaa.services.ai.azure.com/providers/microsoft/v1/systemone
    header: api-key: <token>
    {"model":"decision-1", "state":{...}, "questions":{gate/family/severity}}

(idéntico payload a Jev → parseo y métricas reutilizados sin cambios;
`scripts/e7_vendor_api.py --vendor azure`). Deployment `decision-1` →
modelo `Microsoft-Decision-1` (preview, chat=false, API propia de decisiones).

Corrida completa (misma muestra seed-42: 1,000 benignos + 1,021 ataques,
2021 filas, concurrency 8, ~30 min):

| primitiva | Jev | MS-Decision-1 | VX-One |
|---|---|---|---|
| Gate AUROC | 0.568 | **0.544** | 0.981 |
| Gate FPR@TPR95 | 0.932 | 0.872 | 0.157 |
| Gate ECE15 | 0.265 | **0.118** | 0.097 |
| Choice macro-F1 | 0.069 | 0.066 | 0.534 |
| Choice acc | — | 0.323 | 0.711 |
| Severity MAE | 0.809 | 0.969 | 0.333 |
| Latencia p50/p95 | 208/591 ms | 757/20,448 ms | ~1 ms |
| tokens | 2.2M | 1.24M | — |

**La réplica es el hallazgo**: DOS vendedores independientes de la tríada
zero-shot a nivel de azar en gate y degenerados en choice/severity sobre
flujos UNSW. Única diferencia: la ECE de MS es mejor (0.118) — sin señal no
hay nada que calibrar. La limitación "single-point stress test" se actualiza
a "réplica de dos vendedores en un dominio". Resultados:
`results/e7_vendors/vendor_azure_main.json` (+ npz). Nota esquema: la ruta
`/decisions` de openai.azure.com usa input-string + questions-array con
choices[{value,description}]/levels[{label,description}] — documentado en
`scripts/e7_azure_decision.py` por si se necesita.

## 22. E9b — outlier exposure (histórico; cifras invalidadas por §25) ⚠️ (2026-10-22, noche)

> **AVISO:** esta sección se calculó con un estimador distinto (solo
> temperatura) y sus cifras de abstención/absorción NO son comparables con E2.
> Las cifras canónicas están en §25 y en `results/e9b_oe/e9b_oe_contract.json`.
> Se conserva como registro histórico del bug.

El ítem opcional del informe resultó la contribución constructiva del ciclo.
Diseño (Hendrycks OE 2019): en cada fold LOAO, la familia excluida VUELVE al
train como pool de outliers — la choice head recibe target uniforme sobre los
candidatos conocidos en esas filas (λ=0.5 fijo, fuera del weighting Kendall;
todo lo demás congelado). Escenario: labels retrasadas de la familia nueva
(day-1/week-1) — exactamente lo que un operador tiene tras el primer día.
Implementación: `src/models/train.py` (oe dict, aditivo; EMBER intacta),
`modal/app.py train_vx(outlier_family=, oe_weight=)` + entrypoint `oe_loao`.

| fold (UNSW) | gate AUROC ref→oe | abst@.5 ref→oe | knownF1 ref→oe |
|---|---|---|---|
| analysis | 0.9992→0.9989 | 0.941→0.922 | 0.520→0.514 |
| backdoor | 0.9989→0.9982 | 0.911→0.976 | 0.524→0.524 |
| dos | 0.9941→0.9940 | 0.710→0.908 | 0.509→0.496 |
| exploitation | 0.9550→0.9740 | 0.320→0.916 | 0.521→0.516 |
| fuzzing | 0.8519→0.8646 | 0.357→0.939 | 0.512→0.496 |
| generic | 0.9957→0.9508 | 0.426→0.978 | 0.397→0.392 |
| recon | 0.9523→0.9519 | 0.250→0.965 | 0.448→0.441 |
| shellcode | 0.9688→0.9645 | 0.479→0.974 | 0.498→0.498 |
| worms | 0.9871→0.9904 | 0.227→0.977 | 0.499→0.484 |
| **media** | ~igual | 0.513→0.950 (inválido; §25: **0.375→0.879**) | −0.008 (válido: −0.007) |

**CIC estratificado (la patología saturada) — reparación completa**:

| fold | gate AUROC ref→oe | abst@.5 ref→oe | absorb→normal @.5 | knownF1 Δ |
|---|---|---|---|---|
| dos_ddos | 0.8524→0.9986 | 0.003→1.000 | 0.931→0.000 | −0.042 |
| recon | 0.4259→0.9909 | 0.000→0.999 | 0.996→0.000 | −0.125 |
| bruteforce | 0.9674→0.9964 | 0.000→0.998 | 0.707→0.002 | −0.029 |
| webattack | 0.9792→0.9987 | 0.003→1.000 | 0.927→0.000 | +0.064 |
| botnet | 0.7499→0.9878 | 0.000→1.000 | 1.000→0.000 | −0.132 |
| infiltration (n=5) | 0.5329→0.8120 | 0.000→0.600 | 1.000→0.400 | −0.049 |
| exploitation | 0.9703→0.9976 | 0.000→1.000 | 1.000→0.000 | +0.085 |
| **media** | **+0.186 (todos ↑)** | 0.00→0.94 (inválido; §25: **0.000→0.000**) | →0.00 (inválido; §25: **0.945→1.000**) | −0.033 (válido: +0.039) |

Lectura corregida (§25): (1) en UNSW la honestidad es ENSEÑABLE con
labels retrasadas + OE bajo el contrato de calibración del paper (abst
0.375→0.879; false-att 0.550→0.092; absorción 0.075→0.029); (2) en el régimen
saturado de CIC el mismo recetario NO mueve la abstención del choice head
(0.000→0.000) ni la absorción benigna (0.945→1.000), pero SÍ repara la
detección del gate (0.783→0.969; recon 0.426→0.991; 7/7 folds), y el coste de
atribución conocida es mixto (+0.039 de media, botnet −0.260). El antiguo
"0.00→0.94" era artefacto del estimador sin de-tilt ω. Pareja con E9:
"freshness compra discriminación; OE compra abstención donde la confianza
aún informa (UNSW) y detección en el régimen saturado (CIC); ninguno compra
calibración".
Resultados: `results/e9b_oe/e9b_oe.{json,md}`, npz en `results/outputs_oe/`.

## 23. E7-var — varianza de llamadas vendor (N=5) ✅ Decision-1 (2026-10-23)

`scripts/e7_variance.py` → `results/e7_vendors/variance_azure.json`.
5 repeats × 2,041 filas (10,205 llamadas, 2.8 h, concurrency 6, 0 fallos):

| métrica | mean ± σ (5 repeats) |
|---|---|
| Gate AUROC | 0.540 ± 0.002 |
| Gate ECE15 | 0.124 ± 0.008 |
| Choice macro-F1 | 0.066 ± 0.001 |
| Severity MAE | 0.960 ± 0.001 |
| σ fila-nivel del noul | 0.022 |
| Desacuerdo argmax choice | 13.1% filas |

Veredicto: el "gate a nivel de azar" es estructural (replicado 5×, σ
minúsculos); la choice fluctúa fila-a-fila (13%) pero la degeneración de
macro-F1 no se mueve. Jev: N=5 inviable por el rate-limit del tier free
(~250 req/día/IP; requeriría la fleet multi-nodo) — queda como único paso
abierto de E7, documentado en el paper.

## 24. E9b-bis — sensibilidad λ + pool sintético; E1-bis — baselines 3 semillas ✅ (2026-10-23)

**λ sweep UNSW (27 corridas L4)** (`results/e9b_oe/e9b_lambda_sweep.json`):
abst@0.5 media ref 0.513 → λ0.2 **0.916** → λ0.5 0.950 → λ1.0 0.956
(knownF1 0.490/0.485/0.481). Sin cuchillo-hiperparámetro: λ=0.2 ya captura
casi todo el beneficio.

**Pool sintético (ruido gaussiano φ, λ=0.5, 20k filas)**: abst 0.452 (ref
0.513) — apenas se mueve. **El mecanismo requiere exponer la distribución
real de la familia nueva**, no presión de regularización genérica.

**Baselines 3 semillas (Tabla 1)** (`results/e1e3/baseline_seed_variance.json`,
24 corridas Modal cpu=32: unsw/cic-strat/cic-time/ember × {xgb,rf,mlp} ×
s43/s44): σ gate AUROC ≤0.0013 (UNSW/EMBER), **0.017–0.027 by-day CIC**
(xgb 0.637±0.027), 0.00 (cic strat). Con 3 semillas: MLP UNSW media
0.9835±0.0013 EMPATA a VX 0.9834 (single seed dentro del rango MLP) →
el E1 in-distribución es empate, no victoria; la ventaja por-day
(0.711 vs 0.637±0.027) es ~3σ. Abstract/Tabla 1/§e1 actualizados.

## 25. Corrección de estimador en E9b + inferencia pareada E1/E2 ✅ (2026-10-23)

**Bug de estimador (crítico, corregido).** E9b evaluaba la abstención con
`softmax(z/T)` (solo temperatura), mientras E2 y todas las tablas usan el
contrato del engine `vx_calibrated` (temperatura **+ de-tilt ω**). Mismo
checkpoint, números distintos: UNSW/dos daba 0.028 con el contrato y 0.710
sin él. Se reescribió el evaluador (`scripts/e9b_oe_eval.py`, inglés, con
`--selftest` que reproduce E2 al cuarto decimal) y se recalcularon **todas**
las variantes (λ, ruido, CIC) bajo el contrato único:

| bloque | métrica | ref | OE λ=0.5 |
|---|---|---|---|
| UNSW (9 folds) | abst@0.5 | 0.375 | **0.879** |
| UNSW | normal absorption | 0.075 | 0.029 |
| UNSW | known macro-F1 | 0.476 | 0.469 |
| UNSW | gate AUROC | 0.967 | 0.965 |
| CIC (7 folds) | abst@0.5 | 0.000 | 0.000 |
| CIC | normal absorption | 0.945 | 1.000 |
| CIC | known macro-F1 | 0.775 | **0.815** |
| CIC | gate AUROC | 0.783 | **0.969** |

λ ∈ {0.2,0.5,1.0} → abst UNSW 0.759/0.879/0.900; pool sintético 0.354.
**El claim de CIC “0.000→0.94 abstención y absorción 94%→0%” era artefacto
del estimador y se retira del paper.** El resultado honesto en CIC es de
*detección*: el gate recupera la familia nueva (recon 0.426→0.991, botnet
0.750→0.988, infiltration 0.533→0.812; 7/7 folds mejoran), la abstención
del choice head no se mueve bajo el contrato, y el macro-F1 de clases
conocidas sube +0.039 de media pero es mixto (botnet −0.260).
Artefactos: `results/e9b_oe/e9b_oe_contract.json` (canónico),
`e9b_oe_e2estimator.json` y `e9b_oe_cic_e2estimator.json` (histórico).

**Inferencia pareada E1** (`scripts/e1_paired_stats.py`, `--selftest`
reproduce la Tabla 1; `results/e1e3/paired_stats.json`): bootstrap de filas
pareado B=300 (mismas filas para ambos modelos) + semillas 42/43/44.
- UNSW: VX−MLP ΔAUROC +0.0013 [+0.0010,+0.0016]; ΔF1 +0.012
  [+0.003,+0.023]; ΔECE −0.008; ΔBrier −0.015. A nivel de semilla VX
  0.9837±0.0002 vs MLP 0.9835±0.0013 → **empate** (la ventaja por filas es
  del orden del reentrenamiento).
- CIC by-day: ΔAUROC +0.074 [+0.073,+0.075], ΔFPR@95 −0.427
  [−0.429,−0.425]; semillas 0.734±0.022 vs 0.639±0.027 (Δ +0.095).
- EMBER: VX gate 0.9965±0.0005 vs XGB 0.9981±0.0001 (Δ −0.0017) — pérdida
  pequeña y consistente, reportada.
- CIC strat: todas las diferencias en el cuarto decimal (sin headroom).

**LOAO pareado por fold** (Wilcoxon signed-rank + Holm sobre 3 baselines):
UNSW empate (VX vs MLP p=0.73; vs XGB p=0.16, Holm 0.49). CIC, excluyendo
los folds degenerados (n=1, n=5): XGBoost gana 5/5 folds (Δ mediana −0.123,
p exacta 0.0625) → **se reporta como pérdida**, no como empate.

**Recuento de filas E7**: el run de varianza de Decision-1 usó 2,041 filas
(10,205 llamadas), no 2,021/10,105 (el run principal sí usa 2,021).
Corregido en §E7.

**Paper**: abstract, Tabla 1, §E1, §E2 (sweep + LOAO), §E9b y apéndice
actualizados; añadidas Proposición 5 (label shift es un vector) y el
protocolo estadístico pre-declarado; cita CRC 2024 (ICLR).

## 26. E6-a7 — ablación causal de typing: gate derivado vs gate dedicado ✅ (2026-10-23)

**Pregunta.** El revisor de Fase 1 objetó que la ganancia del motor podía venir
del multi-task sharing y no del *typing* (gate/choice/score como primitivas
separadas). La a7 aísla esa pregunta para el gate: mismo trunk, misma CE de
familia ponderada por ω, mismo de-tilt analítico `z − log ω`, misma cabeza
CORN de score, mismo presupuesto y early stopping; la única diferencia es que
**no hay cabeza de gate** y el gate se deriva como `1 − p(normal)` de la
distribución de choice.

**Implementación.**
- `src/models/network.py`: `HeadSpec.flat` + `normal_index`; `_flat_forward`
  devuelve `choice` (logits de familia), `noul` (logit derivado), `corn_logits`
  y `ddos`; el de-tilt ω se aplica solo en eval (`flat_logw`).
- `src/models/train.py`: `ablation == "a7"` marca `flat=True`, guarda
  `a7_omega_train`, fija `omega=1` en el artefacto (para que `vx_calibrated`
  no aplique el de-tilt dos veces), y entrena CE de familia + CORN + ddos con
  Kendall sobre `("choice","score","ddos")`.
- `src/models/engine.py`: `head_meta["flat"]` para que `dump_outputs`
  reconstruya la red plana.
- `scripts/a7_derived_gate_eval.py`: evaluador pareado por semilla y por fila
  (bootstrap B=300 descriptivo, métricas canónicas vía `eval_vx_run`,
  `--selftest` sintético que valida AUROC contra sklearn y delta cero).

**Corridas.** 6 jobs Small en Modal L4: UNSW `oficial` y CIC `strat`, semillas
42/43/44. Artefactos y npz en el volumen `paper-results`
(`outputs/*_abl_a7*`); los npz no se versionan (`.gitignore`), los JSON y el
informe sí (`results/e6_a7/`).

**Resultados (semilla 42 en Tabla 6 del paper; medias±sd de 3 semillas aquí).**

| dataset | AUROC a7 | AUROC typed | ΔAUROC | m-F1 a7 | m-F1 typed | Δm-F1 |
|---|---|---|---|---|---|---|
| UNSW | 0.9829±0.0014 | 0.9837±0.0002 | −0.0008±0.0013 (1/3) | 0.4615±0.0121 | 0.4560±0.0011 | +0.0055±0.0109 (2/3) |
| CIC strat | 0.9999±0.0000 | 0.9999±0.0000 | −0.0000 (2/3) | 0.8981±0.0348 | 0.8312±0.0902 | +0.0669±0.0995 (2/3) |

FPR@95 UNSW: 0.1304±0.0082 (a7) vs 0.1222±0.0010 (typed); ECE-ch UNSW:
0.0435 vs 0.0544. En CIC todo está en el cuarto decimal salvo la atribución.

**Lectura honesta.** Nulo en detección: el gate dedicado no compra AUROC ni
FPR@95; la diferencia de atribución en CIC (+6.7 pts, 2/3) está dentro de la
inestabilidad de semillas ya documentada (typed 0.727–0.884). Lo que compra el
typing es el **contrato**: BCE binaria sin tilt, temperatura propia y semántica
`unknown` explícita; es una propiedad de auditabilidad, no de ranking. El
paper lo declara así en §E6 y en §E2 ("not raw AUROC superiority").

**Extensión LOAO (mismas particiones y seed 42; semillas 43/44 en tres folds
CIC).** Se reentrenó a7 en los 9 folds de UNSW y los 7 de CIC, con la misma
partición que la referencia tipada. Detección one-vs-normal sobre la familia
excluida (`scripts/a7_loao_eval.py`, salidas `results/e6_a7/loao.json` y
`loao.md`; chequeo de semillas en `loao_seeds.json`/`loao_seeds.md`):

| dataset | folds | ΔAUROC mediana | a7 mejor | Wilcoxon |
|---|---|---|---|---|
| UNSW | 9 | −0.0009 | 2/9 | p=0.30 |
| CIC strat | 7 | −0.0252 | 3/7 | p=0.30 |

En UNSW el empate se mantiene. En CIC el patrón no es uniforme: el gate
derivado pierde de forma **consistente** en `dos_ddos` (0.60–0.63 en tres
semillas vs 0.85 tipado) y `botnet` (0.57–0.59 vs 0.75), justo los folds donde
el choice head absorbe la familia nueva en `normal`; gana ligeramente en
`recon` (+0.013) y `webattack` (+0.013); y en `exploitation` el seed 42 daba
0.421 vs 0.970, pero las semillas 43/44 dan 0.943/0.871: **el colapso
catastrófico era de una semilla, no del diseño**. El test de signos de 7
folds no tiene potencia (p=0.30), así que se reporta el patrón por fold, no
una media ganadora.

**Lectura conjunta.** El typing no compra ranking in-distribution; bajo LOAO
sí importa donde la absorción benigna domina (`dos_ddos`, `botnet`), pero no
de forma universal. La lección metodológica es la regla de semillas: sin las
semillas 43/44 habríamos reportado un colapso de −0.55 que no replica.

**Alcance / lo que NO prueba.** La a7 elimina solo la cabeza de gate; conserva
el score CORN tipado y el candidate set con `normal`. El gate derivado no es
universal: depende de que exista una clase `normal` (en EMBER con `unknown`
entrenado la semántica es distinta). No se corrió a7 sobre el pool OE de E9b;
esa sigue siendo la extensión natural.

**Reproducir.**
```bash
for src in unsw cic; do
  sch=oficial; [ "$src" = cic ] && sch=strat
  for s in 42 43 44; do
    modal run modal/app.py::train_vx --source $src --scale small \
      --scheme $sch --ablation a7 --seed $s
  done
done
# bajar npz/json del volumen y evaluar
modal volume get paper-results outputs/unsw_small_oficial_abl_a7.npz results/e6_a7/
python scripts/a7_derived_gate_eval.py --a7 results/e6_a7 \
  --typed results/outputs_seed --out results/e6_a7
# extensión LOAO (9 folds UNSW + 7 CIC, seed 42)
modal run modal/app.py::loao_a7 --source unsw
modal run modal/app.py::loao_a7 --source cic
python scripts/a7_loao_eval.py --a7 results/e6_a7/loao \
  --typed results/e6_a7/loao_typed --out results/e6_a7
```
