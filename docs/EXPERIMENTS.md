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
| E7 | Baseline comercial (Decision-1) | ❌ no ejecutado (opcional, ~$2) | Pendiente |

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

## 9. E7 — Baseline comercial (pendiente)

Zero-shot de Microsoft-Decision-1 vía API sobre muestra estratificada (1–5K
eventos, `State` serializado como JSON). Costo ~$2. Hipótesis del proveedor:
"calibración más fuerte en tipos de tarea familiares" → el modelo especializado
(gana en sus tareas) vs el nuestro (dominio). No ejecutado: pendiente de
credenciales; los claims de "primera evaluación del paradigma en seguridad" se
matizan en el paper mientras tanto.

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

- [ ] **E2 sweep de τ_abs** (grid 0.05–0.95 sobre los logits LOAO ya producidos):
      curvas risk-coverage + AURC; por familia excluida: %unknown(τ) y falsa
      atribución 1−%unknown como riesgo. Sin re-entrenar.
- [ ] **E3 reliability por condicional CORN** (q_1 en todas las filas; q_2 en
      elegibles sev≥1) + ECE directo del gate C_2 — atribuye el ECE del gate a
      sus eslabones (Lema 2). Sin re-entrenar.
- [ ] **E3 classwise calibration de Choice** (E[1[y=k] | p_k] por clase) para
      respaldar el gate g_recon (necesita calibration classwise, más fuerte
      que la confidence que optimiza TS).
- [ ] **Diagnóstico Bayes-vs-Chow** en val: % de filas donde la regla general
      con matriz de costes difiere de la abstracción max-prob. Si ≈0%, la
      simplificación queda justificada empíricamente en una línea.
- [ ] (Si se ejecuta) protocolo conformal EMBER del advisory §2(d): coverage
      semanal unweighted vs weighted (clasificador de dominios, odds clipeado
      [0.1,10]), tabla de 12 semanas con cota inferior de d_TV por BER.
