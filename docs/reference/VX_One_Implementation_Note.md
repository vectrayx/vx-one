# VX-One — implementación del modelo (F2) — 2026-10-17

Estado: **COMPLETO y verificado con 60/60 tests** (`pytest tests/ -q`, incluye
integración con parquets reales UNSW/CIC y smoke E2E CPU).

## Módulos (`src/models/`)

| módulo | contenido | verificación clave |
|---|---|---|
| `feature_stage.py` | D6: mediana de no-finitos + flag fila, signed log1p, escala robusta train (mediana/IQR), clip ±8 | tests de matemática + roundtrip |
| `vocab.py` | TextVocab con slots MISS=0/UNK=1; UNSW proto 133→135 slots | sizes §2.3 |
| `network.py` | VXOneNet §2: proj(d+1→d0) + emb + tronco L×(Linear→LN→GELU→Dropout.1, residual desde bloque 2) + cabezas; CORN init b=(1,0,−1) | **tabla §2.3 exacta**: 9 configs + CIC split_time (87,258) |
| `losses.py` | ω sqrt media-1 (D4), BCE sin pesos (Prop. 1), CE ponderada, CORN condicional (D3), Kendall homeostático con máscaras (D5) | **tablas ω §3.5 exactas** (UNSW/CIC/ddos/EMBER), pmf CORN válida, de-tilt recupera π (Prop. 2) |
| `calibration.py` | §5: grid log-uniforme 200 en [0.05,20] + sección dorada (tol 1e-3), NLL; de-tilt ANTES de T | recupera T conocido (1.7/0.5/1.23), determinista, CORN T conjunta |
| `data.py` | RunData + splits + D12 candidate set + ω por corrida + LOAO + streaming EMBER (perm shards seed 42+epoch, drop_last, prefetch thread) + φ submuestra 60,024 | conteos/ω = §3.5 sobre parquets reales; splits 149,039/26,302/82,332; CIC strat 1,981,520; ddos subset 642,862 (376,380/89,598/169,034/7,850) |
| `engine.py` | §8.1: Decision/ChoiceResult/ScoreResult tipados, de-tilt→T→softmax, abstención τ_abs, gates D9, checkpoint §8.2 | roundtrip exacto vs forward directo, invariantes probs=1, abstención, gates |
| `train.py` | §6.3: AdamW (wd 1e-2 solo matrices 2D Linear), warmup 5%+coseno, clip 1.0, AMP bf16/fp16/off, patience 4 sobre M_val, calibración final, artefacto | smoke E2E: parquet sintético → train 2 epochs → artefacto → engine |

## Decisiones de implementación (nuevas, documentadas)

1. **Sentinela y_fam=−1** para filas de val/test con familia fuera del
   candidate set (LOAO test completo §6.7; recon/botnet en val/test de CIC
   split_time). Los nombres nativos viajan en `RunData.val_fam_names /
   test_fam_names` para E2. Choice (métrica M_val y calibración T) excluye
   esas filas; Noul/Score las usan completas.
2. **ddos subset** = (day ∈ {Wednesday, Friday-DDoS}) ∧ (label_full ∈
   DDOS_FULL) = 642,862 filas train (de las 918,437 del día; las patators/
   bot/web de esos días no entrenan el head). `y_ddos` alineado a longitud
   completa. Mapeo por **label_full** (etiqueta cruda), no label_family
   colapsada — segunda vez que este bug aparece, el test lo fija ahora.
3. **τ_abs serializado** en head_meta con default TAU_ABS=0.5; overridable
   por run_meta (política del controlador, Q3 del advisory lo re-justifica).
4. **Alias de registro**: `unsw/cic/ember` → `unsw_nb15_v1/...` centralizado
   en `data._registry()` y `engine.SOURCE_ALIAS`.

## Cómo entrenar

```python
from src.models import build_run, train_run
run = build_run("unsw", "nano", "oficial")     # o "cic"+"strat"/"time", "ember"
res = train_run(run, "results/artifact.pt", device="cuda:0")  # AMP auto
from src.models import VXOneEngine
eng = VXOneEngine("results/artifact.pt")
dec = eng.predict_batch(x_num, x_txt)           # → list[Decision]
```

## Pendiente (hacia el paper)

- Baselines (`src/baselines/`): XGBoost, RF, MLP, DistilBERT, Kitsune —
  mismo protocolo de calibración (anti-strawman §5.3). xgboost por instalar.
- Modal: app + 10 corridas principales + folds LOAO (UNSW 9×Small,
  CIC 7×Small); E1-E7 en `src/eval/`.
- Advisory Opus entregado en `docs/VX_One_F2_Advisory_Opus.md` (Q1-Q3 +
  7 cambios propuestos a la spec) — fusionar al escribir el paper.

## Baselines tabulares + evaluación E1/E3 (2026-10-17, tarde)

- `src/baselines/tabular.py`: XGBoost (500×d8 hist), RF (300, min_leaf 1),
  MLP (2×256 BN→ReLU, multi-head) — **mismos inputs** (u_r + one-hot textual
  con el vocab de train), **mismos ω** por sample_weight, **misma
  calibración** (1 T/cabeza, mismo buscador NLL, de-tilt simétrico) y
  **mismo M_val** de early stop. Score nominal softmax (vs CORN).
- `src/baselines/run_tabular.py`: CLI por dataset/scheme/exclude-family;
  guarda probs crudas npz + temperaturas json (E3 reporta raw y cal).
- `src/eval/dump_outputs.py`: logits crudos val/test de un artefacto → npz
  + meta json (T, ω, τ, candidatos, modo). Las métricas E1-E3 se computan
  LOCAL desde npz (los jobs GPU solo entrenan y dumpean).
- `src/eval/metrics.py`: FPR@95TPR (umbral = cuantil 1−tpr de positivos),
  AUROC/PR-AUC, macro-F1 con labels explícitos, ECE/MCE 15 bins + NLL +
  Brier raw/cal, reliability data, abstención (cobertura/precisión).
  ⚠️ ECE binario usa conf = max(p, 1−p) (p directo infla el gap en
  negativos correctos de baja p).
- `src/eval/run_e1e3.py`: fusiona VX (npz) + baselines (npz) →
  results/e1e3/{e1e3.json,e1e3.md}.

**Checkpointing de recuperación (pedido del usuario)**: `train_run` acepta
`ckpt_path`/`ckpt_every_steps`/`resume`; escribe atómicamente (tmp→replace)
cada N pasos y al fin de cada epoch (net+opt+AdamW states+sched+scaler+
contadores early-stop+rng); `resume=True` replaya el epoch en curso con
sched fresco (alineación exacta del coseno) — probado con kill -9 real:
RESUME retomó en el epoch siguiente y el artefacto final salió idéntico
en formato; el ckpt muere cuando la corrida termina bien. Modal lo usa
(volumen, resume automático al relanzar).

## Primeros números (test, temperatura calibrada §5)

| modelo | AUROC | FPR@95TPR | macro-F1 | ECE noul | ECE choice |
|---|---|---|---|---|---|
| VX-One UNSW Nano | 0.9838 | **0.1085** | 0.4332 | 0.0564 | 0.0625 |
| VX-One UNSW Small | 0.9834 | 0.1211 | 0.4571 | **0.0427** | **0.0496** |
| VX-One UNSW Base | **0.9842** | 0.1191 | 0.4543 | 0.0504 | 0.0609 |
| XGBoost | 0.9818 | 0.1369 | **0.4968** | 0.0641 | 0.0647 |
| RF | 0.9792 | 0.1399 | 0.4378 | 0.0504 | 0.0742 |
| MLP | 0.9821 | 0.1279 | 0.4445 | 0.0571 | 0.0578 |

Lectura honesta: VX-One gana detección (AUROC/FPR@95TPR) y calibración;
XGBoost gana macro-F1 de atribución en UNSW (tabular chico — anti-strawman
funcionó; se reporta tal cual; CIC/EMBER y E2 zero-day son el terreno
restante). Nano≈Base en UNSW (40 features — la escala no importa ahí).

## Infra Modal

- `modal/app.py`: imagen py3.12 = versiones locales (torch 2.14.1+cu130,
  pyarrow 26, sklearn 1.9.1, xgboost 3.4.1); `src/` montado; volúmenes
  paper-data (→ /repo/data, root `processed/`) y paper-results (→ /vol).
- Funciones: `train_vx` (L4, ckpt al volumen + resume auto + dump npz),
  `dump`, `baseline` (16 CPU), `probe`; entrypoints `mains` (9 corridas en
  paralelo) y `loao` (folds en paralelo).
- Bugs pillados en el primer despliegue: numpy 2.5.4 requiere py≥3.12;
  `modal volume put` con directorio anida (subir archivo por archivo con
  REMOTE_PATH acabado en `/`); import `src` ANTES de `_boot()`; ω y labels
  en CPU vs logits CUDA (mover a device en `_batch_losses` y generadores);
  `/vol/artifacts` inexistente (mkdir en `_boot()`); `ddos_candidates`
  faltante en run_meta de `save_artifact` (latente: el smoke era UNSW).

## Modal: corridas CIC/EMBER + E2 + E4 (2026-10-17 noche)

- **Lección de orquestación**: `.remote()` en Modal 2.x BLOQUEA — el
  launcher `mains` corría en secuencia y al crashear cic_nano (KeyError
  ddos_candidates) murió entero. Fan-out real = `.spawn()` (retorna
  FunctionCall con `.get()`); arreglado en mains y loao. Corridas largas =
  mejor una invocación `modal run` separada por corrida (apps
  independientes, un fallo no cancela hermanos).
- **CIC strat completo** (nano/small/base, test): AUROC 0.9999, FPR@95
  0.0003-0.0004, macro-F1 choice 0.883-0.903, ddos F1 0.998, score M
  0.98-0.997, ECE ≤0.001 — saturado como la literatura (los baselines
  dirán lo mismo; benchmark de sanidad, no de ventaja).
- **CIC time small (drift)**: AUROC 0.7112, FPR@95TPR 0.5731, ECE noul
  0.4037 — el drift temporal colapsa detección Y calibración (val
  estacionaria miente); early stop en epoch 5 con best epoch 0. K=6
  (botnet/recon solo aparecen en val/test → sentinela y_fam=−1).
- **E2 LOAO UNSW (9/9 folds Small)**: Noul detecta la familia nunca vista
  en 9/9 (AUROC 0.852 fuzzing → 0.999 analysis; exploitation FPR@95 0.42
  el peor). Choice: abstención 3-97% por familia (dos atribuye 97% —
  pero Noul lo detecta 0.994; fuzzing/generic/shellcode/recon ~28-50%
  abstienen). macro-F1 de conocidas intacto (0.38-0.56). →
  results/e2/e2.json.
- **E4 latencia (local, 1 hilo CPU, batch 1, n=3000)**: Nano p50 675 µs /
  p99 773 µs; Small p50 955 µs / p99 1022 µs; Base p50 1652 µs / p99
  2082 µs — presupuesto §7 (<10 ms) con margen ≥5×. results/e4/.
- **Fix LOAO×ddos (data.py)**: el subset ddos = BENIGN (376K) + dos_ddos
  (266K); en el fold dos_ddos quedan solo benignos → clases de ataque en
  0. Ahora si `ddos_counts` tiene alguna clase en 0 → esa corrida va sin
  head ddos (E2 la evalúa por noul/choice). Además `day/full` ahora se
  podan con `keep` en LOAO (bug de desalineo 1,981,520 vs 1,715,038).
- **Fix dump EMBER**: el bucle de streaming de `dump_run` llamaba
  `net(...)` sin `torch.no_grad()` (el decorador solo cubría `_forward`)
  → "Can't call numpy() on Tensor that requires grad" tras entrenar bien:
  artefactos safe, re-dump con el entrypoint `dump`.

## Ronda baselines + E6 (2026-10-18)

### Bugs encontrados y arreglados
- **`.remote()` explícito en entrypoints locales (Modal 1.6/2.x)**: llamar
  `train_vx(...)` a pelo en `@app.local_entrypoint` → TypeError 'Function'
  object is not callable. Las 13 ablations de la primera ronda murieron así.
- **gather silencioso (calibración ddos baselines)**: `torch.gather` acepta
  un índice MÁS CORTO que el input en las dims no-indexadas → el NLL ddos
  juntaba p_ddos full-length (424k) con y_ddos pre-mascarado (137k) sin
  error, alineando por posición → T_ddos clavado en 0.501 (borde del grid)
  IDÉNTICO en xgboost/rf. Fix: val_dict recorta p_ddos a las filas de la
  máscara antes de calibrar. T_ddos ahora sanea (1.12/0.70/1.03).
- **Doble máscara MLP ddos**: `va["y_ddos"]` ya viene recortada por
  `_collect_split`; re-indexarla con `va["ddos_mask"]` → IndexError. Fix.
- **Compactación de clases (scheme time)**: el crash real del run viejo era
  la cabeza SCORE (`y_sev` de train = {0,2,3}, severidad 1 sin filas en el
  train temporal) — XGBoost/RF exigen targets contiguos. Fix genérico:
  `_compact` (train), `_scatter` (probs a las columnas originales),
  `_remap_val` (val/test, clases ausentes → −1 fuera de f1/CE/calibración).
  Choice nunca lo necesitó: train tiene las 6 clases completas
  ({0:1.71M, 1:252k, 2:13k, 3:2.1k, 4:23, 5:11}).
- **Dump de artefactos A1**: el ckpt sin vocab ({}) vs run con vocab →
  assert. Fix: si el ckpt no trae vocab, correr sin txt (red sin embeddings).
- **EMBER en baselines**: RunData no materializa splits (streaming) →
  `_collect_ember` materializa u+labels desde los shards vía
  `eval_batches("train")` (nueva key; sin shuffle/drop_last). Sin one-hot
  textual: los inputs del motor son u nada más — comparación justa.
  `train_mlp` ahora enmascara y_fam=−1 en la CE de Choice (red de
  seguridad; en EMBER estándar las familias raras van al candidato unknown).

### Baselines CIC (results/baselines/, 32 CPU)
- **cic_strat (saturado)**: XGBoost gana TODO: AUROC 1.0000, FPR@95 0.0002,
  macro-F1 0.9416 (VX: 0.9999/0.0003/0.8826; base 0.9026). RF colapsa en
  macro-F1 (0.698) con el desbalance extremo; MLP 0.859. Anti-strawman
  confirmado: en saturado el árbol es mejor modelo y el paper lo dice.
- **cic_time (drift temporal)**: TODOS los baselines colapsan en detección
  (AUROC 0.596 mlp / 0.631 rf / 0.637 xgb vs VX 0.7112) — VX conserva ~7
  pts de ventaja. PERO xgboost macro-F1 0.8336 vs VX 0.4518: la atribución
  de VX quedó sub-entrenada (early stop epoch 5, best epoch 0) mientras el
  árbol sin early-stop aprende dos_ddos (252k filas de train) bien. Lectura
  honesta: bajo drift, detección semi-dura mejor en VX; atribución mejor en
  el árbol.

### E6 ablations (Small, misma receta; results/e6/)
- **a1 sin rama textual (UNSW)**: AUROC 0.9834→0.9716, macro-F1
  0.4571→0.4079 (−4.9 pts). La rama proto/service/state vale ~5 pts.
- **a3 sin calibración (T=1)**: métricas de ranking idénticas (T no cambia
  argmax); ECE choice UNSW 0.0496→0.0980 (2× peor) — la calibración aporta
  en probabilidad, no en ranking, como dice la spec.
- **a4 ω≡1**: UNSW ≈ igual (0.4545 vs 0.4571); **CIC macro-F1 0.8826→0.9313
  (+4.9 pts SIN tilt)** — hallazgo incómodo y honesto: el tilt ω perjudica
  la atribución en CIC (el de-tilt empuja abajo las clases raras que la CE
  ponderada ya había subido). Discutir en el paper; candidato a revisión
  del diseño ω para datasets saturados.
- **a5 unknown/abstención off (UNSW)**: métricas idénticas a la referencia
  (en el test completo nadie abstiene con τ=0.5; la abstención solo vive en
  los folds LOAO de E2).
- **a6 dropout 0**: CIC macro-F1 0.8826→0.7360 (−14.7 pts, overfit sin
  regularización); UNSW −0.6 pts + ECE noul 0.043→0.060. El dropout es
  crítico en CIC.
- EMBER a3-a6 en vuelo.

### E2 EMBER drift (Small; results/e2/e2.json)
- Noul global: AUROC 0.9969, FPR@95TPR 0.0119 — la detección resiste 4
  meses de drift. Per-family Choice: berbew 0.998 (crece en test), cosmu/
  upatre/sfone 0.94-0.97; **wacatac recall 0.129 con 85% → unknown**
  (mutó), **grandoreiro 0.221 con 78% → unknown** (casi desapareció del
  train), sinowal 0.444 (n=18). El modo trained_unknown canaliza el drift
  a `unknown` en vez de atribuciones falsas — la promesa del diseño.

### E2 CIC LOAO (7 folds)
- Detección mucho más dura que UNSW: recon 0.4259 (BAJO azar),
  infiltration 0.5329, botnet 0.7499 vs bruteforce 0.9674 / webattack
  0.9792 / exploitation 0.9703 / dos_ddos 0.8524.
- Choice NUNCA abstiene en la familia excluida (abst=0.00, attr=1.00): la
  calibración saturada de CIC hace que τ_abs=0.5 nunca muerda. Asimetría
  con UNSW (abstiene 3-97%) — discutir: τ fijo vs calibración por dataset.

## Ronda E6 completa + Kitsune + A2 (2026-10-18, tarde)

### E6 14/14 (todos los ablations)

- **a5-EMBER (unknown off, 10-way)**: M_val 0.9618, best_epoch 1. Dos
  fallos previos documentados: (i) `run.heads` es frozen dataclass →
  `dataclasses.replace(run.heads, n_candidates=10)`; (ii) el re-launch
  chocabó con el checkpoint del intento buggy (red 11-wide): el resume
  cargaba un state_dict incompatible → borrar
  `ckpts/ember_small_temporal_abl_a5.ckpt.pt` del volumen y relanzar limpio.
- **a6-EMBER (dropout 0)**: macro-F1 0.7821 vs ref 0.7549 (**+2.7 pts SIN
  dropout**) — opuesto a CIC (−14.7). En EMBER el dropout no ayuda (la
  clase mayoritaria domina y el early-stop regula). Mixto, se reporta tal cual.
- **a2 sin multi-task (UNSW, 3 modelos single-head)**: implementado como
  `a2_noul`/`a2_choice`/`a2_score` (una sola cabeza con pérdida y M_val;
  las demás sin gradiente; anchura sin ajuste porque las cabezas aportan
  <1% de params). `merge_a2_outputs` (dump_outputs.py) combina los tres
  dumps en el npz estándar de la corrida. Resultado vs ref:
  AUROC 0.9784 vs 0.9834 (−0.5), FPR@95 0.1446 vs 0.1211, macro-F1 0.4598
  vs 0.4571 (+0.3). **La ganancia del multi-task está en Noul (detección);
  Choice no la necesita.** Los M_val de los single-head: noul 0.9918,
  choice 0.6065 (early-stop solo sobre F1 es ruidoso: bad=4 → stop ep 21),
  score 0.9127.

### Kitsune (baseline no-supervisado obligatorio)

`src/baselines/kitsune.py`. Adaptación a flujos documentada: ensemble de 9
AEs embudo con feature bagging por correlación (greedy del original),
entrenado SOLO en filas benignas de train, score = RMSE máximo del
ensemble, Platt afín σ((s−b)/T) fitted en val (LogisticRegression sobre s:
w=1/T, b=−w₀/T), npz guarda la salida de Platt con T_json=1.

- **Hallazgo de diseño**: con el FeatureStage compartido (φ fitted en TODO
  train, 57% ataques en UNSW) la señal queda INVERTIDA ya en train
  (AUROC(s,y)=0.38: los ataques se reconstruyen mejor). Es el sustrato el
  hostil para benign-only, no el modelo. Fix: Kitsune re-fitea su PROPIO φ
  (misma mediana+IQR+log1p) solo en sus filas benignas — lo que exigiría un
  despliegue no-supervisado real.
- **Resultados** (test): UNSW oficial AUROC 0.6086 / FPR@95TPR 0.5617 /
  ECE 0.2179; CIC strat AUROC 0.6227 / FPR@95 0.6910 / ECE 0.0373. Señal
  correcta (Platt T>0, s mal > s ben) pero débil: los AEs sobre flujos
  estandarizados no separan. **Punto del paper**: Platt calibra la
  probabilidad pero no crea señal — bien calibrado y no discriminativo a
  la vez (exactamente lo que la tipología Noul vs score puro separa).
- E1/E3 regenerado: 26 VX + 11 baselines.

### DistilBERT: decisión de alcance

Los datasets no tienen texto libre: EMBER solo x_num (φ de 2568), UNSW
tiene 3 tokens categóricos (proto/service/state), CIC 1 (protocol).
Fine-tunear DistilBERT sobre eso es un strawman al revés; la spec (§2.1
nota) ya lo define como baseline E7/API — queda ahí, documentado en el
paper como comparador de encoders a nivel API (pendiente credenciales).

### Evaluator y tolerancias

- `run_e1e3` ahora tolera npz sin p_choice (Kitsune: solo detección — la
  columna macro-F1 sale "—" y así se reporta).
- `run_tabular`: el marcador de "no materializado" en EMBER es
  `torch.empty(0)`, NO None — el check correcto es
  `run.ember_train_shards is not None` (el check `run.train is None` no
  disparaba nunca → AxisError en _collect_split).
- Suite completa 60/60 tras los cambios del día.

## E5 v0.1: pcaps sintéticos + Suricata + cascada two-sample (2026-10-18, noche)

### Setup Suricata (Modal)
- `modal/app.py::suricata_e5`: imagen propia = base + capa apt con
  backports de Debian (bookworm) → **Suricata 7.0.10** (fallback 6.x si
  backports no lo tiene; la versión real queda en el output). Gotcha de
  Modal 1.6: `add_local_dir` requiere `copy=True` para poder apilar
  `run_commands` encima (si no, la build falla).
- Reglas: ET Open frescas por corrida (rules.emergingthreats.net/open/
  suricata/emerging.rules.tar.gz), todos los *.rules del tarball
  concatenados en un suricata.rules y cargadas con `-S` → **52,674 reglas**.
- `--set flow.memcap=1gb --set flow.prealloc=600000` (el default de 128MB
  se queda corto con ~458K flujos/día).
- Output por día: `eve_{day}.json.gz` en `paper-results:/vol/e5/`.

### Hallazgo: el enhanced NO es copia fila-a-fila de MLCVE
Hipótesis inicial: alinear CICIDS_Flow.parquet (enhanced, 2,827,677 filas,
5-tupla) con processed/cicids2017.parquet (MLCVE, 2,830,743) por hash de
las 76 features comunes con two-pointer. FALSADA empíricamente: las filas
0-1 coinciden exactas, pero la fila 2 difiere en 12/76 features (maxdiff
8e6 µs en Flow Duration) — el enhanced **recomputa** las estadísticas de
flujo (misma secuencia de captura, stats corregidas; el defecto conocido
de CICFlowMeter en MLCVE). La fila 0 igual no era señal de copia, sino de
flujos triviales que recomputados salen igual. Consecuencia: no existe
join fila-a-fila enhanced↔MLCVE (el script de alineación ligó 479/2.8M y
se descartó).

### Diseño cascada: two-sample por día (documentado en cascade.py)
- **Fase 1 (per-row real)**: flujos del día del enhanced + alertas eve
  casadas por 5-tupla + ventana [t0−2s, t0+dur+5s] → cobertura F1 por
  familia, composición de la zona gris (lo que F1 NO resuelve), latencia
  de 1ª alerta.
- **Fase 2 (estimación insesgada)**: filas TEST del día (split_strat +
  day_idx del processed) con logits del npz → recall por familia y FPR de
  VX-One sobre ese día.
- **Combinado (estimación)**: supuesto documentado — P(F2 flag | familia)
  es independiente de "¿F1 lo detectó?" dado el tipo de flujo. Exacto para
  familias invisibles a F1 sin payload (web: la zona gris ES la clase
  completa); optimista donde F1 falla justo los sigilosos. El paper lo
  reporta como estimación con el supuesto explícito.
- Fallback F2 latency: E4 unsw small p50 = 955 µs (batch=1, CPU local).

### Bugs del día
- `run_tabular.py:267`: specs guarda 5-tuplas (y,w,cols,n,rows) desde que
  se añadió `ch_rows` para EMBER, pero el dict-comp desempaquetaba 4 →
  "too many values to unpack"; y en el primer fix renombré a `_n` sin
  actualizar el valor → NameError. EMBER baselines re-lanzado tras ambos.
- `flow2pcap.py` escribía paquetes IP crudos con linktype=1 (Ethernet):
  Suricata leía 14 bytes fantasma y descartaba TODO (0 alertas, 0 flujos
  en la 1ª corrida; la advertencia de scapy "Inconsistent linktypes" era
  la señal). Fix: envolver en `Ether()`.

### Depuración Suricata 0-flujos/0-alertas (lecciones)
1. **IP.len mentiroso**: fijaba `pk[IP].len = ln` (tamaño de FRAME de las
   stats) → header IP declaraba 14 bytes de más que el frame real →
   Suricata descartaba TODO como truncado ("read 1478405 packets" pero 0
   flujos). Fix: no fijar ip.len a mano (scapy lo calcula) + `patch_iplen`
   para reparar in-place los pcaps ya generados (ip.len = caplen−14,
   2 bytes por paquete, sin regenerar).
2. **TCP checksum=0** (scapy no lo calcula on-write): `--set
   checksum-checks=no` — pcap sintético, documentado.
3. **ET sin payload ≈ sin señal de ataque**: con las 52,674 reglas ET
   Open el jueves disparó SOLO 2 alertas (CDN Akamai) — las reglas de
   Hulk/DDoS/PortScan son de contenido HTTP o de patrones de payload.
   La cota inferior se mantiene para contenido; se añadieron **8 reglas
   estructurales propias** (VXSTRUCT sid 9700001-8: syn-flood, portscan
   por umbral, probes 445/22/80, udp/icmp flood — cabecera+threshold
   track by_src, sin payload) para que Fase 1 tenga señal estructural
   real sobre la reconstrucción. Fase 1 = ET Open + 8 VXSTRUCT,
   documentado en el paper.

### Resultados E5 (5 días, cascade_{day}.json + results/e5/e5.md)
| día | flujos | ataque | F1 acc | F2 acc | comb acc | comb P | comb R |
|---|---|---|---|---|---|---|---|
| 03 (lun) | 529,450 | 0 | 0.9989 | 0.9996 | 0.9985 | — | — |
| 04 (mar) | 445,641 | 13,832 | 0.9681 | 0.9994 | 0.9985 | 0.9579 | 0.9961 |
| 05 (mié) | 691,390 | 251,722 | 0.6359 | 0.9986 | 0.9979 | 0.9971 | 0.9971 |
| 06 (jue) | 458,483 | 2,216 | 0.9942 | 0.9971 | 0.9961 | 0.5521 | 0.9779 |
| 07 (vie) | 702,713 | 288,785 | 0.5887 | 0.9979 | 0.9974 | 0.9980 | 0.9958 |

Lectura del paper: F1 (sin payload → cota inferior) resuelve solo lo
estructural (probes de puertos, floods); la zona gris queda en ~99.9% del
día y F2 la recupera: el accuracy combinado (0.996-0.999) supera a F1-solo
en los días de ataque masivo (mié 0.64→0.998, vie 0.59→0.997) con P/R
≥0.95. En jueves (web attacks, F1 invisible sin payload) el combinado P
baja a 0.55: los FPs de F2 dominan porque hay 456K benignos × FPR 0.28%.
La latencia E2E: F1 alerta in-situ (mediana 0.0s sobre la reconstrucción,
los paquetes están en el mismo ts del flujo) + F2 p50 955 µs (E4).
