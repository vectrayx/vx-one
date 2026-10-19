# SCHEMA — Esquema canónico de features y mapeos (F1)

**Versión:** v0-draft (2026-10-16) — se congela v1.0 DESPUÉS de verificar headers reales de los CSVs descargados. Toda columna marcada ⚠️ requiere verificación contra el archivo real antes de congelar.
**Sirve a:** `VectrayX_One_Model_Spec_F0.md` §1 (objeto `State`) y §3.1 (configs de `Choice`).

---

## 0. Decisiones de schema

| # | Decisión | Racional |
|---|---|---|
| S1 | **CIC-IDS2017: fuente primaria = `GeneratedLabelledFlows`** (84 cols; incluye Flow ID, IPs, puertos, Protocol, Timestamp) en vez de `MachineLearningCSV` (78 cols) | Splits por tiempo (E1-robustez), features de ventana v0.2, correlación flujo↔PCAP (E5), y campos textuales (`Protocol`, Flow ID) para la rama textual |
| S2 | `MachineLearningCSV` se mantiene como variante para comparabilidad con literatura | La mayoría de papers publicados usan las 78 features |
| S3 | UNSW-NB15: split oficial (training/testing-set) | Comparabilidad con toda la literatura del dataset |
| S4 | Features asimétricas (exclusivas de un dataset) van a "extras por dataset", no se fuerzan al núcleo canónico | El núcleo canónico permite experimentos cross-dataset; los extras maximizan señal por dataset |
| S5 | Campos categóricos de UNSW (`proto`, `service`, `state`) → `x_txt` | Alimenta la rama textual; el modelo debe aprenderlos, no hardcodearlos |
| S6 | **EMBER2024 heads**: Noul `is_malicious` (label_binary) + Choice `choice_family` = **top-10 familias AV del train + `unknown`**; SIN head Score | Censo train completo: `family` cubre 87.9% de los malicious; top-10 (berbew…flystudio) cubre 40.7%; Unknown absorbe benignos + sin familia (12.1%) + cola de ~46,590 familias; P(Unknown) ≈79.6% se REPORTA como honestidad, no como fallo; no hay GT de severidad → Score sin objetivo (§3.4) |
| S7 | **EMBER2024: vectorización local por semanas + borrado** del jsonl de cada semana tras escribir su part parquet (float32); challenge se conserva como jsonl + `challenge.parquet` único | 53 GB de jsonl vs ~68 GB libres: el borrado semanal mantiene el pico de disco; float32 a 2568 dims × 3.85M filas = la mitad que float64; parts sharded = resume sin re-procesar semanas; todo re-descargable con el bootstrap |

---

## 1. Núcleo canónico (intersección IDS2017 ∩ UNSW-NB15)

Grupos de features presentes en ambos datasets (mapeo exacto por tabla):

| Grupo canónico | CIC-IDS2017 (GeneratedLabelledFlows) | UNSW-NB15 |
|---|---|---|
| **duración** | `Flow Duration` | `dur` |
| **conteo paquetes fwd/bwd** | `Total Fwd Packets` / `Total Backward Packets` | `spkts` / `dpkts` |
| **bytes fwd/bwd** | `Total Length of Fwd Packets` / `... Bwd ...` | `sbytes` / `dbytes` |
| **tasas** | `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s` | `rate` |
| **IAT / inter-arrival** | `Flow IAT Mean/Std/Max/Min`, `Fwd IAT Total/Mean/Std/Max/Min`, `Bwd IAT ...` | `sinpkt`, `dinpkt` (aprox: per-pkt, no flow-level) |
| **jitter** | — | `sjit`, `djit` |
| **tamaño medio de paquete fwd/bwd** | `Avg Fwd Segment Size`, `Avg Bwd Segment Size`, `Packet Length Mean/Std/Max/Min/Variance`, `Fwd/Bwd Packet Length Max/Min/Mean/Std` | `smean`, `dmean` |
| **ventanas TCP init** | `INIT Win bytes forward`, `INIT Win bytes backward` | `swin`, `dwin` ✓ (verificado en header real, 2026-10-16) |
| **flags** | `FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count` (8 cols) | — (sin equivalentes) |
| **TTL** | — | `sttl`, `dttl` |
| **cargas** | `sload`-equivalentes no existen; usar `Flow Bytes/s` | `sload`, `dload` |
| **pérdidas/retransmisiones** | — | `sloss`, `dloss` |
| **RTT TCP** | — | `tcprtt`, `synack`, `ackdat` |
| **subflow** | `Subflow Fwd Packets/Bytes`, `Subflow Bwd Packets/Bytes` | — |
| **bulk** | `Fwd/Bwd Avg Bytes/Bulk Rate/Packets/Bulk` (6 cols) | — |
| **active/idle** | `Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min` | — |
| **conexiones previas** | — | `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_srv_dst` ✓ |
| **HTTP/FTP** | — | `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd`, `trans_depth`, `response_body_len`, `is_sm_ips_ports` |
| **header len** | `Fwd Header Length`, `Bwd Header Length` ⚠️ (algunas versiones duplican col) | — |
| **down/up ratio** | `Down/Up Ratio` | — |

⚠️ Los nombres exactos varían entre versiones del dataset (espacios, guiones, duplicados raros como la columna `Fwd Header Length` duplicada en algunas versiones de MachineLearningCSV). El script `src/features/build_canonical.py` (F1) hará el mapeo por diccionario explícito y FALLARÁ si falta una columna esperada — sin heurísticas silenciosas.

### Normalización (compartida)
- Todas las features numéricas → quantile transform (o robust scale) ajustado SOLO en train (evitar leakage).
- Valores infinitos/NaN (producidos por divisiones por flujo de duración 0 — típicos en IDS2017) → imputación documentada + indicador binario.
- Log1p en features de escala monetaria de bytes/rates.

## 2. `x_txt` por dataset

| Dataset | Campos `x_txt` |
|---|---|
| CIC-IDS2017 (GenLabelledFlows) | `protocol` (TCP/UDP/…), `flow_id` (ip-puerto-par serializado), `timestamp` (no-feature: meta) |
| UNSW-NB15 | `proto` (tcp/udp/…), `service` (http, dns, ftp…, "-" si vacío), `state` (INT, FIN, CON, ACC, …) |
| EMBER2024 | (vacío — rama textual desactivada) |

⚠️ UNSW `service`/`state` son de bajo cardenalidad: el n-gram hasher de la rama textual los maneja sin necesidad de vocabulario fijo (v0.1); documentar.

## 3. Taxonomía de etiquetas → familias de `Choice`

### 3.1 CIC-IDS2017 (`Label` en CSV) — censo verificado 2026-10-16: 15 etiquetas, 2,830,743 flujos

| Etiqueta nativa (graffía EXACTA) | `choice_family` | n | archivo(s) |
|---|---|---|---|
| BENIGN | Normal | 2,273,097 | los 8 |
| DDoS | DoS/DDoS | 128,027 | Friday-DDos |
| DoS Hulk | DoS/DDoS | 231,073 | Wednesday |
| DoS GoldenEye | DoS/DDoS | 10,293 | Wednesday |
| DoS slowloris | DoS/DDoS | 5,796 | Wednesday |
| DoS Slowhttptest | DoS/DDoS | 5,499 | Wednesday |
| PortScan | Recon | 158,930 | Friday-PortScan |
| FTP-Patator | BruteForce | 7,938 | Tuesday |
| SSH-Patator | BruteForce | 5,897 | Tuesday |
| Web Attack � Brute Force | WebAttack | 1,507 | Thursday-Morning |
| Web Attack � XSS | WebAttack | 652 | Thursday-Morning |
| Web Attack � Sql Injection | WebAttack | 21 | Thursday-Morning |
| Bot | Botnet | 1,966 | Friday-Morning |
| Infiltration | Infiltration | 36 | Thursday-Afternoon |
| Heartbleed | Exploitation | 11 | Wednesday |

Balance global: 80.3% Normal / 19.7% ataque. Notas: `DoS slowloris` con **s minúscula**, `DoS GoldenEye` con mayúsculas. **El separador de las etiquetas Web Attack es U+FFFD** (carácter de reemplazo — mojibake del CSV oficial; bytes `ef bf bd` verificados por hexdump 2026-10-16; el archivo es UTF-8 válido, y otras versiones del dataset usan en-dash U+2013). El compilador normaliza (casefold + �/–/— → '-') antes del lookup. El typo `Infilteration` está SOLO en el nombre del archivo; el Label dice `Infiltration`. ⚠️ Para E1/E2: `Infiltration`=36 y `Heartbleed`=11 son microscópicas — el LOAO de E2 las sostiene (la clase hold-out se detecta, no se entrena), pero E1 debe reportar micro+macro y por-clase.

### 3.2 UNSW-NB15 (`attack_cat`) — política NATIVA sin colapso; censo verificado 2026-10-16

| attack_cat nativo (grafía del CSV) | n train | n test | `choice_family` |
|---|---|---|---|
| Normal | 56,000 | 37,000 | normal |
| Analysis | 2,000 | 677 | analysis |
| **Backdoor** (⚠️ SINGULAR en el CSV; la doc del dataset dice "Backdoors") | 1,746 | 583 | backdoor |
| DoS | 12,264 | 4,089 | dos |
| Exploits | 33,393 | 11,132 | exploitation |
| Fuzzers | 18,184 | 6,062 | fuzzing |
| Generic | 40,000 | 18,871 | generic |
| Reconnaissance | 10,491 | 3,496 | recon |
| Shellcode | 1,133 | 378 | shellcode |
| Worms | 130 | 44 | worms |

Totales: 175,341 train + 82,332 test = 257,673 ✓ (censo por `awk` sobre los CSVs crudos, bytes verificados con `cat -A`).

**Decisión congelada:** familia nativa por dataset + `Unknown` — SIN colapso. Los mapeos aproximados al unión (Backdoor→Infiltration, Analysis→Recon, Worms→Exploitation) quedan SOLO para experimentos cross-dataset futuros (E2b, stretch). El compilador acepta ambas grafías "Backdoor"/"Backdoors".

### 3.3 `choice_ddos_vector` (IDS2017, subconjunto DoS/DDoS) — conteos verificados 2026-10-16

| Etiqueta nativa | `choice_ddos_vector` | n | archivo |
|---|---|---|---|
| BENIGN | Normal | (del subconjunto elegido) | — |
| DDoS (LOIT) | tcp_flood | 128,027 | **Friday-DDos** |
| DoS Hulk / DoS GoldenEye | http_dos | 231,073 / 10,293 | Wednesday |
| DoS slowloris / DoS Slowhttptest | slowloris | 5,796 / 5,499 | Wednesday |

⚠️ **Hallazgo del censo:** el label `DDoS` NO aparece el miércoles — vive en `Friday-...-DDos.csv`. La spec F0 congeló el subset en "miércoles"; tal cual, la clase `SYN_Flood` quedaría con **0 muestras de entrenamiento**.

**✅ RESUELTA (2026-10-16, evidencia sobre el parquet canónico):**
1. **Subset = Wednesday + Friday-DDoS** (verificado por columna `day`: DDoS 128,027/128,027 en `Friday-WorkingHours-Afternoon-DDos`; Hulk 231,073, GoldenEye 10,293, slowloris 5,796 y Slowhttptest 5,499 al 100% en `Wednesday-workingHours`; BENIGN presente los 8 días — el subconjunto toma el BENIGN de esos dos días). No hay alternativa: sin el viernes, `SYN_Flood` no tiene ni una muestra.
2. **Modo de LOIT: flood TCP volumétrico, NO HTTP semántico** → el candidato se llama **`tcp_flood`** (nombre cerrado por el diseño F2, decisión D8: `SYN Flag Count = 0` en el 100% de los flujos DDoS por artefacto de CICFlowMeter — llamarlo SYN_Flood reclamaría un mecanismo que los datos no exhiben; la definición es operacional, "flood TCP volumétrico"). Firma empírica DDoS vs DoS Hulk (HTTP confirmado): longitud media de paquete fwd **7.4 B** vs 44.4 B (el GET mínimo de HTTP pesa >16 B — 7 B es junk, no peticiones), flujos cortos (mediana 1.9 s vs 84.9 s), PSH medio 0.45 vs 0.06, puerto 80 en 128,024/128,027. Mapeo congelado (F2 D8): DDoS→tcp_flood, DoS Hulk→http_dos, DoS GoldenEye→http_dos, DoS slowloris→slowloris, DoS Slowhttptest→slowloris, BENIGN→normal.
3. Las 4 clases quedan pobladas: tcp_flood ← DDoS 128,027; http_dos ← Hulk+GoldenEye 241,366; slowloris ← slowloris+Slowhttptest 11,295; normal ← BENIGN del subconjunto. Subconjunto verificado por F2: 918,437 filas (Wednesday completo + Friday-DDoS, sin Friday-PortScan); train strat: normal 376,380 / tcp_flood 89,598 / http_dos 169,034 / slowloris 7,850.

### 3.4 EMBER2024 (Win32) — labels, familias y heads (censo 2026-10-16)

**Labels (grep exacto sobre los 64 jsonl, cuadra con los totales oficiales):**

| Subset | Líneas | label=0 (benign) | label=1 (malicious) | unlabeled |
|---|---|---|---|---|
| 52 semanas train | 3,120,000 | 1,560,000 | 1,560,000 | 0 |
| 12 semanas test | 720,000 | 360,000 | 360,000 | 0 |
| challenge (evasivo) | 6,315 | 0 | 6,315 | 0 |

Balance 50/50 exacto en train y test; **cero filas sin label** (a diferencia de EMBER2018); todas las semanas traen exactamente **60,000 líneas**. El challenge es 100% malicious y evasivo → test de estrés de E2. Fechas train 2023-09-24 → 2024-09-21; test 2024-09-22 → 2024-12-14.

**Familias AV (CENSO DEL TRAIN COMPLETO, 2026-10-16, `logs/ember_family_census.log`):** cobertura `family` entre malicious = **87.9%** (1,371,318/1,560,000; sin familia 188,682 = 12.1%); **46,607 familias distintas** → cola extrema (muchas singletons; la muestra de 4 semanas solo veía ~1,490). Top-15:

| # | familia | n |
|---|---|---|
| 1 | berbew | 256,810 |
| 2 | cosmu | 60,786 |
| 3 | wacatac | 45,118 |
| 4 | sfone | 44,216 |
| 5 | upatre | 43,954 |
| 6 | grandoreiro | 40,728 |
| 7 | glupteba | 39,814 |
| 8 | expiro | 35,434 |
| 9 | sinowal | 34,560 |
| 10 | **flystudio** | 34,262 |
| 11 | tempedreve | 17,974 |
| 12 | virut | 14,972 |
| 13 | barys | 14,386 |
| 14 | floxif | 13,686 |
| 15 | emdup | 12,764 |

**Top-10 congelado (S6):** berbew, cosmu, wacatac, sfone, upatre, grandoreiro, glupteba, expiro, sinowal, flystudio = 635,682 malicious (**40.7%** de los malicious; el corte 10→11 es limpio: 34,262 vs 17,974, casi 2×). ⚠️ El placeholder de la muestra de 4 semanas tenía **3 familias fuera del top-10 real** (eggnog/gepys/urelas eran ráfagas de sept-oct 2023): correr el censo ANTES de materializar evitó re-descargar 53 GB.

**Heads congelados (S6):**
- Noul `is_malicious` ← `label` binario (balance 50/50; métrica principal E1: FPR@95TPR).
- Choice `choice_family` ← top-10 familias + `unknown`. Reglas del compilador (`build_canonical.compile_ember_line`):
  - benign → `("unknown", "benign")`
  - malicious + familia top-10 → `(familia, familia)`
  - malicious + familia de cola (~1,480) → `("unknown", familia_cruda)`
  - malicious sin familia (null) → `("unknown", "malicious")`
- **Sin head Score**: EMBER no trae GT de severidad → `label_severity=None` siempre.

P(Unknown) esperado ≈ **79.6%** global = 50% benignos + 29.6% de malicious (12.1% sin familia + ~47.2% en la cola fuera del top-10) — se REPORTA como honestidad del Choice (S6), no como fallo: el detector es Noul; Choice aporta atribución cuando la familia cae en el top-10 (40.7% de los malicious). ⚠️ Corrección 2026-10-16: una nota anterior estimaba ≈57% al confundir la cobertura de `family` (87.9%) con la del top-10 (40.7%).

## 4. Rubrik de severidad (head `Score`, 4 niveles [Benign, Low, High, Critical])

| familia | severidad |
|---|---|
| Normal | Benign (0) |
| Recon, Fuzzing, Generic | Low (1) |
| DoS/DDoS, BruteForce, WebAttack, Analysis | High (2) |
| Botnet, Infiltration, Exploitation, Shellcode/Backdoor/Worms | Critical (3) |

⚠️ **Limitación declarada** (spec §8.1): el ground truth de severidad NO existe en los datasets; esta tabla es un rubrik de diseño documentado, no etiquetas reales.

## 5. Política de splits (congela F1)

| Dataset | Split |
|---|---|
| UNSW-NB15 | Oficial: `training-set` (175,341) → train+val (85/15 interno, estratificado por familia), `testing-set` (82,332) → test |
| CIC-IDS2017 | `split_strat`: estratificado 70/15/15 por familia (`SEED=42`) **Y** `split_time`: corte posicional 70/15/15 sobre el orden Lun→Mar→Mié→Jue→Vie preservando el orden de captura intra-día (el DÍA es el proxy temporal — MLCVE no trae timestamps) — ambas columnas en el parquet, ambas reportadas en E1 |
| EMBER2024 | **Temporal por semana (S7), NO estratificado** — estratificar mezclaría el futuro en train: 52 semanas train → **semanas 0-45 train (2,760,000) / 46-51 val (360,000)**; 12 semanas test → test (720,000); challenge evasivo (6,315) → test de estrés E2. Congelado en columnas `subset` + `week_idx` (0-51 train/val, 52-63 test, 64 challenge) |
| Seeds | `SEED = 42` congelada en `src/features/materialize.py` (D7); cambiarla invalida los splits previos |
| Regla dura | Calibración (temperature/isotonic) SOLO sobre val; test se toca una vez por experimento |

**Salida de F1** (`src/features/materialize.py`): `data/processed/unsw_nb15/unsw_nb15.parquet`, `data/processed/cicids2017/cicids2017.parquet` y `data/processed/ember2024/part_{ii:02d}_{subset}.parquet` + `challenge.parquet` — `x_num` como `fixed_size_list<float>` (39 / 77 en float64; **2568 en float32** por volumen), labels (`label_family`/`label_full`/`label_severity`; EMBER añade `label_binary` y NO trae `label_severity`) y columnas de split por fila (`split` en UNSW; `split_strat`+`split_time` en CIC; `subset`+`week_idx` en EMBER). NaN/inf se conservan tal cual en `x_num` (UNSW/CIC): la imputación train-only se ajusta en la etapa de features del modelo, no aquí. EMBER se materializa con **12 workers que cada uno escribe su semana y borra su jsonl** (la escritura vive en el worker: devolver 615 MB/semana por IPC reventaba la RAM de 15 GB; buffer 25K ≈ 400 MB pico/proceso) y verificación global 2,760,000 / 360,000 / 726,315 (= 720,000 test + 6,315 challenge) leyendo solo `subset`+`label_family` antes de dar el lote por bueno.

## 6. Checklist de congelación v1.0

- [x] **S1 RESUELTA — headers y contenido reales del enhanced verificados** (2026-10-16, inspección de `CICIDS_Flow.parquet`): **84 columnas** — metadatos (`flow_id` int64 único sin nulls, `source_ip/port`, `destination_ip/port`, `protocol` **string** ∈ {tcp 1,826,505, udp 999,476, other 1,696}, `Timestamp` string en 2 formatos `DD/MM/YYYY HH:MM:SS` y `D/M/YYYY H:MM`, cero nulls) + las 78 features de MLCVE (mismos nombres) + `attack_label`. **2,827,677 filas** (−3,066 vs MLCVE = −0.1%; distribución por clase casi idéntica, mismas micro-clases: Infiltration 36, Heartbleed 11). **Labels LIMPIAS**: `Web Attack – Brute Force` con en-dash U+2013 (sin el mojibake U+FFFD del MLCVE; `_norm_cic_label` ya normaliza ambas). **Decisión congelada:** (a) el enhanced es el **sustrato E5** — la 5-tupla + timestamp permite casar alertas de Suricata con flujos (imposible con MLCVE), y sus 77 features numéricas se computan con el MISMO orden de `CIC_FEATURE_COLUMNS`, así el modelo entrenado en MLCVE aplica directo; (b) `split_time` por `Timestamp` real (dos formatos a parsear) cuando se use como fuente de splits; (c) MLCVE queda como fuente de features de comparabilidad (parquet materializado, d_num=77 — **F2 §4.4 lo cerró para v0.1: NO re-materializar**; el modelo v0.1 va sin rama textual en CIC porque `protocol` es None en MLCVE, y el enhanced queda congelado como sustrato E5); (d) `flow_id` abre un join opcional con el parquet `Payload-Bytes` del mismo autor (rdpahalavan) para x_txt rico — decisión v0.2, NO bloquea. ⚠️ Corrección a la nota previa: este archivo NO trae Payload-Bytes (esa es otra parte del repo del autor).
- [x] Headers reales de `UNSW_NB15_training-set.csv` (2026-10-16): 45 columnas = id + proto/service/state (→ x_txt) + 39 features numéricas + attack_cat + label; UTF-8 BOM presente → leer con `encoding="utf-8-sig"`; `dwin` y `ct_srv_dst` confirmados. **Corrección misma fecha:** el conteo original decía "43 features" — en realidad contaba id + 3 textuales + 39 numéricas (= 43 columnas no-label); `d_num` congelado es **39**. Lo pilló el assert fail-fast de `build_canonical.py` antes de materializar nada.
- [x] UNSW `testing-set.csv` canónico descargado y verificado (2026-10-16, mirror GitHub Nir-J): 82,332 registros; distribución por familia correcta (Normal 37K, Generic 18.9K, Exploits 11.1K, …, Worms 44). ⚠️ El mirror HF `jharrrry/UNSW-NB15` tiene `test.csv` mal etiquetado (md5 idéntico al training-set) — NO usar. Censo `attack_cat` completo en §3.2 — incluida la grafía real **"Backdoor"** (singular): la 1ª materialización la mandó a `unknown` (2,329 filas), pillado por el crosstab del materializador y corregido antes de cualquier experimento.
- [x] Headers reales de MachineLearningCVE (2026-10-16, mirror `c01dsnap/CIC-IDS2017`): **79 columnas** = 78 features + `Label`. Nombres con espacios (strip), separador coma, sin BOM en Wednesday. Detalles confirmados: `act_data_pkt_fwd`, `min_seg_size_forward`, `Init_Win_bytes_forward/backward` (guiones bajos), **`Fwd Header Length` duplicada** (pandas la renombrará `Fwd Header Length.1` — descartar la duplicada), SIN IPs/ports origen ni timestamp (solo `Destination Port`) — confirma que el parquet enhanced es la fuente primaria (S1).
- [x] Lista exacta de etiquetas Label (IDS2017) y conteo por clase (censo 2026-10-16): 15 etiquetas, 2,830,743 flujos, totales por familia en §3.1.
- [x] Conteo exacto de columnas por archivo (2026-10-16): los 8 CSVs tienen header **idéntico** (md5 `e1cc556bed…`) — 79 cols con la duplicada `Fwd Header Length` en todos; un solo schema congela los 8.
- [x] Dim exacta de features EMBER2024 (2026-10-16, verificada del código fuente thrember 0.1.0, `PEFeatureExtractor`): **2568** = GeneralFileInfo 7 + ByteHistogram 256 + ByteEntropyHistogram 256 + StringExtractor 34 + HeaderFileInfo 177 + SectionInfo 224 + ImportsInfo 1282 + ExportsInfo 129 + DataDirectories 74 + RichHeader 33 + AuthenticodeSignature 8 + PEFormatWarnings 88 (las 12 por defecto; `features_file` opcional permite subconjuntos). Los jsonl traen `label` (binario), `family`, `group`, `packer`, `exploit`, `behavior`… — mapeo congelado en S6/§3.4 (2026-10-16): labels censados (50/50 exacto, 60,000 líneas/semana, cero unlabeled), heads Noul `is_malicious` + Choice top-10, sin Score.
- [x] Resolver mapeos ⚠️: `dwin` ✓ verificado en header real; Backdoor/Analysis/Worms → decidido en §3.2: NO colapsar (familias nativas por dataset + Unknown); los mapeos aproximados solo para experimentos cross-dataset futuros.
- [x] Congelar `d_num` por dataset (2026-10-16): UNSW 39 (corregido 43→39 el mismo día — ver nota en el ítem de headers), CIC-MLCVE 77, EMBER2024 2568. La configuración cross-dataset "core-only" queda como experimento futuro (no bloquea F1).
- [x] Labels EMBER2024 censados (2026-10-16, grep exacto sobre los 64 jsonl): train 1,560,000/1,560,000, test 360,000/360,000, challenge 6,315 (todo malicious) — cero unlabeled, balance 50/50 exacto, 60,000 líneas/semana. Congelado en §3.4 con los heads S6.
- [x] **Censo top-10 familias del train COMPLETO** (2026-10-16, 3,120,000 líneas, `scripts/census_ember.awk` + `xargs -P 12` → `logs/ember_family_census.log`): totales cuadraron al centavo (1,560,000/1,560,000), cobertura 87.9%, **46,607 familias**, top-15 en §3.4. El placeholder de 4 semanas tenía 3 familias fuera (eggnog/gepys/urelas → entraron grandoreiro/glupteba/expiro): `native_families` corregido en `state.py` ANTES de materializar — censo-first evitó re-descargar 53 GB.
- [x] **EMBER2024 MATERIALIZADO — F1 COMPLETO** (2026-10-16, `logs/ember_materialize.log`): 65 parquets en `data/processed/ember2024/` (8.9 GB, zstd; 52 train w0-45 + 6 val w46-51 + 12 test w52-63 + challenge.parquet), `x_num` fixed_size_list<float32>[2568]. Verificación global ✓ exacta: **train 2,760,000 / val 360,000 / test 726,315** (720,000 + challenge 6,315); P(Unknown) global **80.0%** (3,076,572/3,846,315; esperado ≈79.6%). Crosstab final (train/val/test, verbatim de la verificación): unknown 2,195,408/288,910/592,254; berbew 222,302/34,508/92,086; cosmu 39,526/21,260/9,562; wacatac 37,658/7,460/12,316; upatre 42,294/1,660/6,430; sfone 43,720/496/**132**; expiro 33,382/2,052/7,967; glupteba 38,180/1,634/3,412; grandoreiro 40,698/30/**226**; flystudio 32,344/1,918/1,912; sinowal 34,488/72/**18**. ⚠️ **Hallazgo de drift temporal**: sfone, grandoreiro y sinowal casi DESAPARECEN en test (0.3%, 0.6%, 0.05% de su train) — el split temporal es drift real, no iid; material para la narrativa E2 (zero-day) y para discutir distribución shift en E1. Challenge (semana 64, evasivo): 6,236/6,315 unknown (98.8%), wacatac 78, expiro 1 — el malware evasivo apenas es atribuible a familias comunes: Choice degrada a Unknown (honestidad), Noul binario queda como detector. Spot-check de esquema ✓ (parte train y challenge: 2568 dims, labels S6 correctos; 1ª fila challenge = `doenerium`, familia cola → full=doenerium/family=unknown). Bugs pillados y corregidos en el mismo día (3): (1) `pyarrow` nombra la columna de conteo de `group_by` distinto según versión → acceso posicional `agg.column(agg.num_columns - 1)`; (2) el resume abortaba con 0 jsonl (estado NORMAL post-S7, los jsonl ya se borraron) → guard `(0,0)|(52,12)`; (3) el bloque challenge exigía jsonl aunque el parquet ya existiera → reordenado. **Los tres datasets de F1 están materializados y verificados: UNSW 257,673 + CIC 2,830,743 + EMBER 3,846,315.**
