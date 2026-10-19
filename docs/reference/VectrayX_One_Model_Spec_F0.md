# VectrayX-One — Spec Formal del Modelo (F0)

**Versión:** 0.1 (2026-10-16) — congelada para implementación
**Alcance:** define el motor de decisión System One de VectrayX-One (Fase 2): el objeto `State`, las primitivas `Noul`/`Choice`/`Score` con definiciones formales, la arquitectura de red, el procedimiento de calibración y el protocolo experimental E1–E7. La Fase 1 (determinista) se especifica como implementación de referencia (Suricata + ET Open) para los experimentos de sistema.
**Documentos hermanos:** `Literature_Review_VectrayX_One.md` (fuentes y posicionamiento), `VectrayX_One_Paper_Draft.md` (narrativa a corregir).

---

## 0. Decisiones fijadas

| # | Decisión | Racional |
|---|---|---|
| D1 | Entrenamiento en Modal (GPU serverless); features y evaluación ligera local (CPU) | Sin GPU local; pago por uso respalda la métrica de costo del paper |
| D2 | Datasets primarios: CIC-IDS2017 (etiquetas auditadas), UNSW-NB15, EMBER2024; EMBER2018 como comparabilidad | Lit review §8 |
| D3 | Fase 1 de referencia: Suricata 7 + reglas ET Open sobre PCAPs de CIC-IDS2017; fallback: firma simulada sobre features de flujo | El motor C++ VectrayX no existe; Suricata usa formato de reglas tipo Snort |
| D4 | E5 (sistema de dos fases) aplica solo al dominio red; el dominio malware evalúa el motor standalone | No hay "Fase 1 de red" aplicable a features estáticas de PE; EMBER no distribuye binarios crudos |
| D5 | Framework: PyTorch; baselines con XGBoost/LightGBM, scikit-learn, HF Transformers | Estándar de la literatura, reproducible |
| D6 | E7 (Microsoft-Decision-1 vía API) es **opcional**, no bloquea nada | Requiere cuenta Azure Foundry/OpenRouter del autor (~$2) |
| D7 | PyTorch, seeds fijas, versiones pinneadas; configs JSON por experimento | Reproducibilidad académica |

---

## 1. El objeto `State` (v0.1)

**Definición formal.** Un evento de telemetría se compila a un objeto de estado denso:

$$S = (x_{num},\ x_{txt}) \in \mathbb{R}^{d_{num}} \times T^{*}$$

donde `x_num` es el vector de características numéricas de tamaño fijo y `x_txt` es una tupla de campos de texto crudo (longitud acotada a 256 chars, ausencia permitida).

### 1.1 Dominio red (CIC-IDS2017 / UNSW-NB15)

- **x_num:** esquema canónico intersección de ambos datasets (duración, conteos de paquetes fwd/bwd, estadísticas de bytes, conteos de flags, bytes/s, pkts/s, estadísticas IAT, subflows, active/idle) + extras por dataset documentados en `data/SCHEMA.md` (F1). El mapeo columna→canónico queda congelado ahí antes de entrenar.
- **x_txt:** `{user_agent, uri, service_banner, conn_state}` cuando existan (IDS2017 con payload extraíble de PCAP; UNSW-NB15 no tiene payloads → `x_txt` vacío, la rama textual se desactiva y es exactamente la abalización E6).
- Unidad de evento v0.1: **flujo**. Las features de ventana del draft (`syn_ratio`, `entropy_ip_source`, `burst_factor` a Δt=10 s por IP origen) quedan como variante derivada v0.2, no bloquean nada.

### 1.2 Dominio malware (EMBER2024)

- **x_num:** vector de features estáticas de PE del dataset (dim exacta se fija en F1 desde el paper/repo — EMBER2018 es 2381).
- **x_txt:** vacío (las features EMBER ya condensan strings/secciones en histogramas). Rama textual no aplica.

**Interfaz de código (contrato del compilador de estado):**

```typescript
interface State {
  telemetry_source: string;          // "cicids2017_v1" | "unsw_nb15_v1" | "ember2024_v1"
  x_num: Float32Array;               // d_num fijo por source, documentado
  x_txt: Record<string, string|null>; // claves fijas por source, valores truncados a 256 chars
  meta: { split_hint: "train"|"val"|"test", time: number };
}
```

---

## 2. Primitivas — definiciones formales

> Esta es la formalización académica que ningún paper tiene todavía (Lit review §1). Se cita a Microsoft-Decision-1 como definición canónica de producto y a Gu et al. 2018 por el término "no-autorregresivo".

Sea $f_\theta: S \mapsto h$ el **encoder** (una sola pasada forward — sin decodificación secuencial), y $Q$ un conjunto de **preguntas tipadas declaradas en configuración** (no aprendidas):

### 2.1 `Noul` — gate probabilístico

$$\text{noul}_q(S) = \sigma(w_q^\top h + b_q) \in [0,1]$$

con la **propiedad objetivo de calibración**:

$$\mathbb{P}(y_q = 1 \mid \text{noul}_q(S) = p) = p, \quad \forall p \in [0,1]$$

- Se alcanza por calibración post-hoc (§4) y se **verifica empíricamente** (E3).
- `Noul` NO es un booleano: es la probabilidad calibrada. La decisión ejecutable es $(\text{noul}_q(S), \tau_q) \mapsto \{true, false\}$, donde el umbral $\tau_q$ es **política del controlador compilado**, no del modelo. (Corrige el error de tipado del draft.)
- Relación con conformal prediction (discusión del paper): un wrapper conformal sobre $\text{noul}_q$ da garantías de cobertura distribution-free (Angelopoulos & Bates 2021); se implementa como extensión opcional v0.2.

### 2.2 `Choice` — clasificación multiclase sobre conjunto fijo

$$p(k \mid S) = \text{softmax}(W_q h), \quad \text{choice}_q(S) = (\arg\max_k\, p(k \mid S),\ p(\cdot \mid S))$$

- El conjunto de candidatos $C_q$ es **cerrado y declarado**, e incluye SIEMPRE la clase explícita `Unknown` — decisión clave para honestidad ante día cero (E2): un ataque no visto debe caer en `Unknown` o escalar, no ser absorbido por una clase conocida.
- Escalación por confianza baja: si $\max_k p(k|S) < \tau_{esc}$, el evento se marca para revisión (selective classification, Geifman & El-Yaniv 2017).

### 2.3 `Score` — escala ordinal

Niveles declarados $L_q = [\ell_0 < \dots < \ell_m]$:

$$p(\ell_j \mid S) = \text{softmax}(W_q' h), \qquad \text{score}_q(S) = \sum_{j=0}^{m} j \cdot p(\ell_j \mid S) \in [0, m]$$

(promedio ponderado por probabilidad — idéntico a la definición de Microsoft-Decision-1; citar).

### 2.4 No-autoregresividad

Todas las preguntas $Q$ se evalúan en **una sola pasada** $f_\theta(S)$: los heads comparten el encoder, no hay bucle de muestreo $P(t_n \mid t_{<n})$, no hay tokens de salida. La garantía de formato es estructural (los heads son logits tipados), no probabilística.

---

## 3. Arquitectura del motor (implementable)

```
State
 ├─ x_num ──► tabular_encoder ──► h_num   (MLP: [Linear→BN→ReLU]×N; variante FT-Transformer)
 ├─ x_txt ──► text_encoder    ──► h_txt  (hash de n-grams de caracteres → mini-transformer
 │                                          o char-CNN; solo dominio red)
 └─ concat(h_num, h_txt) ──► fusion MLP ──► h
       │
       ├─ Choice head:  Linear(K) ──► softmax        (una por configuración de choice)
       ├─ Score  head:  Linear(m+1) ──► softmax
       └─ Noul   heads: Linear(1) ──► sigmoid        (una por gate, G gates)
```

- **Tamaños (para E4/E6):** Nano ≈ 100K params (objetivo edge/CPU < 10 ms), Small ≈ 1M, Base ≈ 5–10M.
- **Entrenamiento multi-tarea:** $\mathcal{L} = \sum_q \lambda_q \mathcal{L}_q$ con $\lambda_q$ por ponderación de incertidumbre homeostática (Kendall et al. 2018) o grid sobre val.
- **Gates `Noul` v0.1 (G=3):** `is_attack`, `requires_immediate_drop`, `is_recon_low_urgency` — cada uno con instrucción y criterios true/false declarados (mismo formato que el API de Microsoft-Decision-1).
- **Score v0.1:** severidad `[Benign, Low, High, Critical]`. **Ground truth:** no existe en los datasets → etiqueta derivada de un **rubrik documentado familia→nivel** (tabla en apéndice del paper). Limitación declarada (§8).

### 3.1 Configuraciones de `Choice` (dominio red)

| Config | Candidatos | Dataset | Nota |
|---|---|---|---|
| `choice_family` | {Normal, DoS/DDoS, Recon, BruteForce, WebAttack, Botnet, Infiltration, Exploitation, Unknown} | IDS2017 + UNSW (mapeo por familia) | Configuración principal |
| `choice_ddos_vector` | {Normal, SYN_Flood, HTTP_DoS, Slowloris, Unknown} | IDS2017 (miércoles DDoS) | Reproduce el ejemplo fino del draft (Slowloris ✓, GoldenEye/Hulk → HTTP_DoS ✓, LOIT → SYN_Flood ✓) |

---

## 4. Calibración

1. **Post-hoc, por head, sobre el conjunto de validación** (nunca sobre test).
2. `Choice`/`Score`: temperature scaling (Guo et al. 2017). `Noul`: Platt scaling e isotonic regression; selección por ECE en val.
3. **Comparación justa (anti-strawman):** los baselines clásicos (XGBoost/RF) se reportan **sin calibrar Y calibrados** (Platt/isotonic) — mostramos la distorsión típica de boosted trees sin calibrar (Niculescu-Mizil & Caruana 2005) sin negar el fix a los baselines.
4. Reporte: ECE (15 bins), MCE, Brier, reliability diagrams (E3).

---

## 5. Protocolo experimental (E1–E7)

### E1 — Calidad de detección
- **Splits:** UNSW-NB15 split oficial (175,341/82,332); IDS2017 estratificado 70/15/15 **más variante por tiempo** (robustez); EMBER split canónico del paper (con su parte unlabeled para el challenge).
- **Baselines:** XGBoost, Random Forest, MLP tabular, DistilBERT fine-tuned (texto del flujo serializado como línea de log — el baseline "semántico" honesto), y opcionalmente E7.
- **Métricas:** macro-F1, F1 por clase, PR-AUC, **FPR@95TPR** (métrica operativa de la línea de fuego), latencia co-reportada.

### E2 — Día cero (LOAO — el experimento estrella)
- Leave-one-attack-family-out: entrenar sin la familia $F$, evaluar contra $F$.
- UNSW: 9 folds. IDS2017: leave-out por tipo de ataque.
- Métricas sobre familia no vista: (a) AUC del gate `is_attack`, (b) % asignado a `Unknown` por el choice (debe ser alto), (c) ECE fuera de distribución.
- Este experimento distingue a VectrayX-One de Suricata (que no detecta lo que no tiene en firmas) y valida el claim central del draft.

### E3 — Calibración
- Reliability diagrams por gate; ECE/MCE/Brier; tabla raw-vs-calibrado de baselines (§4.3).

### E4 — Latencia y costo
- Hardware: CPU local (modelo exacto reportado), Modal T4/A10/L4; batch 1 y 64.
- p50/p95/p99 single-event, throughput, costo por 1M eventos desde billing real de Modal.
- DistilBERT medido en el mismo hardware (comparación de latencia honesta, no de datasheet).

### E5 — Sistema de dos fases (solo dominio red)
- Primario: Suricata 7 + ET Open sobre PCAPs de CIC-IDS2017 (~50 GB; disco OK). Alertas ET = hits de Fase 1; flujos sin alerta = **Zona Gris** → Fase 2.
- Métricas: % de eventos resueltos en F1 **por clase de ataque** (qué atrapan las firmas vs qué llega a la zona gris — tabla clave del paper), latencia E2E por fase, accuracy del sistema combinado vs F1-solo vs F2-solo.
- Fallback si los PCAPs no son tratables: firma simulada sobre features de flujo, declarado como limitación.

### E6 — Ablaciones
- Rama textual on/off (red); calibración on/off; opción `Unknown` on/off (impacto en E2); tamaño Nano/Small/Base; esquema de ponderación de loss.

### E7 — Baseline comercial (OPCIONAL, D6)
- Microsoft-Decision-1 zero-shot vía API (OpenRouter/Azure Foundry) sobre muestra estratificada de 1–5K eventos de test (`State` serializado como texto/JSON), mismas métricas, costo reportado.
- Resultado esperado por la literatura del proveedor: calibración "más fuerte en tipos de tarea familiares" → hipótesis: el modelo especializado gana en el dominio. Si E7 no se ejecuta, queda como trabajo futuro.

---

## 6. Infraestructura Modal

- App `vectrayx-one`; volumen `paper-data` (~30–60 GB); imagen: pytorch + xgboost + lightgbm + scikit-learn + transformers.
- GPUs: L4/T4 para entrenamiento (modelos chicos — basta GPU pequeña); DistilBERT fine-tune en L4; benchmarks E4 en T4/A10/CPU.
- Tracking: configs JSON + seeds + `requirements.txt` pinneado; resultados en `results/E{n}/` como JSON + plots.
- Suricata (E5) corre **local en CPU** (no necesita Modal).
- Presupuesto estimado: < $20–30 total en Modal (el fine-tune de DistilBERT es el ítem más caro).

---

## 7. Estructura del repo

```
paper/
  VectrayX_One_Paper_Draft.md
  Literature_Review_VectrayX_One.md
  VectrayX_One_Model_Spec_F0.md        ← este documento
  data/
    SCHEMA.md                          (mapeo columna→canónico por dataset; F1)
    raw/<dataset>/                     (cada dataset en su propio directorio)
  src/
    state/         (schemas + validación)
    features/      (mapeos dataset → State)
    models/        (encoder, heads, calibración)
    baselines/     (xgboost, rf, distilbert, suricata_runner)
    experiments/   (E1..E7)
    eval/          (métricas, reliability diagrams, plots)
  modal/          (app + imágenes)
  results/        (JSON/plots por experimento)
  paper/          (LaTeX final, cuando toque)
```

---

## 8. Riesgos y limitaciones (a declarar en el paper)

1. **Ground truth de `Score` es rubrik**, no etiquetas reales de severidad.
2. **Datasets sintéticos + errores de etiquetado** (Engels 2021; Rosay et al.) → usamos etiquetas auditadas/reportamos ambas variantes.
3. **LOAO mide generalización entre familias**, no contra amenazas del mundo real.
4. **Inyección en `x_txt`** (Greshake et al. 2023): fuera de alcance v0.1 → trabajo futuro de sanitización (ya en el draft).
5. **Fase 1 es referencia** (Suricata), no el motor C++ VectrayX real.
6. **E7 depende de un servicio externo** cuyo SLA/precios pueden cambiar.

## 9. Pendientes que no bloquean la implementación

- [ ] Dim exacta de features EMBER2024 (F1, del paper arXiv:2506.05074)
- [ ] `data/SCHEMA.md` — intersección canónica IDS2017 ∩ UNSW-NB15 (F1)
- [ ] Confirmar descarga de PCAPs CIC-IDS2017 (servidor UNB respondió 301 — resolver redirect)
- [ ] (Si se activa E7) credenciales Azure Foundry u OpenRouter del autor
