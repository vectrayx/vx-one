# VectrayX-One — Especificación de Diseño F2: Motor de Decisión VX-One

**Versión:** 1.0 · **Fecha:** 2026-10-17 · **Estado:** CONGELADO para implementación
**Documentos hermanos:** `VectrayX_One_Model_Spec_F0.md` (primitivas y protocolo E1–E7, congelada), `data/SCHEMA.md` (contratos de datos F1, congelado), `src/state/state.py` y `src/features/build_canonical.py` (implementación congelada de labels/features).
**Alcance:** arquitectura, matemática de losses, encodings, calibración, receta de entrenamiento en Modal, contrato de inferencia. Toda cifra de datos fue verificada por conteo directo sobre los parquets materializados (2026-10-17; procedencia en Apéndice B).

---

## 0. Alcance, contrato y resumen de decisiones

### 0.1 Contrato de entrada

El motor consume el estado canónico F1 por dataset (parquets materializados, `SEED=42`):

| dataset | filas (train/val/test) | d_num | split | rama textual |
|---|---|---|---|---|
| UNSW-NB15 | 149,039 / 26,302 / 82,332 (total 257,673) | 39 | oficial 85/15 interno | proto, service, state |
| CIC-IDS2017 (MLCVE) | 1,981,520 / 424,611 / 424,612 (total 2,830,743) | 77 | `split_strat` y `split_time` | **ninguna** (protocol=None) |
| EMBER2024 | 2,760,000 / 360,000 / 726,315 (total 3,846,315; test = 720,000 + challenge 6,315) | 2568 (float32) | temporal por semanas | ninguna |

Modelos por corrida: uno por (dataset, escala, split_scheme). **10 corridas principales**: UNSW × {Nano, Small, Base} (split oficial); CIC × {Nano, Small, Base} en `split_strat` + Small en `split_time`; EMBER × {Nano, Small, Base}. Folds LOAO en §6.7.

### 0.2 Resumen de decisiones (D1–D12)

| # | Decisión | Justificación (una línea) | § |
|---|---|---|---|
| D1 | Choice `unknown` híbrido: clase entrenada en EMBER (K=11); abstención por umbral τ_abs en UNSW/CIC | EMBER tiene GT de unknown (reglas S6, 2.19M filas); UNSW/CIC tienen 0 muestras de unknown → una clase entrenada sería indefinida | 3.3 |
| D2 | Noul: BCE **sin** pesos de clase | la BCE ponderada desplaza el logit óptimo por log(α+/α−) (shift aditivo) que la temperatura no puede revertir; desbalances moderados (68%/20%/50%) | 3.2 |
| D3 | Score: **CORN** (regresión ordinal condicional), sin pesos extra | pmf válida por construcción (CORAL exige biases ordenados); la partición condicional ya redistribuye la masa hacia severidades raras | 3.4 |
| D4 | Pesos sqrt-normalizados media-1 en Choice + **de-tilt analítico** z−log ω en inferencia | mitiga el colapso de gradiente (EMBER unknown 79.5%→40.0% de masa) y recupera el posterior exacto en población | 3.3, 3.5 |
| D5 | Multi-task: Kendall homeostático **modificado** (s_q init 0, máscaras de actividad) | escalas de loss entre cabezas difieren (ln K vs ln 2); arranque homogéneo + robustez a batches sin filas elegibles | 3.6 |
| D6 | Normalización: mediana-imputación de no-finitos + flag de fila + signed log1p + robust scale (mediana/IQR train) + clip ±8 | robustez a colas pesadas y a las 2,867 filas no-finitas de CIC; serialización mínima (3 vectores) | 4.1 |
| D7 | x_txt: embedding **lookup** con `<miss>`/`<unk>` (no hashing); CIC v0.1 sin rama textual y **sin re-materializar** | vocabs cerrados y diminutos (≤135): el hashing es pérdida pura; MLCVE mantiene comparabilidad con la literatura | 4.2–4.4 |
| D8 | Candidato 1 de `choice_ddos_vector` = **`tcp_flood`** ("flood TCP volumétrico") | SYN Flag Count=0 en el 100% de los flujos DDoS (artefacto CICFlowMeter): llamarlo SYN_Flood sería falsamente mecanístico | 3.1 |
| D9 | Gates del controlador: G=1 entrenado (Noul) + 2 gates **derivados** de salidas calibradas | la rubrik hace que P(sev≥2) y p(recon) sean márgenes exactas: sin cabezas redundantes y con ECE medible | 3.1 |
| D10 | Dims congeladas por escala; EMBER con d0 reducido (20/96/384) | EMBER es input-bound (la proyección 2568 domina); todos los totales caen dentro de presupuesto | 2 |
| D11 | Hiperparámetros congelados (AdamW, warmup+cosine, batch, epochs, patience, AMP, clip) | grid uniforme reproducible; sin tuning por dataset para no fabricar comparabilidad | 6 |
| D12 | Candidate set por corrida = familias nativas con ≥1 fila en train; `split_time` solo en Small | evita pesos indefinidos (n=0) y expone drift; el run temporal es variante de robustez, no producto principal | 3.3, 6 |

### 0.3 Cambios autorizados respecto de F0

1. **§2.3 F0 (Score):** softmax nominal → **CORN**. Misma salida tipada (distribución sobre niveles + valor esperado ∈ [0,3]); nueva parametrización ordinal. El brief F2 autoriza explícitamente la elección.
2. **§3 F0 (gates):** G=3 entrenados → G=1 entrenado + 2 derivados exactos (D9; deducción en §3.1.3).
3. **§3 F0 (Choice unknown):** mecanismo único → mecanismo híbrido por dataset (D1).
4. **§4 F0 (calibración):** "Platt/isotonic según ECE" → **temperature scaling únicamente**, 1 escalar por cabeza, ajustado por NLL en val (decisión congelada del brief).
5. **`build_canonical.CIC_DDoS_VECTOR_MAP`:** placeholder `SYN_Flood` → **`tcp_flood`** (D8). Se actualiza al implementar; este doc cierra el nombre.

Todo lo demás de F0 (primitivas formales, E1–E7, presupuesto, anti-strawman) se hereda sin cambios.

---

## 1. Notación

- Dataset $\mathcal{D} \in \{\text{UNSW}, \text{CIC}, \text{EMBER}\}$; $d = d_{\text{num}}$ ∈ {39, 77, 2568}.
- Estado $S = (x_{\text{num}}, x_{\text{txt}})$, con $x_{\text{num}} \in \mathbb{R}^{d}$ y $x_{\text{txt}}$ campos de texto opcionales.
- Feature stage $\phi$: sanitiza y normaliza → $\phi(x_{\text{num}}) \in \mathbb{R}^{d+1}$ (§4.1).
- Codificador: proyección numérica $P_0$ (Linear $d{+}1 \to d_0$), embeddings de texto $E_f$ (uno por campo), y tronco de $L$ bloques → $h = f_\theta(S) \in \mathbb{R}^{w}$.
- Cabezas $q$ (por dataset, §3.1): logits $z_q$, pesos de clase $\omega_k$ (griega, para no colisionar con el ancho $w$ del tronco).
- Choice: candidatos $\mathcal{C}_q = (c_1, \dots, c_K)$ con orden congelado; softmax sobre $z \in \mathbb{R}^K$.
- Score: niveles $\ell_0..\ell_3$ = {benign, low, high, critical}; condicionales $q_k$, acumuladas $C_k = P(y \ge k \mid x)$, pmf $\pi_k$.
- Calibración: temperatura $\hat T_q > 0$ por cabeza; probabilidades calibradas con superíndice cal.
- SEED=42 en todo (splits ya congelados en F1; entrenamiento en §6.1).

---

## 2. Arquitectura por escala

### 2.1 Template estructural

```
x_num ──φ (§4.1)──► [u ; r] ∈ R^{d+1} ──P0: Linear(d+1 → d0)──┐
                                                              ├─► concat ──► Tronco: L bloques
x_txt_f ──E_f: Embedding(V_f+2, e)──► e_f ∈ R^e (por campo) ──┘   [Linear → LN → GELU → Dropout 0.1]
                                                                  (residual en bloques 2..L)
                                                                        │
                                                                        ▼ h ∈ R^w
                                        ┌───────────────┬──────────────┴──────────┬────────────────┐
                                   Noul (w→1)    Choice_family (w→K)     Score CORN (u∈R^w, b1..b3)  Choice_ddos (w→4, solo CIC)
```

- Tronco: bloque $b$ = $\text{Dropout}_{0.1}(\text{GELU}(\text{LN}_b(\text{Linear}_b(\cdot))))$; bloque 1 mapea $\mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^w$, bloques 2..L son $w \to w$ con **residual**: $h_b = h_{b-1} + \text{bloque}_b(h_{b-1})$.
- **LayerNorm, no BatchNorm:** el entrenamiento de EMBER es streaming por shards (estadísticas de batch inestables), la inferencia incluye batch-1 (E4), y hay drift temporal entre train y test — LN es independiente del batch.
- Sin pesos preentrenados: se entrena desde cero (decisión congelada). La cabeza de texto NO es DistilBERT (esto es un clasificador tabular con embeddings lookup; DistilBERT es solo baseline E7/API).
- Inicialización: default de PyTorch (Kaiming-uniform en Linear, LN en (1,0)); **excepción:** biases de CORN $b_1{=}1, b_2{=}0, b_3{=}-1$ (prior ordinal razonable; CORN no los exige, solo acondiciona).

### 2.2 Dimensiones por escala

| escala | L | $w$ UNSW/CIC | $w$ EMBER | $d_0$ UNSW/CIC | $d_0$ EMBER | $e$ (embeddings) | presupuesto |
|---|---|---|---|---|---|---|---|
| Nano | 2 | 224 | 192 | 112 | 20 | 16 | ≤ 100K |
| Small | 3 | 576 | 576 | 288 | 96 | 32 | ~ 1M |
| Base | 4 | 1280 | 1280 | 512 | 384 | 64 | 5–10M |

Justificación de la frontera EMBER: con $d=2568$, la proyección cuesta $(2568{+}1)\cdot d_0 + d_0$. En Nano esto ya son 51,400 parámetros (53.8% del total): EMBER es **input-bound**. Reducir $d_0$ (20/96/384) y $w$ en Nano (192) mantiene el presupuesto sin estrangular el tronco. La progresión de profundidad L=2/3/4 y anchura 224→576→1280 es uniforme para UNSW/CIC. La alternativa de proyección por bloques (los 2568 features de thrember son histogramas por grupo semántico) queda como Q6.

### 2.3 Conteo de parámetros (verificado por script; Apéndice B)

Fórmulas por componente:
- $P_0$: $(d{+}1)\,d_0 + d_0$ (el +1 es el flag de fila no-finita).
- Embeddings: $\sum_f (V_f + 2)\,e$ (slots `<miss>`=0 y `<unk>`=1); solo UNSW (proto $V{=}133$, service $13$, state $9$).
- Tronco: $d_{\text{in}}\,w + w + (L{-}1)(w^2 + w) + L \cdot 2w$, con $d_{\text{in}} = d_0 + m\,e$ ($m$ = #campos de texto).
- Cabezas: Noul $w{+}1$; Choice $wK{+}K$; CORN $w{+}3$; ddos $4w{+}4$.

| escala | dataset | proj | txt | trunk | heads | **total** | presupuesto |
|---|---|---|---|---|---|---|---|
| Nano | UNSW | 4,592 | 2,576 | 87,360 | 2,702 | **97,230** | ✓ ≤100K |
| Nano | CIC | 8,848 | 0 | 76,608 | 3,152 | **88,608** | ✓ ≤100K |
| Nano | EMBER | 51,400 | 0 | 41,856 | 2,316 | **95,572** | ✓ ≤100K |
| Small | UNSW | 11,808 | 5,152 | 889,920 | 6,926 | **913,806** | ✓ ~1M |
| Small | CIC | 22,752 | 0 | 834,624 | 8,080 | **865,456** | ✓ ~1M |
| Small | EMBER | 246,720 | 0 | 724,032 | 6,924 | **977,676** | ✓ ~1M |
| Base | UNSW | 20,992 | 10,304 | 5,831,680 | 15,374 | **5,878,350** | ✓ 5–10M |
| Base | CIC | 40,448 | 0 | 5,585,920 | 17,936 | **5,644,304** | ✓ 5–10M |
| Base | EMBER | 986,880 | 0 | 5,422,080 | 15,372 | **6,424,332** | ✓ 5–10M |

Notas: (i) los conteos de heads asumen `split_strat` (CIC: Noul + Choice-8 + CORN + ddos-4); bajo `split_time` el head Choice de CIC tiene 6 candidatos (§3.3), −2·w parámetros; (ii) CIC/EMBER sin rama textual ($m{=}0$); (iii) cabezas: UNSW = Noul + Choice-10 + CORN; EMBER = Noul + Choice-11 (sin Score, S6).

---

## 3. Primitivas ↔ cabezas: mapeo y matemática de losses

### 3.1 Mapeo por dataset, candidatos congelados y gates derivados

**Mapeo primitivas ↔ cabezas** (F0 §2; implementado sobre `state.py`):

| dataset | Noul | Choice | Score | cabeza extra |
|---|---|---|---|---|
| UNSW | `is_attack` = 1[familia ≠ normal] | `choice_family`, 10 candidatos, unknown por **abstención** | CORN 0–3 | — |
| CIC | `is_attack` | `choice_family`, 8 candidatos (6 bajo `split_time`, D12), unknown por **abstención** | CORN 0–3 | `choice_ddos_vector`, 4 candidatos, subset DoS/DDoS, solo `split_strat` |
| EMBER | `is_malicious` (`label_binary`) | `choice_family`, 11 candidatos, **unknown entrenado** (reglas S6) | — (sin GT de severidad, S6) | — |

**Órdenes congelados de candidatos** (idénticos a `SOURCE_REGISTRY`):

- UNSW: `normal, dos, recon, exploitation, shellcode, backdoor, analysis, fuzzing, generic, worms`
- CIC: `normal, dos_ddos, recon, bruteforce, webattack, botnet, infiltration, exploitation`
- ddos_vector: `normal, tcp_flood, http_dos, slowloris`
- EMBER (censo top-10 + unknown último): `berbew, cosmu, wacatac, sfone, upatre, grandoreiro, glupteba, expiro, sinowal, flystudio, unknown`

**Rubrik de severidad** (familia → nivel; verificada por suma exacta sobre train):

- UNSW: normal→0; generic→1, fuzzing→1, recon→1; dos→2, analysis→2; exploitation→3, shellcode→3, backdoor→3, worms→3.
  (Train: nivel 0 = 47,600; 1 = 58,373; 2 = 12,124; 3 = 30,942.)
- CIC: normal→0; recon→1; dos_ddos→2, bruteforce→2, webattack→2; botnet→3, infiltration→3, exploitation→3.
  (Train strat: nivel 0 = 1,591,168; 1 = 111,251; 2 = 277,692; 3 = 1,409.)

**D8 — nombre del candidato DDoS.** El label `DDoS` del viernes de CIC define un flood TCP volumétrico (paquete fwd medio 7.4 B, mediana de duración 1.9 s, PSH 0.45, puerto 80) y NO un handshake SYN real: `SYN Flag Count = 0` en el **100%** de esos flujos (artefacto de CICFlowMeter). El candidato se nombra **`tcp_flood`** con definición operacional "flood TCP volumétrico"; llamarlo `SYN_Flood` reclamaría un mecanismo que los datos no exhiben. Mapping congelado: `DDoS→tcp_flood`, `DoS Hulk→http_dos`, `DoS GoldenEye→http_dos`, `DoS slowloris→slowloris`, `DoS Slowhttptest→slowloris`, `BENIGN→normal`.

**Subconjunto `choice_ddos_vector`** (filas Wednesday + Friday-DDoS con label_full ∈ {DDoS, DoS*, BENIGN}): 918,437 filas de 2,830,743. Train `split_strat`: normal 376,380 / tcp_flood 89,598 / http_dos 169,034 / slowloris 7,850 (val: 80,572 / 19,216 / 36,181 / 1,706). El head se entrena solo con filas del subconjunto (máscara) y evalúa igual.

**D9 — gates del controlador (G=1 entrenado + 2 derivados).** F0 declaró 3 gates. v0.1 entrena solo Noul; los otros dos son **políticas de referencia** del controlador, derivadas de salidas calibradas, sin parámetros entrenados:

1. $g_{\text{attack}}$: $p^{\text{cal}}_{\text{mal}} \ge \tau_{\text{det}}$ (default 0.5) — el gate entrenado.
2. $g_{\text{drop}}$ (requiere descarga inmediata): $P(\text{sev} \ge 2 \mid x) = C_2(x) \ge \tau_{\text{drop}}$ (default 0.5) — la acumulada del head CORN.
3. $g_{\text{recon}}$ (recon de baja urgencia): $p^{\text{cal}}_{\text{choice}}(\text{recon}) \ge \tau_{\text{recon}}$ (default 0.5) — masa del head Choice.

*Deducción (por qué son márgenes exactas):* por la rubrik, $\text{sev} \ge 2 \Rightarrow \text{familia} \ne \text{normal} \Rightarrow$ ataque, luego $P(\text{sev}{\ge}2 \wedge \text{attack} \mid x) = C_2(x)$; y $\text{familia}=\text{recon} \Rightarrow \text{sev}=1$, luego $P(\text{recon} \wedge \text{low} \mid x) = p_{\text{choice}}(\text{recon} \mid x)$. Ninguna requiere suponer independencia entre cabezas, y ambas tienen ground truth medible (E3 mide su ECE contra $y_{\text{sev}} \ge 2$ y $y = \text{recon}$). La composición general (gates no reducibles por rubrik) queda en Q1.

### 3.2 Noul — BCE sin ponderar

Cabeza: $z(x) = \mathbf{u}_n^\top h(x) + b_n \in \mathbb{R}$, salida cruda $p_{\text{raw}} = \sigma(z)$.

$$\mathcal{L}_{\text{noul}} = -\frac{1}{N} \sum_{i=1}^{N} \big[\, y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \,\big]$$

**Proposición 1 (por qué sin pesos).** Sea la variante ponderada $\mathcal{L} = -\frac{1}{N}\sum_i [\alpha_+\, y_i \log p_i + \alpha_-\,(1-y_i)\log(1-p_i)]$. Minimizando punto a punto sobre $p(x) \in (0,1)$ con $\pi(x) = P(y{=}1 \mid x)$:

$$\frac{p^*(x)}{1-p^*(x)} = \frac{\alpha_+}{\alpha_-} \cdot \frac{\pi(x)}{1-\pi(x)} \;\;\Longrightarrow\;\; \mathrm{logit}(p^*) = \mathrm{logit}(\pi) + \log \tfrac{\alpha_+}{\alpha_-}$$

El ponderar produce un **shift aditivo constante** en el espacio del logit. La temperatura $z \mapsto z/T$ solo cambia la pendiente; no puede revertir un desplazamiento aditivo. Como la calibración del contrato es puramente $T$ (§5), Noul se entrena **sin pesos** para que $\sigma(z)$ apunte al posterior verdadero y $T$ corrija solo la sharpness residual. Los desbalances son moderados (UNSW 68.06%, CIC 19.70%, EMBER 50.00%) y el AUROC es invariante a la prevalencia; la honestidad probabilística frente a la prevalencia del dataset es lo que F0 exige ("$P(y{=}1 \mid \text{noul}=p) = p$"). El ajuste a la prevalencia de despliegue (Q10) es transformación del controlador, no del entrenamiento.

### 3.3 Choice — unknown híbrido, pesos sqrt y de-tilt

Cabeza: $z(x) \in \mathbb{R}^K$ (Linear $w \to K$), candidatos con orden congelado.

**D1 — mecanismo del unknown.**

- **(a) EMBER: unknown como clase entrenada** ($c_{11}$, K=11). Las reglas S6 generan GT: benignos → unknown, malicious sin familia → unknown, malicious fuera del top-10 → unknown. Train: 2,195,408 filas de unknown (79.5%); val: 288,910 (80.2%); test: 592,254 (81.5%); challenge: 6,236 de 6,315 (98.8%). El ~80% no es un colapso: es la prevalencia real y se reporta como honestidad. La abstención pura sería *peor* aquí: sin muestras de unknown no existe la frontera "benigno vs malware atribuible", y forzar atribución sobre benignos fabricaría precision falsa.
- **(b) UNSW/CIC: unknown por abstención.** No existen filas sin familia nativa (censo completo), así que una clase unknown entrenada no tiene definición. En inferencia: $\hat c = \arg\max_k p^{\text{cal}}_k$; si $\max_k p^{\text{cal}}_k < \tau_{\text{abs}}$ (default 0.5, política del controlador) → **unknown**. La distribución calibrada suma 1 sobre los candidatos nativos; unknown es el resultado tipado de abstenerse, no una masa de probabilidad. E2 (LOAO) es la prueba de diseño: familias excluidas de train deben abstenerse mayoritariamente.
- El contrato F0 ("el conjunto de candidatos siempre incluye unknown") se satisface en ambos; lo que varía es la parametrización, dictada por la estructura del GT.

**Peso del riesgo de colapso (EMBER).** Sin ponderar, unknown concentra el 79.5% de la masa de gradiente de la CE y el modelo puede degenerar a "unknown para todo". La ponderación sqrt (abajo) la baja al **40.0%** — sigue siendo la clase dominante, pero las 10 familias conservan el 60% de la masa para definir la frontera de atribución. Se monitorea: recall por familia en val (E1) y FAR de atribución sobre benignos.

**Loss.** Sea $n_k$ el conteo de train de la clase $k$ (de la corrida) y

$$\omega_k = \frac{K \cdot n_k^{-1/2}}{\sum_{j=1}^{K} n_j^{-1/2}} \qquad \text{(media 1 sobre las clases; tabla exacta en §3.5)}$$

$$\mathcal{L}_{\text{choice}} = -\,\frac{\sum_i \omega_{y_i} \log \mathrm{softmax}(z_i)_{y_i}}{\sum_i \omega_{y_i}}$$

(media ponderada por batch; invariante a la escala global de $\omega$). El head `choice_ddos_vector` usa su propia $\omega^{(ddos)}$ (conteos del subconjunto) y solo filas del subconjunto.

**Proposición 2 (tilt y de-tilt).** Para riesgo poblacional con pesos $\omega$: minimizando $-\mathbb{E}[\sum_k \pi_k(x)\, \omega_k \log p_k]$ sobre el simplex, $p^*_k \propto \omega_k\, \pi_k(x)$. Corolario: el softmax óptimo cumple $z^*_k = \log \pi_k(x) + \log \omega_k + \text{const}$. En inferencia, el **de-tilt analítico**

$$z'_k = z_k - \log \omega_k$$

recupera el posterior verdadero (exacto en población; en muestra quedan residuos de estimación y misspecification). El de-tilt es **parte del contrato del modelo** (no un hiperparámetro): se aplica antes de ajustar $T$ (§5), y a TODAS las clases — en EMBER incluida unknown ($\omega_{\text{unknown}} = 0.1511$). La decisión ($\arg\max$) con de-tilt es la decisión bajo el posterior sin ponderar: la ponderación vive solo en el entrenamiento, la política (si se quiere un prior distinto) vive en el controlador.

**Candidate set por corrida (D12).** $\mathcal{C}_q$ = familias nativas con ≥ 1 fila en train bajo el esquema de la corrida. Consecuencias verificadas: bajo `split_time`, recon (0 filas train; 158,930 en test) y botnet (0; 1,966 en val) no son candidatas — sus filas solo pueden atribuirse mal o abstenerse, que es exactamente lo que el split temporal mide. En folds LOAO (§6.7) aplica igual: la familia excluida no es candidata (evita $\omega_k$ con $n_k = 0$, indefinida en la fórmula). EMBER y `split_strat` usan el set completo.

### 3.4 Score — regresión ordinal condicional (CORN)

**Comparación de parametrizaciones** (elige una; F0 §2.3 fija la *salida*, no la cabeza):

| método | pmf válida? | orden | params | observación |
|---|---|---|---|---|
| CE nominal (softmax 4 cols) | siempre | ninguno (ignora niveles) | $4w{+}4$ | trata benign y low como clases sin relación; más params, sin estructura |
| CE ordinal con soft-labels | depende del kernel | sí (suave) | $4w{+}4$ | sensible a la forma del kernel; targets cumulativos suaves arbitrarios |
| CORAL (biases monotónicos) | solo si $b_1 \ge b_2 \ge b_3$ | sí | $w{+}3$ | la restricción es suave: el entrenamiento puede violarla y la pmf sale negativa |
| **CORN (condicional)** | **por construcción** | **sí (exacto)** | $w{+}3$ | condicionales sigmoide; pmf válida sin restricciones (Shi, Cao & Raschka 2021) |
| Regresión + redondeo | no hay pmf | sí (en el valor) | $w{+}1$ | incompatible con la salida tipada de F0 (no hay $\pi_k$ que calibrar) ni con $\hat s = \sum_k k\,\pi_k$ |

**Elección (D3): CORN.** Misma dimensionalidad que CORAL pero la validez de la pmf es estructural: cualquier $(\mathbf{u}, b_1, b_2, b_3)$ produce una distribución sobre niveles con orden respetado por construcción.

**Matemática.** Cabeza: vector compartido $\mathbf{u} \in \mathbb{R}^w$ y biases $b_1, b_2, b_3$ (init $+1, 0, -1$). Condicionales:

$$q_k(x) = \sigma\!\big(\mathbf{u}^\top h(x) + b_k\big) = P(y \ge k \mid y \ge k-1, x), \qquad k \in \{1,2,3\}$$

Acumuladas y pmf (siempre válidas):

$$C_0(x) = 1, \quad C_k(x) = \prod_{j=1}^{k} q_j(x), \qquad \pi_0 = 1 - q_1, \quad \pi_1 = C_1 - C_2, \quad \pi_2 = C_2 - C_3, \quad \pi_3 = C_3$$

Loss (BCE por tarea, cada una normalizada por sus filas elegibles $y_i \ge k-1$):

$$\mathcal{L}_{\text{corn}} = \sum_{k=1}^{3} \frac{1}{N_{\ge k-1}} \sum_{i:\, y_i \ge k-1} \Big[ \mathbb{1}(y_i \ge k) \log q_k(x_i) + \mathbb{1}(y_i < k) \log\big(1 - q_k(x_i)\big) \Big]$$

Elegibles verificados (train): UNSW $N_{\ge 0} = 149{,}039$, $N_{\ge 1} = 101{,}439$, $N_{\ge 2} = 43{,}066$; CIC strat $1{,}981{,}520 / 390{,}352 / 279{,}101$. Bajo `split_time` (CIC) el nivel 1 queda sin filas en train (recon→test): las tareas degeneran a $N_{\ge 1} = N_{\ge 2} = 268{,}710$ con 34 positivos en $k{=}3$ — se documenta tal cual; es la realidad del split temporal.

**Decodificación (salida tipada de F0 §2.3, sin cambios):**

$$\hat s(x) = \sum_{k=0}^{3} k\, \pi_k(x) \in [0, 3] \qquad \text{(valor esperado)}, \qquad \text{modo}(x) = \arg\max_k \pi_k(x) \qquad \text{(nivel)}$$

La acumulada $C_2$ alimenta el gate derivado $g_{\text{drop}}$ (§3.1).

**Por qué sin pesos extra.** La partición condicional ya redistribuye la masa: una fila $y{=}3$ contribuye a las 3 tareas y una fila $y{=}0$ solo a la primera. En CIC strat una fila sev-3 lleva ~13× la masa de gradiente de una fila sev-0 ($5.0\text{e-}7$ vs $6.7\text{e-}6$ por fila). Cada tarea es un problema binario bien planteado con su propia prevalencia (task 3 de CIC: 1,409 positivos de 279,101 elegibles = 0.5%); ponderarla desplazaría la condicional estimada (misma estructura de sesgo que la Proposición 1, ahora dentro de la tarea) y la pmf resultante dejaría de ser la verdadera. La severidad alta se vigila con métricas que la exponen (per-level recall en E1, $M_{\text{val}}$ con MAE) y no adulterando la distribución.

**Observación estructural.** Severidad ≡ rubrik(familia): el GT de Score es un relabel determinístico del GT de Choice. El head CORN separado se mantiene por el contrato F0 (primitiva tipada) y como regularizador multi-task (E6-A2 lo mide); la alternativa derivada — empujar el posterior de Choice por la rubrik — es exacta y gratis, queda como Q4 para v0.2.

### 3.5 Estrategia de desbalance (fórmulas congeladas y tablas exactas)

**Fórmulas congeladas** (una sola familia de pesos, aplicada solo a Choice; Noul sin pesos por Prop. 1, Score sin pesos por §3.4):

$$\omega_k = \frac{K \cdot n_k^{-1/2}}{\sum_j n_j^{-1/2}}, \qquad \frac{1}{K}\sum_k \omega_k = 1, \qquad n_k = \text{conteo de train de la corrida}$$

- Exponente $-1/2$ (no $-1$ de inversa pura, no effective number of samples de Cui et al.): inversa pura hace que 8 filas de Heartbleed dominen a 1.59M de normal (razón 200,000:1) y desestabiliza; sqrt reduce la razón de masas de $n^1$ a $n^{1/2}$ (normal:infiltration en CIC: 63,647:1 → 252:1). E6-A4 abla contra $\omega \equiv 1$.
- Media 1: mantiene la escala global del loss estable entre cabezas/datasets (interacción limpia con §3.6).
- Conteos de train de la corrida (recalculados si cambia el split o el fold, D12); los pesos exactos se **serializan** en el checkpoint (§8) porque el de-tilt los necesita en inferencia.

**Tabla exacta — UNSW `choice_family` (train):**

| clase | n | $\omega_k$ | | clase | n | $\omega_k$ |
|---|---|---|---|---|---|---|
| normal | 47,600 | 0.2067 | | analysis | 1,700 | 1.0939 |
| dos | 10,424 | 0.4417 | | backdoor | 1,484 | 1.1708 |
| recon | 8,917 | 0.4776 | | shellcode | 963 | 1.4534 |
| exploitation | 28,384 | 0.2677 | | worms | 111 | 4.2808 |
| fuzzing | 15,456 | 0.3628 | | generic | 34,000 | 0.2446 |

**Tabla exacta — CIC `choice_family` (train, `split_strat`):**

| clase | n | $\omega_k$ | | clase | n | $\omega_k$ |
|---|---|---|---|---|---|---|
| normal | 1,591,168 | 0.0102 | | webattack | 1,526 | 0.3292 |
| dos_ddos | 266,482 | 0.0249 | | botnet | 1,376 | 0.3467 |
| recon | 111,251 | 0.0386 | | infiltration | 25 | 2.5723 |
| bruteforce | 9,684 | 0.1307 | | exploitation | 8 | 4.5473 |

**Tabla exacta — CIC `choice_ddos_vector` (train, `split_strat`):**

| clase | n | $\omega_k$ |
|---|---|---|
| normal | 376,380 | 0.3489 |
| tcp_flood | 89,598 | 0.7150 |
| http_dos | 169,034 | 0.5206 |
| slowloris | 7,850 | 2.4156 |

**Tabla exacta — EMBER `choice_family` (train):**

| clase | n | $\omega_k$ | | clase | n | $\omega_k$ |
|---|---|---|---|---|---|---|
| berbew | 222,302 | 0.4750 | | glupteba | 38,180 | 1.1462 |
| cosmu | 39,526 | 1.1265 | | expiro | 33,382 | 1.2258 |
| wacatac | 37,658 | 1.1541 | | sinowal | 34,488 | 1.2059 |
| sfone | 43,720 | 1.0711 | | flystudio | 32,344 | 1.2453 |
| upatre | 42,294 | 1.0890 | | **unknown** | 2,195,408 | **0.1511** |
| grandoreiro | 40,698 | 1.1101 | | | | |

Masa de gradiente de unknown en EMBER: 79.5% sin ponderar → **40.0%** con sqrt (share $= \omega_k n_k / \sum_j \omega_j n_j \propto \sqrt{n_k}$).

**Micro-clases.** CIC: infiltration 25/6/5 y exploitation (Heartbleed) 8/2/1 (train/val/test); UNSW: worms 111/19/44. Protocolo congelado: (i) pesos sqrt sin casos especiales; (ii) macro-métricas las pesan como a cualquiera (por eso macro-F1 y no accuracy); (iii) LOAO evalúa cada familia sobre val+test **agrupados** (infiltration 11 filas, Heartbleed 3) reportando siempre $n$; (iv) nada de oversampling sintético (flujos generados de IDS introducen artefactos de distribución; los pesos bastan y E2 lo mide).

### 3.6 Multi-task weighting — Kendall homeostático modificado

Las cabezas activas por dataset difieren (2 en EMBER, 3 en UNSW, 4 en CIC) y sus losses arrancan en escalas distintas ($\mathcal{L}_{\text{choice}}(0) \approx \ln K$: 2.30/2.08/2.40; $\mathcal{L}_{\text{noul}}(0) \approx 0.69$; $\mathcal{L}_{\text{corn}}(0) \approx 0.69$). Se usa incertidumbre homeostática de Kendall (2018) con dos modificaciones propias:

$$\mathcal{L} = \sum_{q} a_q \big[ e^{-s_q}\, \tilde{\mathcal{L}}_q + s_q \big]$$

- $s_q \in \mathbb{R}$ aprendible (log-varianza), **init 0** → $\lambda_q = e^{-s_q}$ arranca en 1 (homogéneo; sin ventaja inicial a ninguna cabeza).
- $\tilde{\mathcal{L}}_q$ = loss de la cabeza $q$ sobre sus filas elegibles del batch (§3.2–3.4; ddos solo filas del subconjunto).
- $a_q \in \{0,1\}$ = **máscara de actividad**: 1 si el batch tiene ≥ 1 fila elegible para $q$, si no la cabeza aporta 0 ese paso (y su $s_q$ no deriva hacia arriba por inactividad). En CIC el 32.5% del train pertenece al subconjunto ddos → con batches ≥ 1024 prácticamente todo batch es activo.
- El término $+s_q$ es el regularizador que impide el colapso $\lambda_q \to 0$ (homeostasis).
- Trayectoria de $\lambda_q$ se registra por corrida (diagnóstico de dominancia).

Rechazos documentados: suma homogénea (dominación de Choice por escala inicial), GradNorm (requiere backward extra por cabeza y es frágil bajo máscaras de actividad), PCGrad (conflictos de gradiente entre cabezas del mismo estado rara vez importan a estas escalas). Caveat honesto: la incertidumbre de Kendall es una heurística (asume ruido Gaussiano homocedástico en la regresión original), no un posterior bayesiano; E6-A2 (single-head con presupuesto igual) es la verificación empírica.

---

## 4. Encodings de entrada

### 4.1 x_num — sanitización y normalización train-only (política exacta)

Pipeline por columna $j$ (stats SOLO de train; aplicado igual a train/val/test/challenge):

1. **No-finitos:** $\tilde x_{ij} = x_{ij}$ si $x_{ij} \in \mathbb{R}$; si es NaN/±inf → $m_j$ = mediana de los valores finitos de train (fallback 0 si la columna no tiene finitos). Flag de fila: $r_i = \mathbb{1}[\text{la fila tenía algún valor no-finito}]$.
2. **Signed log1p:** $v_{ij} = \mathrm{sign}(\tilde x_{ij}) \cdot \log(1 + |\tilde x_{ij}|)$ (comprime el rango dinámico; simétrico para negativos, p.ej. el centinela −1 de `Init_Win_bytes`).
3. **Robust scale:** $u_{ij} = (v_{ij} - \mathrm{med}_j) / \max(\mathrm{IQR}_j, 10^{-8})$, con mediana e IQR (q75−q25) calculados sobre la columna **transformada** de train. Columna constante → IQR 0 → escala efectiva 1 → $u \approx 0$.
4. **Clip:** $u_{ij} \leftarrow \mathrm{clip}(u_{ij}, -8, +8)$.
5. **Entrada del modelo:** $[u_{i\cdot}\,;\, r_i] \in \mathbb{R}^{d+1}$ — el flag es la dimensión extra del conteo de §2.3.

Precisión: stats en float64, aplicación en float32. **Serializado** en el checkpoint: $m \in \mathbb{R}^d$, $\mathrm{med} \in \mathbb{R}^d$, $\mathrm{IQR} \in \mathbb{R}^d$, clip = 8.

Cifras verificadas de filas no-finitas: **UNSW 0** (de 257,673); **CIC 2,867** (train 2,021 / val 434 / test 412; todas con duración 0 — flujos degenerados de CICFlowMeter); **EMBER 0** (verificado en `part_00_train` y `challenge`). La política se aplica uniforme aunque UNSW/EMBER no la ejerzan hoy.

Justificaciones: **vs z-score** — la media/std quedan contaminadas por colas pesadas y por los propios no-finitos; mediana/IQR tienen breakdown point 25%. **vs quantile transform** — serializa un knot-set por feature, rompe los empates masivos de features discretas (flag counts con decenas de miles de empates los dispersaría en intervalos) y añade opacidad; log1p+IQR es monótono, transparente y barato. **Imputación mediana vs centinela** — un centinela grande explota bajo linear+LN; la mediana es inerte y el flag preserva el evento "flujo degenerado".

### 4.2 x_txt — embeddings lookup (no hashing)

Por campo $f$ (solo UNSW: `proto`, `service`, `state`): tabla $E_f \in \mathbb{R}^{(V_f + 2) \times e}$ con slots reservados `<miss>`=0 y `<unk>`=1; vocabulario $V_f$ = valores distinctos de **train**, normalización casefold+strip al ajustar. Dimensión $e$ = 16/32/64 por escala. La entrada del tronco concatena $[P_0(u;r)\,;\,E_{\text{proto}}\,;\,E_{\text{service}}\,;\,E_{\text{state}}]$ (la "fusion MLP" de F0 §3 es simplemente la primera capa del tronco).

Vocabularios verificados (train): proto 133, service 13, state 9. `state` tiene 11 valores en total — los 2 restantes solo aparecen en test, así que `<unk>` no es un reserved de cortesía sino un caso real medible (aparece en inferencia). **Hashing rechazado:** los vocabularios son cerrados y diminutos (≤ 133); el hashing introduce colisiones que aquí son pérdida pura, y el lookup serializa un vocabulario interpretable.

### 4.3 EMBER — proyección 2568 → d0

Proyección lineal única $(2569) \to d_0$ con $d_0$ = 20 / 96 / 384 (Nano/Small/Base). EMBER es **input-bound**: en Nano la proyección es el 53.8% del presupuesto (51,400 de 95,572). Los 2568 features de thrember llegan pre-estructurados (histogramas y conteos por grupo semántico); a estas escalas una proyección lineal es suficiente. La alternativa estructurada por bloques (12 grupos de features con proyecciones separadas) queda como Q6 para v0.2.

### 4.4 Decisión sobre re-materialización de CIC (pendiente de este doc — CERRADA)

**No se re-materializa CIC para v0.1.** El modelo v0.1 entrena sobre el parquet MLCVE ya materializado (d_num=77) y **sin rama textual**: en MLCVE `protocol` es None (el feature numérico ya codifica el protocolo). Razones: (i) MLCVE es el estándar de facto de la literatura sobre CIC-IDS2017 — re-materializar rompería la comparabilidad de E1 con los baselines publicados; (ii) `protocol` del parquet enhanced tiene 3 valores con 99.9% de la masa en {tcp, udp} — información ya redundante con Destination Port y flag counts; (iii) el parquet enhanced (S1) queda congelado como **sustrato de E5** para casar alertas Suricata con flujos por 5-tupla + timestamp, y sus 77 features comparten orden con `CIC_FEATURE_COLUMNS`, así que el modelo entrenado aplica directo en inferencia. La re-materialización con protocol + Timestamp real solo se considera en v0.2 si se quiere un `split_time` con marcas temporales reales (Q7).

---

## 5. Calibración

### 5.1 Protocolo exacto (congelado)

Post-entrenamiento, con $\theta$ congelado, sobre **val** (nunca test), una temperatura escalar por cabeza:

1. **Noul:** buscar $T > 0$ que minimiza $\mathrm{NLL}_{\text{val}} = -\frac{1}{N}\sum_i [y_i \log \sigma(z_i/T) + (1-y_i)\log(1-\sigma(z_i/T))]$.
2. **Choice:** aplicar el de-tilt analítico $z'_k = z_k - \log \omega_k$ (Prop. 2; parte del contrato, no un fitting) y luego ajustar $T$ minimizando la NLL de $\mathrm{softmax}(z'/T)$.
3. **Score (CORN):** un único $T$ escala los tres logits condicionales, $q_k^{(T)} = \sigma((\mathbf{u}^\top h + b_k)/T)$, y la pmf $\pi^{(T)}$ se recompone; $T$ minimiza la NLL de $\pi^{(T)}_{y_i}$.
4. **Búsqueda determinista:** grid log-uniforme de 200 puntos en $[0.05, 20]$ + refinamiento por sección dorada en el mejor intervalo hasta tolerancia relativa $10^{-3}$. Determinista → sin semilla extra.
5. **Criterio NLL, no ECE:** la NLL es una proper scoring rule; el ECE depende del binning (varianza alta con pocas filas por bin). El ECE 15 bins se **reporta** en E3 pero no participa en la selección de $T$.
6. **Un escalar por cabeza.** Vector scaling rechazado: con micro-clases (val: Heartbleed 2, infiltration 6 en CIC) un $T$ por clase sobreajusta; Guo et al. 2017 encuentran que el escalar iguala al vector con val grandes; y 1 parámetro/cabeza mantiene el contrato mínimo.
7. **Doble uso de val** (early stopping §6.4 + $T$): aceptado — $T$ es 1 parámetro por cabeza, la fuga es despreciable (Q5 documenta el fallback si un reviewer lo exige).
8. $T = 1$ (sin calibración) es el ablation E6-A3; los mismos pasos corren para él.

### 5.2 Interacción con cada primitiva

- **Noul:** la salida desplegada es $p^{\text{cal}} = \sigma(z/\hat T)$ y el umbral del controlador opera sobre ella: $p^{\text{cal}} \ge \tau \Leftrightarrow z \ge \hat T\,\sigma^{-1}(\tau)$ — la política vive en el espacio de probabilidad. Las métricas threshold-free (AUROC, FPR@95TPR) son invariantes ante $T$ (monótona); el valor de la calibración está en la semántica absoluta (E3, políticas τ, gates derivados).
- **Choice (abstención):** la distribución calibrada suma 1 sobre candidatos nativos; unknown se decide por $\max_k p^{\text{cal}}_k < \tau_{\text{abs}}$. El de-tilt se aplica ANTES de $T$; el orden importa (después dejaría residuos del tilt que $T$ absorbería mal).
- **Choice (EMBER):** el de-tilt incluye a unknown; el resultado es un posterior honesto sobre las 11 clases.
- **Score:** $T$ actúa sobre las condicionales y se propaga exactamente a la pmf, a $\hat s$ y a $C_2$ (gate $g_{\text{drop}}$).
- **Gates derivados (D9):** al ser funciones exactas de salidas calibradas, heredan calibración; E3 mide su ECE contra sus GT ($y_{\text{sev}} \ge 2$; $y = \text{recon}$).

### 5.3 Anti-strawman: baselines calibrados igual

- **Probabilísticos** (XGBoost, Random Forest, MLP tabular, DistilBERT): mismo protocolo — logits/probs → grid NLL en el MISMO val, 1 $T$ (multiclase) o $T$ binario. Para RF (probabilidades discretas conocidas por su mala calibración) esto es especialmente importante.
- **Score puro sin semántica probabilística** (Kitsune, score RMSE): familia afín logística $p = \sigma((s - b)/T)$ ajustada en val (2 parámetros, análogo a Platt) — la calibración más fuerte que su salida admite.
- **Suricata (Fase 1):** no emite probabilidades; entra en E1 como punto operativo (TPR/FPR de su alerting), no como curva calibrable. Documentado explícitamente para no parecer evasión.
- Todos: fitted en val, reportado raw y calibrado en E3; mismos datos, misma búsqueda.

---

## 6. Receta de entrenamiento (Modal)

### 6.1 Semillas y determinismo

`SEED = 42` en: `torch.manual_seed`, `numpy`, `random`, semillas CUDA; permutaciones por epoch derivadas (`seed = 42 + epoch`); dropout y workers con `worker_init_fn = 42 + worker_id + epoch`. `torch.use_deterministic_algorithms(True)` en modo best-effort (documentado: algunos kernels AMP de CUDA no son bit-exactos entre ejecuciones; mismo seed + mismo tipo de GPU → misma trayectoria en la práctica). El manifest serializa versiones efectivas.

### 6.2 Loaders

- **UNSW / CIC:** tensores en RAM (CIC train: $1.98\text{M} \times 78 \times 4\,\text{B} \approx 620$ MB float32) + permutación completa por epoch.
- **EMBER (streaming por shards):** train = 46 parquets de 60,000 filas (616 MB decodificados c/u; 65 shards totales = 46 train + 6 val + 12 test + 1 challenge). Por epoch: permutación de shards (seed `42+epoch`) → para cada shard: decodificación completa, shuffle intra-shard, emisión de batches; **prefetch de 1 shard** en thread aparte → pico CPU ≈ 1.3 GB (fallback documentado: half-shard si la instancia tuviera < 4 GB libres). Nota honesta: esto es un shuffle a dos niveles (shard + intra-shard), no un shuffle global exacto; a 60K filas/shard el efecto práctico es despreciable y se documenta como aproximación.
- `drop_last = True` (steps/epoch estables y deterministas); val se evalúa entera cada epoch.

### 6.3 Hiperparámetros congelados

| parámetro | Nano | Small | Base |
|---|---|---|---|
| lr (AdamW) | 3e-3 | 1e-3 | 3e-4 |
| batch UNSW / CIC / EMBER | 256 / 1024 / 2048 | 512 / 2048 / 4096 | 1024 / 4096 / 8192 |
| epochs máx UNSW / CIC / EMBER | 40 / 25 / 12 | 30 / 20 / 10 | 25 / 15 / 8 |
| GPU | T4 | T4 | L4 |

Comunes: AdamW $\beta=(0.9, 0.999)$, eps $10^{-8}$; **weight decay 1e-2 solo en weight matrices** (excluye LN, biases, embeddings, $s_q$ — tokens raros como worms no deben encogerse); schedule linear warmup del 5% de los steps + cosine hasta 0 al horizonte; **clip de gradiente global 1.0**; dropout 0.1 en cada bloque del tronco; patience 4 epochs.

Steps/epoch (`drop_last`): UNSW 582/291/145; CIC 1,935/967/483; EMBER 1,347/673/336 (Nano/Small/Base).

**AMP:** bf16 en L4 (sin GradScaler); fp16 + GradScaler (init 65536, dinámico) en T4. Pérdidas en fp32 bajo autocast; $s_q$ y estado del optimizador en fp32.

**Memoria (estimación, verificada por fórmula):** pesos+momentos ≈ 12 B/param (Base EMBER 6.4M → 77 MB); activaciones peores ≈ batch × ($d{+}1 + L \cdot w$) × 2 B (EMBER Base: 8192 × 7689 × 2 ≈ 126 MB). Pico total < 2 GB en todas las corridas → cualquier GPU de 16 GB sobra; el tamaño lo dicta el throughput, no la memoria.

### 6.4 Early stopping y selección ($M_{\text{val}}$)

Evaluación al final de cada epoch sobre val; checkpoint `best.pt` por $M_{\text{val}}$ + `last.pt`; patience 4 epochs sin mejora; restore best; luego job de calibración (§5) que escribe el artefacto final.

$$M_{\text{val}} = \text{media de las métricas de las cabezas activas:} \quad m_{\text{noul}} = \text{AUROC}_{\text{val}}, \;\; m_{\text{choice}} = \text{macro-F1}_{\text{val}}, \;\; m_{\text{score}} = 1 - \mathrm{MAE}(\hat s, y)/3, \;\; m_{\text{ddos}} = \text{macro-F1 subset}$$

- UNSW: $\frac{1}{3}(m_{\text{noul}} + m_{\text{choice}} + m_{\text{score}})$ · CIC: $\frac{1}{4}(\dots + m_{\text{ddos}})$ · EMBER: $\frac{1}{2}(m_{\text{noul}} + m_{\text{choice}})$ (sobre las 11 clases incl. unknown).
- macro-F1 sin abstención en selección (τ_abs es política del controlador; la selección queda policy-free); AUROC threshold-free; el val de EMBER es futuro temporal → $M_{\text{val}}$ ya está expuesto al drift, que es lo que queremos seleccionar.

### 6.5 GPUs, tiempos y costos (estimaciones de orden; facturación real al correr)

| corrida | GPU | min/epoch est. | epochs máx | costo est. |
|---|---|---|---|---|
| UNSW ×3 escalas | T4 | 1 / 2 / 4 | 40/30/25 | < $1.5 total |
| CIC strat ×3 | T4/T4/L4 | 10 / 18 / 28 | 25/20/15 | ~ $6–9 |
| CIC time (Small) | T4 | 18 | 20 | ~ $1.5–2.5 |
| EMBER ×3 | T4/T4/L4 | 17 / 25 / 50 | 12/10/8 | ~ $10–12 |
| LOAO UNSW 9×Small | T4 | ~2 | 20 | ~ $1.5 |
| LOAO CIC 7×Small | T4 | ~10–20 | 20 | ~ $4–7 |

**Total ≈ $20–35** (precios Modal de referencia T4 ≈ $0.59/h, L4 ≈ $0.80/h). Early stopping típicamente recorta 30–50% del horizonte. Dentro del presupuesto de F0 §6; los baselines (XGBoost/RF/MLP en CPU, DistilBERT aparte) no están en esta tabla.

### 6.6 Corridas principales (matriz congelada)

10 corridas: UNSW {Nano,Small,Base} (split oficial) · CIC {Nano,Small,Base} `split_strat` + Small `split_time` · EMBER {Nano,Small,Base} (temporal). `split_time` solo en Small (D12: variante de robustez; el grid completo solo si el presupuesto sobra). Jobs Modal: `fit(ds, escala, split)` → `calibrate(...)` → `export(...)`; idempotentes por artefacto (si existe y el config coincide, skip); logs a Volume.

### 6.7 Corridas LOAO (E2)

UNSW: 9 folds (una por familia de ataque: dos, recon, exploitation, shellcode, backdoor, analysis, fuzzing, generic, worms) a **Small**. CIC: 7 folds (dos_ddos, recon, bruteforce, webattack, botnet, infiltration, exploitation) a **Small**, `split_strat`. EMBER sin LOAO (el challenge evasivo + los cohorts de drift son su prueba). En cada fold: la familia sale de train Y val; el candidate set se recalcula (D12); evaluación sobre las filas val+test de la familia (agrupadas; con $n$ reportado: p.ej. infiltration 11, Heartbleed 3, worms 63).

---

## 7. Hooks de evaluación E1–E7 (qué expone el modelo)

El modelo expone, para cualquier split y cualquier corrida: logits crudos, salidas calibradas por cabeza y las salidas tipadas de §8. Definiciones congeladas:

**E1 — detección/clasificación.** Sobre Noul calibrado: FPR@95TPR **micro** (positivos = todas las filas de ataque; $\tau^*$ = percentil 5 empírico de los scores calibrados positivos; $\mathrm{FPR} = P(p \ge \tau^* \mid \text{benign})$) y **macro** (por familia $f$: $\tau_f$ = p5 de la familia; $\mathrm{FPR}_f$; media + por familia). Además AUROC, PR-AUC, macro-F1 calibrado (Choice), F1 por clase, co-reportando la latencia de E4. Suricata entra como punto operativo. Baselines con el mismo protocolo.

**E2 — zero-day (LOAO, §6.7).** Por fold: (a) AUROC de Noul con las filas de la familia como positivos y benignos val+test como negativos; (b) **%unknown** sobre filas de la familia (abstención con τ_abs; en EMBER, predicción unknown); (c) ECE de Noul en esas filas (calibración OOD); (d) tasa de **falsa atribución** (filas atribuidas a un candidato nativo). EMBER: challenge 6,315 (98.8% unknown GT) + cohorts de drift (sfone 43,720→132, sinowal 34,488→18, grandoreiro 40,698→226; berbew crece 222,302→92,086). Nano y Base en 2 folds de UNSW para la interacción con escala.

**E3 — calibración.** Por cabeza y por gate derivado: reliability diagrams 15 bins equal-width, ECE, MCE, Brier, raw vs calibrado. Definiciones: Noul → bins sobre $p^{\text{cal}}$, accuracy $= y$; Choice → bins sobre $\max_k p^{\text{cal}}_k$, accuracy $= [\arg\max = y]$; Score → bins sobre $\max_k \pi_k$, accuracy $= [\text{modo} = y]$ (secundario: reliability del valor esperado, bins sobre $\hat s$).

**E4 — latencia.** Tres modos: (i) CPU batch-1 en máquina de referencia (1 hilo fijado, TorchScript; 100 warmup + 1,000 mediciones; p50/p95/p99); (ii) GPU batch-1 y batch-64 throughput (T4/L4); (iii) cold-start de artefacto. Estimaciones de partida: Nano ~20–80 µs, Small ~150–500 µs, Base ~1–3 ms CPU batch-1. `predict_single` sin maquinaria de batch es parte del contrato (§8).

**E5 — cascada.** `predict_batch` sobre los flujos del parquet **enhanced** (mismo orden de las 77 features; protocol ignorado en v0.1, §4.4), flujo a flujo y por lotes; el harness casa alertas Suricata ↔ flujos por 5-tupla + ventana temporal. Métricas del paper: % de flujos resuelto por F1 por clase de ataque, composición de la zona gris, accuracy combinado vs F1-solo vs F2-solo, latencia E2E.

**E6 — ablations** (a Small; A1 solo UNSW; A3/A4/A5 en los 3 datasets):
A1 sin rama textual (embeddings a cero) · A2 sin multi-task (modelos single-head con presupuesto total igual, anchura ajustada) · A3 sin calibración ($T=1$) · A4 pesos uniformes $\omega \equiv 1$ · A5 unknown off (EMBER: 10-way sin unknown; UNSW/CIC: $\tau_{\text{abs}} \to 1$) · A6 dropout 0.

**E7 — serializer (opcional).** `serialize_state_for_prompt(S)` → JSON canónico ≤ 2 KB (orden de claves determinista) + pregunta en formato Decision-1; el contrato §8 garantiza que VX-One y el baseline API consumen el mismo estado y las mismas salidas tipadas.

---

## 8. Contrato de inferencia y formato de checkpoint

### 8.1 API tipada

```python
class VXOneEngine:
    def __init__(self, artifact: str) -> None: ...          # .pt, self-contained
    def predict(self, x_num: Float32[d_num],
                x_txt: Mapping[str, str | None]) -> Decision: ...      # batch-1, sin maquinaria de batch (E4)
    def predict_batch(self, states: Sequence[State]) -> list[Decision]: ...

@dataclass(frozen=True)
class ChoiceResult:
    candidate: str                    # ∈ candidatos ∪ {"unknown"} (argmax, o abstención/unknown entrenado)
    probs: Mapping[str, float]        # calibradas, de-tilted; suma 1 sobre candidatos NATIVOS
    max_prob: float                    # para auditar la abstención (max_k p_cal)

@dataclass(frozen=True)
class ScoreResult:
    expected: float                   # ŝ = Σ k·π_k ∈ [0, 3]
    probs: Mapping[str, float]        # π calibrada sobre {"benign","low","high","critical"}

@dataclass(frozen=True)
class Decision:
    source: str                        # "unsw" | "cic" | "ember"
    noul: Mapping[str, float]           # {"is_attack" | "is_malicious": p_cal}
    choice: Mapping[str, ChoiceResult]  # {"family": ..., "ddos_vector": ...} (ddos solo CIC/strat)
    score: ScoreResult | None           # None en EMBER (S6)
    policy: Mapping[str, bool]          # gates derivados de §3.1 con los τ default serializados
```

Invariantes: `candidate == "unknown"` ⟺ (head abstención y `max_prob < τ_abs`) o (head trained y unknown ganó); `probs` nunca contiene "unknown" como masa; `policy` se deriva de lo anterior (recomputable, sin estado oculto).

### 8.2 Checkpoint (`.pt`, self-contained — la inferencia no necesita datos de train)

```
vxone_{source}_{escala}_{split}.pt
├── state_dict            # pesos del modelo (feature stage + embeddings + tronco + cabezas)
├── feature_stage         # {impute_median f32[d], center f32[d], scale f32[d], clip: 8.0}
├── vocab                 # {campo: {token: slot}} con <miss>=0, <unk>=1  (solo UNSW)
├── head_meta             # por cabeza: {candidates (orden congelado), class_weights f32[K],
│                         #  mode: "abstain"|"trained_unknown", temperature: T̂, tau_abs, tau defaults}
├── config                # hiperparámetros exactos de §6.3 + seed + dims
└── manifest              # {torch, pyarrow, numpy versions, epochs_run, best_M_val, fecha, git sha}
```

Se serializa TODO lo que la inferencia necesita: pesos, normalizadores, vocabularios, **pesos de clase** (el de-tilt los requiere), **temperaturas**, τ defaults. Export secundario para E4: TorchScript (trace con input de ejemplo; fallback a eager si el trace falla) y ONNX opcional.

---

## 9. Preguntas abiertas (con propuesta por defecto)

Marcadas **[DELEGAR]** las de naturaleza supercompleja que propongo enviar a un modelo superior para una segunda opinión; las demás son experimentos v0.2 o decisiones de política.

- **Q1 [DELEGAR] Composición de la calibración.** Los gates v0.1 son márgenes exactas de cabezas calibradas (deducción rubrik, §3.1), pero políticas generales (AND/OR entre cabezas sin reducción por rubrik) no tienen garantía de ECE compuesto. ¿Existe alguna cota útil bajo temperature scaling por cabeza? **Default:** sin claim formal; E3 mide empíricamente los gates compuestos; conformal como envoltura v0.2.
- **Q2 [DELEGAR] Conformal bajo drift temporal.** Split-conformal sobre val asume exchangeability; EMBER (semanas) y CIC `split_time` la rompen. ¿Weighted conformal bajo covariate shift es defendible aquí con shift detectable pero sin densidades? **Default:** split-conformal opcional en Noul con caveat de cobertura explícito en el paper.
- **Q3 [DELEGAR] τ_abs óptimo.** La abstención con τ_abs=0.5 es una regla de Chow sin costes estimados. Bajo LOAO (familias con prior desconocido) y costes asimétricos (drop vs triage), ¿cuál es la política defendible? **Default:** τ_abs=0.5 + curvas risk-coverage completas en E2.
- **Q4 Score derivado de Choice.** Severidad ≡ rubrik(familia) → el pushforward del posterior de Choice da la pmf de severidad exacta y gratis, consistente por construcción con la atribución. **Default:** mantener el head CORN (contrato F0 + regularizador multi-task medido por E6-A2); computar también el derivado del mismo checkpoint y comparar en v0.2 (experimento barato).
- **Q5 Doble uso de val.** Early stopping y $T$ comparten val. **Default:** aceptar (1 parámetro/cabeza, fuga despreciable); fallback congelado: split 80/20 de val para el fit de $T$ si un reviewer lo exige.
- **Q6 Proyección por bloques de EMBER.** Nano EMBER gasta 53.8% del presupuesto en la proyección. Proyección por los 12 grupos semánticos de thrember (paralela + suma) podría liberar tronco. **Default:** lineal única en v0.1.
- **Q7 Re-materialización de CIC.** §4.4 la cierra para v0.1. **Default:** no; reabrir solo en v0.2 para un `split_time` con Timestamp real del parquet enhanced (join por flow_id ya verificado en S1).
- **Q8 Alternativas al weighting de Kendall.** GradNorm/PCGrad como plan B si las trayectorias de $\lambda_q$ muestran dominancia. **Default:** Kendall modificado + monitoreo; E6-A2 es el control.
- **Q9 Augmentation tabular para UNSW.** El dataset más chico (149K) podría beneficiarse de mixup en el espacio de embeddings / label smoothing en Choice. **Default:** off (dropout + wd + early stop bastan para el alcance; E6 puede ablarlo después).
- **Q10 Prior shift en EMBER.** Train es 50/50 artificial; la prevalencia de despliegue de malware es ≪. La corrección del controlador es $p_{\text{adj}} = \frac{\pi\, p}{\pi\, p + (1-\pi)(1-p)}$ con prior de despliegue $\pi$. **Default:** reportar la calibrada 50/50 (contrato del dataset) + la ajustada con $\pi = 0.01$ como ilustración, dejando claro que π es parámetro de despliegue, no del modelo.

---

## Apéndice A — Pseudocódigo de referencia

```python
# ── Forward + losses (un paso de batch) ─────────────────────────────
def step(model, batch, omega, s, active):
    h = model.trunk(model.proj(batch.x_num), [model.emb(f) for f in batch.x_txt])
    out = {}
    L, lq = 0.0, {}
    if active["noul"]:
        z = model.noul(h).squeeze(-1)
        lq["noul"] = bce_with_logits(z, batch.y_bin)                      # §3.2 sin pesos
    if active["choice"]:
        z = model.choice(h)                                               # [B, K]
        lq["choice"] = -(omega[batch.y_fam] * log_softmax(z).gather(1, y)).sum() \
                       / omega[batch.y_fam].sum()                         # §3.3
    if active["score"]:
        t = model.corn(h)                                                 # [B, 3] logits condicionales
        q = sigmoid(t); C = cumprod(q, dim=1)                             # C1..C3
        pi = stack([1 - C[:,0], C[:,0]-C[:,1], C[:,1]-C[:,2], C[:,2]], 1)
        for k in (1, 2, 3):                                               # §3.4 por tarea
            m = batch.y_sev >= k - 1
            lq[f"corn{k}"] = bce_with_logits(t[m, k-1], (batch.y_sev[m] >= k).float())
        lq["score"] = sum(lq.pop(f"corn{k}") for k in (1, 2, 3))
    for name, loss in lq.items():                                         # §3.6 Kendall modificado
        L = L + exp(-s[name]) * loss + s[name]
    return L, pi.detach(), C.detach()

# ── Decodificación Choice (inferencia) ──────────────────────────────
z = model.choice(h) - log(omega)                    # de-tilt (Prop. 2)
p = softmax(z / T_choice)                           # §5
if head.mode == "abstain":
    cand = argmax(p) if max(p) >= tau_abs else UNKNOWN
else:                                               # trained_unknown (EMBER)
    cand = argmax(p)                                # unknown es un slot más

# ── Ajuste de temperatura (por cabeza, sobre val) ────────────────────
grid = logspace(log10(0.05), log10(20), 200)        # + sección dorada al final
T_hat = argmin_over(grid, lambda T: NLL_val(head_outputs(val), T))   # §5.1

# ── Loader EMBER (streaming) ─────────────────────────────────────────
def epoch_shards(paths, epoch):
    rng = np.random.default_rng(42 + epoch)
    for shard in rng.permutation(paths):            # 46 shards train
        tbl = decode(shard)                         # prefetch 1 shard en thread aparte
        yield rng.permutation(len(tbl)) batches de tbl   # shuffle intra-shard

# ── Loop de entrenamiento (esqueleto) ────────────────────────────────
for epoch in range(max_epochs):
    for batch in loader(epoch):
        with autocast(amp_dtype):
            L, _, _ = step(model, batch, omega, s, active_masks(batch))
        opt.zero_grad(set_to_none=True)
        scaler.scale(L).backward() if fp16 else L.backward()
        (scaler.unscale_(opt) if fp16 else None); clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt) if fp16 else opt.step(); scaler.update() if fp16 else None
        opt.step_sched()                            # warmup 5% + cosine
    M = M_val(model, val)                           # §6.4
    save_if_best(model, M); stop_if(patience > 4)
calibrate(best, val)                                # §5 → artefacto final §8
```

## Apéndice B — Procedencia de las cifras

Todos los conteos (filas por familia/severidad/split, vocabularios de train, filas no-finitas, subconjunto ddos, elegibles de CORN) y las tablas de parámetros (§2.3) y de pesos (§3.5) fueron verificados el 2026-10-17 con un script de conteo en memoria (pyarrow, lectura directa de los parquets materializados de F1, sin tocar el repo). Las fórmulas de §2.3 permiten re-derivar el conteo de parámetros exacto; cualquier divergencia futura con un conteo de `torch` indicaría un bug de implementación contra esta spec. Los conteos de train usados en las $\omega_k$ corresponden al split de la corrida (D12). Cifras de challenge EMBER (98.8% unknown) y drift de test provienen del censo de materialización F1 (`logs/ember_family_census.log`, SCHEMA §3.4).

*Fin de la especificación F2. El siguiente paso es la implementación de `src/models/` contra este contrato y la matriz de corridas de §6.6.*

