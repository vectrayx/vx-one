# VX-One F2 — Advisory de diseño: Q1, Q2, Q3 [DELEGAR]

**Fecha:** 2026-10-17 · **Autor:** Opus (revisor adversarial de diseño F2, delegado para §9 de la spec)
**Objeto:** resolver con rigor las tres preguntas marcadas [DELEGAR] en `docs/VX_One_Model_Spec_F2.md` §9. Este advisory NO modifica la spec; los cambios propuestos van en §5 con anclas exactas.
**Insumos leídos completos:** `docs/VX_One_Model_Spec_F2.md` (586 líneas), `VectrayX_One_Model_Spec_F0.md` (contratos de primitivas), `data/SCHEMA.md`.
**Regla de citas cumplida:** las 19 referencias de §4 se verificaron una a una (autor/venue/año) con WebSearch/WebFetch ANTES de incluirlas. Ningún claim descansa en una referencia no verificada; las no verificadas se listan aparte y no se usan.

**Verificaciones de datos corridas para este advisory** (`.venv/bin/python -I`, pyarrow/numpy, solo lecturas sobre `data/processed/`):

| Verificación | Resultado |
|---|---|
| Cierre del rubrik, UNSW (257,673 filas): `sev≠rubrik(fam)`, `sev≥2∧normal`, `recon∧sev≠1`, `sev≥1∧normal` | **0 violaciones** en las 4 reglas |
| Cierre del rubrik, CIC (2,830,743 filas): mismas 4 reglas | **0 violaciones** |
| Prior binario EMBER por semana (w0–w63) | **30,000 benign + 30,000 malicious exactos en TODAS las semanas** train/val/test |
| Challenge EMBER (w64) | 0 benign / 6,315 malicious; familia: unknown 6,236 (98.8%), wacatac 78, expiro 1 |
| Censo de familias EMBER val (w46–51) vs test (w52–63) | drift temporal real: sfone 496→132, sinowal 72→18, grandoreiro 30→226, berbew 34,508→92,086 |

Estos números se usan abajo como hechos verificados, no como supuestos.

---

## 0. Resumen de veredictos

| Q | Default (spec §9) | Veredicto | Claim para el paper |
|---|---|---|---|
| Q1 | Sin claim formal para composiciones; E3 empírico + conformal v0.2 | **CONFIRMAR + MEJORAR** | Los gates v0.1 son funcionales de UNA cabeza (rubrik-reducibles; 0 violaciones verificadas) y su error de calibración queda acotado por la deficiencia de su cabeza (Lemas A–C). Para composiciones de probabilidad ENTRE cabezas no existe cota útil (Prop D: ECE 0+0 → 1/4). El controlador compone decisiones booleanas, nunca probabilidades. |
| Q2 | Conformal split opcional con caveat | **CONFIRMAR, con protocolo agudo** | No prometer coverage distribution-free en EMBER. Garantía exacta solo donde hay intercambiabilidad (UNSW split oficial, CIC `split_strat`). En EMBER: coverage empírico por semana + IC Clopper-Pearson, cota citable del gap (Barber et al. 2023, ≈2·d_TV), weighted como mitigación empírica. Challenge fuera de toda promesa. |
| Q3 | τ_abs=0.5 + risk-coverage en E2 | **MEJORAR** | τ_abs=0.5 se mantiene, re-justificado: es la regla de Bayes con rechazo bajo costes uniformes declarados (λ_r=0.5; τ_abs = 1−λ_r), no un umbral mágico. Regla general con matriz de costes = política del controlador. Risk-coverage con definición LOAO de dos ejes como producto primario. Conformal abstention rechazado para Choice bajo LOAO. |

---

## 1. Q1 — Composición de la calibración en los gates

### (a) Veredicto sobre el default: **CONFIRMAR + MEJORAR**

El default es correcto y debe mantenerse: no existe — ni puede existir — un claim de composición para políticas que combinan probabilidades de cabezas distintas. Eso se puede **demostrar**, no solo citar (Prop D), lo que hace al default más sólido de lo que la propia spec sospechaba.

Se MEJORA en la dirección opuesta, que la spec no explota: **los tres gates de v0.1 no son composiciones.** Son funcionales de una única cabeza, y para funcionales de una cabeza sí hay lemas de clausura con demostraciones de tres líneas. El paper puede hacer un claim formal sobre sus gates sin pisar el territorio de la composición.

### (b) Argumento

**Definición de trabajo.** Para un evento $A$ y un estimador $f(X)\in[0,1]$:

$$\mathrm{cal}_A(f) \;=\; \mathbb{E}\big[\,|\,\mathbb{P}(A\mid f(X)) - f(X)\,|\,\big]$$

error de calibración L1 poblacional. El ECE con 15 bins es su estimador (con las reservas de Kumar et al. 2019: el binned ECE puede mejorar con modelos peores; por eso E3 co-reporta MCE y reliability diagrams). Nociones multiclase: **vector-calibrado** = $\mathbb{E}[\mathbf{1}[y{=}k]\mid p(X){=}v]=v_k\ \forall k$; **classwise** = lo mismo por coordenada; **confidence** = solo para $\max_k p_k$. Jerarquía: vector ⟹ classwise ⟹ confidence (Vaicenavicius et al., arXiv:1902.06977). La temperature scaling por NLL apunta a confidence; las nociones fuertes hay que MEDIRLAS (§1(d)).

**Lema A (clausura bajo post-procesamiento dentro de una cabeza).** Si $f$ estima $A$ y $h:[0,1]\to[0,1]$ es medible, entonces

$$\mathrm{cal}_A(h\circ f) \;\le\; \mathrm{cal}_A(f) + \mathbb{E}\,|h(f(X))-f(X)|.$$

*Demostración.* Sea $m(v)=\mathbb{P}(A\mid f(X){=}v)$. Torre: $\mathbb{E}[\mathbf{1}_A\mid h\circ f]=\mathbb{E}[m(f(X))\mid h\circ f]$. Jensen: $|\mathbb{E}[m(f)\mid h\circ f]-h(f)|\le\mathbb{E}[\,|m(f)-h(f)|\mid h\circ f\,]$. Triángulo $|m(f)-h(f)|\le|m(f)-f|+|f-h(f)|$; tomar esperanza. ∎

Lectura operativa: transformar la salida de UNA cabeza (de-tilt, temperatura, umbralización para decisiones) no descontrola la calibración; el daño está acotado por el tamaño de la transformación. Lo que el Lema A **no** cubre es $g(f_1,f_2)$ con dos cabezas — exactamente el hueco de la pregunta.

**Lema B (clausura subset-sum).** Si el vector $p(X)$ del Choice head está vector-calibrado, entonces para cualquier $S$: $m_S(X)=\sum_{k\in S}p_k(X)$ está calibrado para el evento $\{y\in S\}$: $\mathbb{P}(y\in S\mid m_S=t)=t$.

*Demostración.* $\mathbb{E}[\mathbf{1}[y\in S]\mid p{=}v]=\sum_{k\in S}v_k=m_S(v)$ (linealidad); $m_S$ es función de $p$; condicionar en $m_S{=}t$ y aplicar torre. ∎

Caveat: requiere vector calibration, que TS no garantiza. Si algún día se usa un subset-sum como probabilidad de un evento compuesto (v0.1 no lo hace), E3 debe medir su ECE directamente.

**Lema C (chain rule CORN: exactitud y degradación).** Con las condicionales verdaderas $\pi_1(x)=\mathbb{P}(\text{sev}\ge1\mid x)$ y $\pi_2(x)=\mathbb{P}(\text{sev}\ge2\mid\text{sev}\ge1,x)$:

(i) **Exactitud:** $\mathbb{P}(\text{sev}\ge2\mid x)=\pi_1(x)\,\pi_2(x)$ **exacto** por la regla de la cadena — sin supuesto de independencia (CORN condiciona el paso 2 en $\text{sev}\ge1$; la sospecha frecuente de independencia escondida es infundada).

(ii) **Degradación:** para modelos $q_1,q_2$,

$$\mathrm{cal}_{\text{sev}\ge2}(q_1q_2)\;\le\;\mathbb{E}|\pi_1\pi_2-q_1q_2|\;\le\;\mathbb{E}[\pi_2\,|\pi_1-q_1|]+\mathbb{E}[q_1\,|\pi_2-q_2|]\;\le\;\mathbb{E}|\pi_1-q_1|+\mathbb{E}|\pi_2-q_2|.$$

*Demostración.* Paso 1: $\mathrm{cal}(f)\le\mathbb{E}|\pi_{\text{true}}-f|$ (mecanismo del Lema A: $|\mathbb{E}[\pi\mid f]-f|\le\mathbb{E}[|\pi-f|\mid f]$, tomar esperanza). Paso 2: $\pi_1\pi_2-q_1q_2=\pi_2(\pi_1-q_1)+q_1(\pi_2-q_2)$ y desigualdad triangular. Paso 3: $\pi_2,q_1\le1$. ∎

**Nota honesta:** los términos del bound son **desviaciones L1 de las condicionales respecto a sus objetivos verdaderos, NO los ECE marginales** de $q_1,q_2$. La calibración marginal de cada condicional no controla la del producto: el mecanismo es el de Kleinberg et al. 2017 (condicionar en información extra — aquí, la otra condicional — cambia la base rate); multicalibration (Hébert-Johnson et al. 2018) es la noción que cerraría el hueco, pero solo para familias de test pre-especificadas. Consecuencia práctica: el ECE del gate $C_2$ se **mide directamente** (E3 ya lo hace vía $g_{\text{drop}}$), y las reliability por condicional (§1(d)) permiten **atribuir** cuánto viene de cada eslabón.

**Prop D (no-go: no existe cota útil en ECEs marginales).** Sea $X$ constante; $f_1\equiv a$, $f_2\equiv b$ (estimadores constantes), perfectamente calibrados para eventos $A$ y $B$ con $\mathbb{P}(A)=a$, $\mathbb{P}(B)=b$ ($\mathrm{cal}_1=\mathrm{cal}_2=0$). La política AND ingenua $g=f_1\cdot f_2=ab$ estima $A\wedge B$; la OR $g'=1-(1-f_1)(1-f_2)=a+b-ab$ estima $A\vee B$. La ley conjunta de $(A,B)$ es libre dentro de las cotas de Fréchet: $\mathbb{P}(A\wedge B)\in[\max(0,a{+}b{-}1),\,\min(a,b)]$, de modo que

$$\sup\big|\mathbb{P}(A\wedge B)-ab\big| \;=\; \tfrac14\qquad\text{(alcanzada en } a=b=\tfrac12\text{, con } A=B \text{ o } A\cap B=\varnothing\text{)},$$

e igualmente $\sup|\mathbb{P}(A\vee B)-(a+b-ab)|=1/4$. Es decir: $\varphi(0,0)\ge 1/4$ para cualquier bound $\varphi(\mathrm{cal}_1,\mathrm{cal}_2)$ — **dos cabezas perfectamente calibradas pueden componer una política AND con ECE $=1/4$** (el máximo del escenario). Ninguna función no trivial de los ECEs marginales acota la composición.

El caso $A=B$ no es patológico en seguridad: es la situación realista de dos cabezas que estiman el mismo evento subyacente ("¿es ataque?") — p. ej. una política ingenua $g=\text{noul}\wedge(1-p_{\text{choice}}(\text{normal}))$. v0.1 no contiene ninguna política así (tabla abajo); esa disciplina es lo que hace defendible al diseño, y el paper debería decirlo.

**Prop E (condiciones suficientes para composición futura).** Si (i) **calibración conjunta**: $\mathbb{E}[A\mid f_1,f_2]=f_1$ y $\mathbb{E}[B\mid f_1,f_2]=f_2$ (multicalibration respecto al par), y (ii) $A\perp B\mid(f_1,f_2)$, entonces $f_1f_2$ está calibrado para $A\wedge B$ y $1-(1-f_1)(1-f_2)$ para $A\vee B$. Si además $A\cap B=\varnothing$ (**¡verificable con datos vía rubrik!**), (i) sola ya basta para la suma $f_1+f_2$ como estimador de $A\vee B$.

*Demostración.* Bajo (i)+(ii): $\mathbb{E}[\mathbf{1}_{A\wedge B}\mid f_1,f_2]=f_1f_2=:g$; $g$ es función del par; condicionar en $g$ y aplicar torre. Disjoint: $\mathbf{1}_{A\vee B}=\mathbf{1}_A+\mathbf{1}_B$ c.s. y linealidad. ∎

Lectura: (i) es **medible** (ECE de $f_1$ dentro de estratos de $f_2$; §1(d)); (ii) es un supuesto estructural **no verificable** y típicamente **falso** en seguridad (dos indicadores de ataque correlacionan). El rubrik de la spec evita necesitar (ii): hace que cada evento objetivo sea función del ground truth de UNA cabeza.

**Tabla: los gates v0.1 bajo esta luz** (reducciones verificadas con 0 violaciones en 3,088,416 filas de red):

| Gate | Evento objetivo | Cabeza | Estructura | Qué se necesita para el claim |
|---|---|---|---|---|
| $g_{\text{attack}}$ | $\{y\neq\text{normal}\}$ | Noul | identidad | calibración del Noul head (E3 ya la mide) |
| $g_{\text{recon}}$ | $\{y=\text{recon}\}$ | Choice | proyección a la coordenada `recon` | classwise calibration de Choice para recon (Lema A con $h$=coordenada; medir, §1(d)) |
| $g_{\text{drop}}$ | $\{\text{sev}\ge2\}$ | Score (CORN) | $C_2=q_1q_2$ | Lema C: exacto en condicionales verdaderas; degradación ≤ suma de desviaciones; ECE del gate medido directo |

Notas: $g_{\text{drop}}$ requiere head Score → solo UNSW/CIC (EMBER no tiene Score, S6). $g_{\text{recon}}$ requiere que `recon` sea candidata → no existe como gate en CIC `split_time` (D12: 0 filas train). Rubrik verificada: $\text{sev}\ge2\Rightarrow\text{attack}$ (0 filas $\text{sev}\ge2\wedge\text{normal}$) y $\text{recon}\Rightarrow\text{sev}=1$ (0 violaciones) — el evento de cada gate coincide **exactamente** con el evento que su cabeza estima, sin composición.

Y la observación estructural que cierra la pregunta: **v0.1 compone DECISIONES (comparaciones con τ; AND/OR booleanos en el controlador) entre gates; nunca compone PROBABILIDADES entre cabezas.** Las decisiones booleanas se componen libremente: el coste esperado de una decisión ya fue evaluado aguas arriba por el estimador calibrado de su propio gate. El no-go de la Prop D aplica solo a la segunda. Esta distinción debe aparecer explícita en el paper.

### (c) Qué claimar en el paper

Propongo este texto (núcleo):

> "Los gates de v0.1 son funcionales de una única cabeza: $g_{\text{attack}}$ es la salida del Noul head, $g_{\text{recon}}$ es la coordenada `recon` del Choice head, y $g_{\text{drop}}$ es $C_2=q_1q_2$ del Score head (CORN). El rubrik hace exacta la reducción del evento objetivo a una sola cabeza (verificado: 0 filas violan las implicaciones del rubrik en 3.09M flujos). Para funcionales de una cabeza, la clausura bajo post-procesamiento acota el error de calibración del gate por la deficiencia medida de su cabeza (Apéndice X, Lemas A–C). No hacemos ningún claim de composición entre cabezas: dos cabezas perfectamente calibradas pueden componer una política AND con ECE $1/4$ (Apéndice X, Prop D); por eso el controlador v0.1 compone decisiones booleanas entre gates y nunca probabilidades entre cabezas, y todo gate compuesto futuro exige ser rubrik-reducible o medir su ECE conjunto."

Soporte citable: Guo et al. 2017 (temperature scaling); Kumar et al. 2019 (estimación del ECE); Vaicenavicius et al. 2019 (jerarquía de nociones de calibración multiclase); Kleinberg et al. 2017 y Hébert-Johnson et al. 2018 (por qué la calibración marginal no compone y qué noción la arreglaría).

### (d) Experimento mínimo (datos ya en `data/processed/`; sin entrenamiento nuevo)

1. **Reliability por condicional CORN (adición a E3).** Diagrama y ECE de $q_1$ sobre TODAS las filas y de $q_2$ sobre filas elegibles ($\text{sev}\ge1$), junto al ECE directo del gate $C_2$. Propósito: atribuir el ECE de $C_2$ a sus eslabones (Lema C). Coste: reusar las predicciones de E3; solo cambia el binning.
2. **Classwise reliability de Choice (adición a E3).** Por clase $k$: $\mathbb{E}[\mathbf{1}[y{=}k]\mid p_k]$ vs $p_k$ (binned). Soporta el claim del gate $g_{\text{recon}}$, que necesita classwise — la confidence calibration que optimiza TS es más débil. Coste: ídem, distinto binning.
3. **(Solo si alguna vez se despliega una política compuesta; NO necesario para v0.1)** Diagnóstico cross-head: ECE del Noul dentro de deciles de $\max_k p_{\text{choice},k}$ (sondeo de la condición (i) de la Prop E). Gate de entrada obligatorio para cualquier gate compuesto v0.2+.
4. Las verificaciones del rubrik ya están corridas y son parte del claim: reportarlas en el paper como chequeo de integridad (0 violaciones en 3,088,416 filas).

---

## 2. Q2 — Conformal bajo drift temporal

### (a) Veredicto sobre el default: **CONFIRMAR, con protocolo agudo**

La dirección del default (conformal split opcional con caveat) es correcta, pero "caveat" sin protocolo se lee como descargo. Abajo queda definido exactamente qué se promete, dónde, con qué cota citable, y qué se reporta empíricamente.

### (b) Argumento

**Qué se rompe exactamente.** Split conformal necesita intercambiabilidad entre calibración y test (Papadopoulos et al. 2002). El split EMBER es temporal por semanas con drift real verificado (sfone 496→132 entre val y test, sinowal 72→18, grandoreiro 30→226, berbew 34,508→92,086). Intercambiabilidad: rota por construcción. El split oficial de UNSW y el `split_strat` de CIC son intercambiables por diseño (aproximación i.i.d. estándar del campo) — ahí la garantía exacta SÍ aplica. CIC `split_time` es temporal (días como proxy): misma clase de caveat que EMBER, con drift intra-semana menor.

**La estructura del drift en EMBER está verificada y es tratable por partes:**

- **Prior binario ESTABLE: 30,000/30,000 benign/malicious en TODAS las semanas** train/val/test (verificado por semana, w0–w63). Para el head Noul (binario) **no hay label shift** en test main: lo que puede romper el coverage de Noul es covariate shift dentro de cada clase. Es el mejor caso honesto para weighted conformal (Tibshirani et al. 2019), cuya garantía exacta asume covariate shift con ratios $w(x)$ **conocidos**.
- **La mezcla de familias SÍ cambia** (crosstab de drift): para Choice eso es label shift sobre la variable familia — la ruta formal es label-shift conformal (Podkopaev & Ramdas 2021), con pesos estimables desde la confusión en val, sin densidades de $x$.
- **Challenge: 0 benign (verificado), 98.8% unknown, evasivo por diseño** — adversarial/concept drift. Ningún wrapper conformal promete algo ahí. La cobertura de la clase positiva (Noul) es medible; la de benign no existe (no hay benignos). El challenge queda fuera de TODA promesa y entra como test de estrés.

**Rutas disponibles (con su estado real):**

| Ruta | Garantía | Con pesos estimados | Veredicto para el paper |
|---|---|---|---|
| Split conformal (Papadopoulos et al. 2002; Angelopoulos & Bates) | exacta bajo intercambiabilidad | — | **Garantía exacta SOLO en UNSW split oficial y CIC `split_strat`** (una frase en el paper) |
| Weighted conformal, covariate shift (Tibshirani et al. 2019) | exacta SOLO con ratios $w(x)$ conocidos | **no exacta** | **Mitigación empírica** en EMBER; nunca prometer |
| PAC prediction sets (Park et al., ICLR 2022) | válida con CIs de los pesos estimados | sí, PERO exige intervalos de confianza de los pesos | Formal pero cara; fuera de alcance v0.1; citar como ruta |
| Beyond exchangeability (Barber et al., AoS 2023) | coverage $\ge 1-\alpha-\sum_i\tilde w_i\,d_{TV}$ | degradación cuantificada | **Ancla cuantitativa citable**: gap $\approx 2\,d_{TV}$ en el caso simétrico |
| Adaptive conformal (Gibbs & Candès 2021) | frecuencia de coverage a largo plazo | no necesita pesos | Alternativa online; promesa distinta (coverage promedio temporal); NO default |
| Label shift conformal (Podkopaev & Ramdas 2021) | exacta con pesos de label shift (BBSE) | estimable | Ruta correcta para coverage por familia de Choice (v0.2) |

**Lema BER–TV (autocontenido; certifica drift, nunca su ausencia).** Para distribuciones $P,Q$ y el error balanceado óptimo $\mathrm{BER}^*=\inf_h\tfrac12(\mathbb{P}_P(h{=}1)+\mathbb{P}_Q(h{=}0))$: $d_{TV}(P,Q)=1-2\,\mathrm{BER}^*$. Luego cualquier clasificador de dominios $h$ entrenado da una **cota inferior certificada** $d_{TV}\ge 1-2\,\mathrm{BER}(h)$. (Demostración: el óptimo de la clasificación binaria equiponderada es el test de likelihood ratio, con error $(1-d_{TV})/2$; tres líneas, va al apéndice.) Dos consecuencias: (1) si un clasificador val-vs-semana-$w$ separa, el drift queda **certificado con magnitud**, y esa magnitud entra en el gap de Barber et al. 2023; (2) si NO separa, no certifica nada — ausencia de evidencia de drift NO es evidencia de intercambiabilidad. Asimetría que el paper debe declarar.

> Nota de proveniencia: la identidad es estándar en domain adaptation (asociada a Ben-David et al. 2010), pero no pude verificar la edición exacta con acceso comprobable; por regla de citas se omite la referencia y el lema va demostrado autocontenido.

**Pesos sin densidades.** No hace falta estimar densidades: (i) clasificador de dominios probabilístico $g$ (logística sobre $x_{\text{num}}$; barato, no el modelo profundo) y odds $w=(1-g)/g$, clipeado a $[0.1,10]$; o (ii) estimación directa de importance (Kanamori et al., JMLR 2009). Con pesos **estimados** la garantía de Tibshirani et al. **no es exacta**; lo honesto es reportar el delta de coverage empírico unweighted vs weighted por semana y citar Park et al. 2022 como la única ruta formal con pesos estimados.

### (c) Qué prometer exactamente (texto scoped para el paper)

- **UNSW split oficial / CIC `split_strat`:** "Bajo splits tratados como intercambiables, el wrapper conformal sobre Noul garantiza $\mathbb{P}(y\in C(X))\ge1-\alpha$." — garantía exacta, una frase.
- **EMBER test main (w52–63):** NINGUNA frase de garantía. Reportar: coverage empírico por semana (global y por clase) con IC Clopper-Pearson; comparación unweighted vs weighted (pesos por clasificador de dominios clipeado); y el anclaje citable: "la cota de Barber et al. (2023) degrada el coverage a razón de $\approx 2\,d_{TV}$ entre calibración y test, con $d_{TV}$ acotado inferiormente por $1-2\,\mathrm{BER}$ del clasificador de dominios" — convierte el drift verificado en un número.
- **CIC `split_time`:** tratarlo como EMBER (empírico, sin promesa), aunque el drift es menor.
- **Challenge:** "adversarial por diseño (0 benignos verificados); no se hace ninguna afirmación de cobertura; la cobertura de la clase positiva se reporta como test de estrés del detector."
- **El claim de fondo:** "reportamos cobertura empírica con intervalos, no garantías distribution-free bajo drift temporal". Esa honestidad es una fortaleza: nadie en NIDS tiene garantías bajo drift temporal real; quien las proclama asume intercambiabilidad escondida.

### (d) Protocolo mínimo (conformal sobre Noul, EMBER; sin entrenamiento nuevo)

- Head: Noul calibrado (checkpoint existente de E3, p. ej. Small), $\alpha=0.1$. Calibración conformal en val EMBER (w46–51, 360,000 filas). Score $s(x,y)=1-p^{\text{cal}}(y\mid x)$; $C(x)=\{y: s(x,y)\le\hat Q\}$ con $\hat Q$ el $\lceil(n{+}1)(1{-}\alpha)\rceil$-ésimo menor (split conformal).
- Por semana $w\in\{52..63\}$: coverage global ($n=60{,}000$ → IC 95% de Clopper-Pearson ≈ ±0.24%) y por clase benign/malicious ($n=30{,}000$ → ≈ ±0.34%).
- Variante weighted: clasificador de dominios logístico val vs semana $w$; odds clipeado $[0.1,10]$; cuantíl ponderado (esquema de Tibshirani et al. 2019 / eq. 11 de Barber et al. 2023); reportar el delta de coverage por semana.
- Drift certificado por semana: BER balanceado del clasificador de dominios → $d_{TV}\ge1-2\,\mathrm{BER}$ → gap de Barber implícito. Tabla de 12 filas: semana | cota inferior de $d_{TV}$ | coverage unweighted (IC) | coverage weighted (IC).
- Challenge: coverage sobre positivos, reportado como estrés, sin lenguaje de garantía.
- (v0.2, opcional) Coverage por familia de Choice con pesos de label shift (Podkopaev & Ramdas 2021) — las familias que casi desaparecen (sfone, sinowal) son el caso interesante: su coverage por familia es lo primero que el drift rompe.

---

## 3. Q3 — τ_abs óptimo con costes asimétricos

### (a) Veredicto sobre el default: **MEJORAR**

τ_abs=0.5 se mantiene; su **justificación** cambia de categoría. El default lo presenta como "regla de Chow sin costes"; eso deja τ_abs como constante mágica y es exactamente lo que un reviewer ataca ("¿por qué 0.5?"). La misma constante es defendible de raíz: es la **regla de Bayes con rechazo bajo costes uniformes declarados**. La derivación cabe en un apéndice de tres ecuaciones y blinda la sección entera.

### (b) Argumento

**1. Hay TRES knobs distintos que el default deja enredados.**

- $\tau_{\text{det}}$ (Noul → alerta): asimetría missed-attack vs false-alarm. Regla clásica: alertar si $p\ge c_{FP}/(c_{FP}+c_{FN})$ (Elkan 2001). Con $c_{FN}=10\,c_{FP}$: $\tau^*\approx0.091$. El 0.5 congelado para $g_{\text{attack}}$ equivale a declarar costes 1:1 — hay que decirlo; las métricas threshold-free de E1 (AUROC, FPR@95TPR) ya exponen todos los puntos de operación, así que NO hace falta descongelar τ_det: basta declarar el mapeo coste→umbral y reportar sensibilidad.
- $\tau_{\text{abs}}$ (Choice → abstención): fricción de triage vs atribución errónea. Otro knob con otra justificación (punto 2).
- **Composición:** dónde se aplica la abstención (¿población completa o solo alertas?) — política del controlador, no del modelo (punto 6).

**2. Bayes con rechazo (la derivación que justifica τ_abs).** Acciones $a\in\text{candidates}\cup\{\text{unknown}\}$; coste esperado $C(a\mid x)=\sum_k p_k(x)\,c(a,k)$ con $c(\text{unknown},k)=\lambda_r\ \forall k$. Regla de Bayes: abstenerse sii $\min_{a\in\text{candidates}}C(a\mid x)>\lambda_r$. Con costes de error uniformes ($c(a{=}y)=0$, $c(a{\neq}y)=1$): $C(a\mid x)=1-p_a(x)$, luego

$$\min_a C(a\mid x)=1-\max_k p_k(x)\;\Rightarrow\;\text{abstenerse sii }\max_k p_k(x)<1-\lambda_r.$$

**$\tau_{\text{abs}}=1-\lambda_r$.** El default 0.5 ≡ costes uniformes con $\lambda_r=0.5$: un unknown cuesta lo mismo que una atribución errónea, **declarado**. Chow (1970) es exactamente este caso 0-1 (tradeoff óptimo error/rechazo); el marco general con matrices de coste es learning with rejection (Cortes et al., ALT 2016); la curva risk-coverage como objeto primario es selective prediction (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017).

**3. Por qué max-prob NO es Bayes con costes asimétricos (contraejemplo de una línea).** $p=(\text{exploitation }0.55,\ \text{dos }0.40,\ \text{normal }0.05)$; $c(\text{exploit}\mid y{=}\text{dos})=3$, demás errores 1, $\lambda_r=0.5$:
$C(\text{exploit})=0.40\cdot3+0.05=1.25$; $C(\text{dos})=0.55+0.05=0.60$; $\min=0.60>0.5$ → **Bayes se abstiene**; Chow ($0.55\ge0.5$) atribuye "exploitation" con coste esperado 1.25. Conclusión: la regla general (matriz de costes) pertenece al **controlador**; el modelo expone $p$ calibrado y el controlador evalúa $C(a\mid x)$. La spec ya lo insinúa ("la política vive en el espacio de probabilidad", §5.2) — hay que hacerlo explícito también para Choice.

**4. Support mismatch bajo LOAO (qué mide realmente %unknown).** En LOAO, $p^{\text{cal}}$ es una posterior sobre los **nativos**, calibrada condicionalmente en "la fila es nativa". Las filas de la familia excluida están fuera del soporte del entrenamiento: la abstención ahí es **detección por semejanza** (max-softmax; Hendrycks & Gimpel 2017 — baseline conocido por overconfident en OOD), no una afirmación probabilística sobre la familia excluida. Consecuencias para E2: (i) %unknown bajo LOAO es la métrica correcta, pero nombrada con precisión; (ii) el híbrido D1 es exactamente lo correcto — EMBER **entrena** la clase unknown (rechazo in-distribution, afirmación probabilística legítima), UNSW/CIC **abstiene por umbral** (rechazo OOD) — y el paper no puede dejar que el lector crea que son el mismo mecanismo.

**5. Conformal abstention para Choice bajo LOAO: RECHAZADO como primario.** (i) Las filas LOAO están ausentes de la calibración también (salen de train Y val): la intercambiabilidad falla **justo en las filas de las que trata el experimento**; un set conformal no carga garantía ahí. (ii) "unknown" pasaría a significar tres cosas distintas (clase entrenada en EMBER, acción de abstención en UNSW/CIC, pertenencia a set conformal). (iii) Para native rows en splits estratificados el conformal sí es legítimo → v0.2 como complemento, nunca mecanismo primario de LOAO.

**6. Condicionar la abstención al gate es legítimo — como política del controlador.** Aplicar $\tau_{\text{abs}}$ solo entre alertas ($g_{\text{attack}}$ true) es cascada selectiva estándar; equivale a un τ por subpoblación ($\tau_{\text{abs}}\mid\text{alertas}\ge\tau_{\text{abs}}\mid\text{población}$ — razonable: entre alertas la prior de ataque sube). El invariante §8 (`candidate=="unknown"` ⟺ `max_prob < τ_abs`) se mantiene sin cambio mientras τ quede fijo por contexto de despliegue y documentado en `head_meta`. Si se despliega, E3 debe reportar ECE estratificado por gate.

**7. Mínimo para que Choice+unknown no parezca ad hoc:**

1. **Curvas risk-coverage como producto primario de E2**, con definición LOAO de dos ejes: (a) curva estándar (riesgo vs coverage barriendo τ_abs sobre la población completa) + AURC; (b) por familia excluida: $\%\text{unknown}(f;\tau_{\text{abs}})$ (eje de semejanza) y **falsa atribución** $1-\%\text{unknown}$ como riesgo en las filas de la familia. Grid $\tau_{\text{abs}}\in\{0.05,\dots,0.95\}$.
2. **La derivación Bayes con rechazo en un apéndice** (3 ecuaciones, punto 2) + tabla analítica $\lambda_r\to\tau_{\text{abs}}$ (0.1→0.9, 0.25→0.75, 0.5→0.5) + el ejemplo asimétrico del punto 3.
3. **Costes declarados explícitos:** para Noul, el mapeo de Elkan con el ejemplo 10:1 ($\tau^*\approx0.091$) y la declaración de que v0.1 opera en 0.5 = costes 1:1; para Choice, costes uniformes + $\lambda_r=0.5$ → $\tau_{\text{abs}}=0.5$. El aporte del paper es la **derivación + sensibilidad**; los constantes son operativos y se declaran como tales.
4. **Diagnóstico barato:** % de filas de val donde la regla Bayes general (con la matriz declarada) difiere de Chow max-prob. Si ≈0%, la simplificación queda justificada empíricamente en una línea; si no, es evidencia para el controlador con matriz de costes.
5. **El párrafo de support mismatch** (punto 4), citando Hendrycks & Gimpel 2017, distinguiendo rechazo in-distribution (EMBER) de rechazo OOD (UNSW/CIC).

### (c) Qué claimar en el paper

> "La abstención de Choice es la regla de Bayes con rechazo bajo costes declarados: con costes de error uniformes y coste de escalada $\lambda_r$, se abstiene sii $\max_k p_k<1-\lambda_r$ (Chow 1970 como caso 0-1; Cortes et al. 2016 para el marco general con matrices de coste, que reside en el controlador). v0.1 declara $\lambda_r=0.5$ (τ_abs=0.5) y opera el Noul en τ_det=0.5 con el mapeo de Elkan (2001) documentado para operadores. Bajo LOAO, la abstención de familias excluidas es detección por semejanza fuera de soporte (Hendrycks & Gimpel 2017), no probabilidad calibrada sobre la familia; reportamos %unknown y falsa atribución por familia junto a curvas risk-coverage (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017)."

### (d) Experimentos mínimos

1. E2 con sweep de τ_abs (19 puntos) → las dos curvas + AURC del punto 7.1 (predicciones ya producidas por los folds LOAO de §6.7; solo varía el umbral).
2. Diagnóstico Bayes-vs-Chow del punto 7.4 sobre val (mismas predicciones de E3; comparar $\arg\min_a C(a\mid x)$ vs $\arg\max_k p_k$).
3. Tabla $\lambda_r\to\tau_{\text{abs}}$ y ejemplo asimétrico: analíticos, sin datos.
4. Si se despliega τ por subpoblación (punto 6): E3 estratificado por gate.

---

## 4. Referencias

### 4.1 Verificadas una a una (autor, título, venue, año) antes de incluir

1. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. "On Calibration of Modern Neural Networks". ICML 2017. arXiv:1706.04599.
2. J. Kleinberg, S. Mullainathan, M. Raghavan. "Inherent Trade-Offs in the Fair Determination of Risk Scores". ITCS 2017. arXiv:1609.05807.
3. U. Hébert-Johnson, M. Kim, O. Reingold, G. Rothblum. "Multicalibration: Calibration for the (Computationally-Identifiable) Masses". ICML 2018 (PMLR v80). arXiv:1711.08513.
4. A. Kumar, P. Liang, T. Ma. "Verified Uncertainty Calibration". NeurIPS 2019. arXiv:1909.10155.
5. D. Vaicenavicius, D. Widmann, C. Andersson, F. Lindsten, J. Roll, T. B. Schön. "Evaluating model calibration in classification". arXiv:1902.06977 (preprint; venue no confirmado → **citar como arXiv**).
6. H. Papadopoulos, K. Proedrou, V. Vovk, A. Gammerman. "Inductive Confidence Machines for Regression". ECML 2002.
7. R. J. Tibshirani, R. F. Barber, E. Candès, A. Ramdas. "Conformal Prediction Under Covariate Shift". NeurIPS 2019. arXiv:1904.06019.
8. R. F. Barber, E. Candès, A. Ramdas, R. J. Tibshirani. "Conformal prediction beyond exchangeability". Annals of Statistics 51(2), 2023. arXiv:2202.13415.
9. S. Park, E. Dobriban, I. Lee, O. Bastani. "PAC Prediction Sets Under Covariate Shift". ICLR 2022. arXiv:2106.09848.
10. I. Gibbs, E. Candès. "Adaptive Conformal Inference Under Distribution Shift". NeurIPS 2021. arXiv:2106.00170.
11. A. Podkopaev, A. Ramdas. "Distribution-free uncertainty quantification for classification under label shift". UAI 2021 (PMLR v161). arXiv:2103.03323.
12. T. Kanamori, S. Hido, M. Sugiyama. "A Least-squares Approach to Direct Importance Estimation". JMLR 10 (2009).
13. A. N. Angelopoulos, S. Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification". arXiv:2107.07511.
14. C. K. Chow. "On optimum recognition error and reject tradeoff". IEEE Transactions on Information Theory 16(1), 1970.
15. R. El-Yaniv, Y. Wiener. "On the Foundations of Noise-free Selective Classification". JMLR 11, 2010.
16. Y. Geifman, R. El-Yaniv. "Selective Classification for Deep Neural Networks". NeurIPS 2017. arXiv:1705.08500.
17. C. Elkan. "The Foundations of Cost-Sensitive Learning". IJCAI 2001.
18. C. Cortes, G. DeSalvo, M. Mohri. "Learning with Rejection". ALT 2016.
19. D. Hendrycks, K. Gimpel. "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks". ICLR 2017. arXiv:1610.02136.

### 4.2 No verificadas (NO usar como soporte de claims)

- **"Tong et al." (cierre de la calibración bajo post-procesamiento):** no encontrada tras 4 intentos de verificación (WebSearch, arXiv API, Semantic Scholar). El resultado correspondiente va **demostrado autocontenido** como Lema A de este advisory; si el autor localiza la referencia, puede añadirse como soporte, no es necesaria.
- **Ben-David et al. 2010** (teoría de aprendizaje entre dominios, fuente canónica de la identidad BER–TV): acceso no verificable (paywall/anti-bot). Omitida; el Lema BER–TV va demostrado autocontenido (§2(b)).

---

## 5. Cambios propuestos a la spec F2

1. **§3.1.3 (tras la deducción rubrik de D9):** añadir nota de cierre de Q1 — los gates v0.1 son funcionales de una sola cabeza (tabla del advisory §1(b)); su error de calibración queda acotado por la deficiencia de su cabeza (Lemas A–C del advisory); **no existe cota útil para composiciones de probabilidad entre cabezas** (Prop D: dos cabezas calibradas del mismo evento → AND con ECE 1/4); el controlador compone **decisiones booleanas** entre gates, no probabilidades; cualquier gate compuesto futuro debe ser rubrik-reducible o medir su ECE conjunto (diagnóstico cross-head del advisory §1(d)(3)). Marcar Q1 resuelta.
2. **§3.3 (D1, mecanismo del unknown):** re-justificar τ_abs=0.5 como **regla de Bayes con rechazo bajo costes uniformes declarados** ($\tau_{\text{abs}}=1-\lambda_r$, $\lambda_r=0.5$; Chow 1970 como caso 0-1), no "regla de Chow sin costes"; añadir la regla general $\min_a\sum_k p_k c(a,k)>\lambda_r$ como política del controlador; añadir la frase de **support mismatch** bajo LOAO (calibración condicional en nativos; abstención = detección por semejanza, Hendrycks & Gimpel 2017). Marcar Q3 resuelta.
3. **§5.2 (bullet "Gates derivados (D9)", línea "heredan calibración"):** precisar a "heredan la (des)calibración de su cabeza de forma acotada (Lemas A–C del advisory); E3 mide el ECE de cada gate directamente".
4. **§7 (E2):** hacer explícito el sweep de τ_abs (grid 0.05–0.95) y los dos ejes LOAO: curva risk-coverage estándar + AURC; y por familia excluida, %unknown($f$, τ_abs) y falsa atribución (1−%unknown) como riesgo en filas de la familia.
5. **§7 (E3):** añadir (i) reliability por condicional CORN ($q_1$ en todas las filas; $q_2$ en filas elegibles sev≥1) para atribuir el ECE del gate $C_2$; (ii) classwise reliability de Choice ($\mathbb{E}[\mathbf{1}[y{=}k]\mid p_k]$ por clase), que soporta el claim del gate $g_{\text{recon}}$.
6. **§9 (Q2):** reemplazar el default por el protocolo scoped del advisory §2: garantía exacta solo en splits intercambiables (UNSW oficial, CIC `split_strat`); EMBER y CIC `split_time` sin promesa de cobertura — coverage empírico por semana con IC Clopper-Pearson (±0.24% global, ±0.34% por clase), unweighted vs weighted (clasificador de dominios, odds clipeado [0.1,10]), cota citable del gap (Barber et al. 2023, ≈2·$d_{TV}$, con $d_{TV}$ acotado inferiormente vía BER), challenge excluido de toda promesa (0 benignos verificado; coverage de positivos como estrés adversarial). Marcar Q2 resuelta.
7. **§8.2 (head_meta):** nota de una línea — τ_abs/τ_det son parámetros de coste declarado (defaults = costes uniformes; $\tau_{\text{abs}}=1-\lambda_r$); sin cambio de serialización.
