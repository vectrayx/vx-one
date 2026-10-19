# Literature Review — VectrayX-One

**Documento de trabajo** (2026-10-16). Alimenta la sección *Related Work* del paper y las decisiones de diseño. No es la redacción final: es el mapa de fuentes con el estado de verificación de cada una.

**Convención de estados:**
- ✓✓ Verificado hoy (2026-10-16) contra la fuente (link/DOI confirmado)
- ✓ Clásico estable de la literatura (alta confianza; verificar DOI/páginas exactas antes de enviar)
- ⚠️ Por verificar — NO citar sin confirmar primero

---

## 1. El paradigma "System One": modelos de decisión tipados no-autorregresivos

**El hueco central de este paper:** esta categoría nació comercialmente entre sept. y oct. de 2026 y **no tiene ningún paper académico**. VectrayX-One sería la primera formalización académica de las primitivas y su primera evaluación en seguridad de redes.

### 1.1 Origen conceptual

- **Kahneman, D. (2011). *Thinking, Fast and Slow*.** FSG.
  - **Citamos:** la distinción System 1 (rápido, automático) / System 2 (lento, deliberativo) que da nombre al paradigma.
  - Estado: ✓ (clásico).

### 1.2 Jev (TypeSafe AI) — referencia del draft ✓✓

- **TypeSafe AI, "Jev"** — primer modelo "System One", early access sept. 2026. Sitio oficial: https://typesafe.ai · Docs: https://docs.typesafe.ai/introduction (consultado 2026-10-16).
  - **Citamos:** (a) definición de la clase de modelo ("System One Models are a new class of AI model built for decisions inside software"); (b) el algoritmo de entrenamiento **RLCD (Reinforcement Learning for Calibrated Decisions)** — terminología de producto, sin paper; (c) claims de velocidad/costo (193.6x más rápido, 444.6x más barato; $0.042/1M input tokens, output gratis) como motivación industrial.
  - **Nota:** jevmodel.org, learnjev.com, jevnotes.com, etc. son guías de terceros NO oficiales — no citar como fuente.
  - Estado: ✓✓ (sitio oficial); ⚠️ (números de benchmark: no reproducibles aún, citar solo como claim del proveedor).

### 1.3 Microsoft-Decision-1 — la fuente primaria más fuerte ✓✓

- **Microsoft, "Microsoft-Decision-1: Our model for fast decision-making"** (anuncio oficial, 9 oct. 2026, Achint Srivastava, VP Office of the CTO). https://commandline.microsoft.com/microsoft-decision-1-model-foundry/ (consultado 2026-10-16).
  - Post-trained de **Qwen3.5-9B**; "single-pass decision scoring"; probabilidades calibradas por opción; P50 35x más rápido que GPT-6 Sol; 1.3% de flips ante perturbaciones de 8 vías; disponible en Microsoft Foundry y OpenRouter.
- **Microsoft Learn, "Deploy and use Microsoft-Decision-1 in Microsoft Foundry"** (docs oficiales, actualizado 9 oct. 2026). https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision (consultado 2026-10-16).
  - **La definición canónica de las tres primitivas** (tipos de pregunta del API):

  | Primitiva | Pregunta | Salida oficial | Uso declarado |
  |---|---|---|---|
  | `noul` | ¿Es esto cierto? | Probabilidad 0–1 | Gates y filtros |
  | `choice` | ¿Cuál opción es? | Una opción + prob. por opción | Clasificación y routing |
  | `score` | ¿Cuánto? | Escala ordinal + prob. por nivel | Ranking y grading |

  - El request pattern: `state` (texto/JSON) + `questions` tipadas; endpoint literal `/providers/microsoft/v1/systemone`.
  - Limitaciones declaradas por Microsoft (relevantes para nuestra sección de limitaciones): calibración "más fuerte en tipos de tarea familiares", sensibilidad al orden/fraseo de opciones, sin explicaciones.
  - Estado: ✓✓.
- **MarkTechPost (9 oct. 2026)** — cobertura del lanzamiento: https://www.marktechpost.com/2026/10/09/microsoft-ai-releases-microsoft-decision-1-a-qwen3-5-9b-decision-scoring-model/ — citable como cobertura de prensa, no como fuente técnica. ✓✓

### 1.3 Ecosistema adyacente ⚠️

- **OpenAI "Decisions"**: mencionado en el anexo del anuncio de Microsoft y en blogs de terceros (~120–200 ms). No verificamos docs oficiales todavía. ⚠️
- **Modelos de decisión open-weight** (H2O-Lightning-4B, Strands-Decider 2B, Quyet-1.0-Large, Surogate Rune 26B-A4B, deck-31B): conocidos solo por el apéndice de benchmarks del anuncio de Microsoft. ⚠️ Si aparecen en el paper, será como "modelos comparados por Microsoft (anuncio, apéndice A)".
- **JevBench** (leaderboard de terceros): ⚠️ — citado por Microsoft pero su metodología es ajena a ambos.
- **Cobertura de contexto** (blogs, NO fuentes primarias): [Análisis System One vs Two](https://www.narenvadapalli.com/blog/system-one-system-two-models-microsoft-decision-1-jev/), [Under the Hood (Substack)](https://pragmaticai1.substack.com/p/jev-under-the-hood).

---

## 2. Fundamentos de ML del enfoque (nuestro marco teórico)

Las primitivas System One no se inventaron de la nada: descansan en literatura estable que debemos citar para que la spec formal (F0) tenga cuerpo académico.

### 2.1 Inferencia no-autorregresiva

- **Gu, J., Bradbury, J., Xiong, C., Li, V.O.K., Socher, R. (2018). "Non-Autoregressive Neural Machine Translation".** ICLR 2018.
  - **Citamos:** origen del término "no-autorregresivo" en deep learning — la generación completa de la salida en una sola pasada forward, sin el bucle de muestreo condicional. Es el fundamento técnico de por qué un System One engine es de una pasada.
  - Estado: ✓ (clásico).

### 2.2 Calibración de probabilidades (el corazón de `Noul`)

- **Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods".** En *Advances in Large Margin Classifiers*, MIT Press. — Platt scaling. ✓
- **Zadrozny, B., Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates".** KDD 2002. — extensión multiclase + evaluación de calibración. ✓
- **Niculescu-Mizil, A., Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning".** ICML 2005. — comparación empírica de métodos de calibración por familia de modelo (relevante: boosted trees tienden a distorsionarse, justifica nuestra crítica a XGBoost sin calibrar). ✓
- **Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks".** ICML 2017.
  - **Citamos:** las redes modernas están mal calibradas por defecto; temperature scaling como fix práctico. Es la cita que ancla nuestro experimento E3 (ECE + reliability diagrams). ✓
- **Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*.** Springer. — conformal prediction. ✓
- **Angelopoulos, A.N., Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification".** arXiv:2107.07511.
  - **Citamos:** garantías de cobertura distribution-free — conecta `Noul` con la literatura de conformal (una `Noul` con umbral de política es un caso degenerado de conformal gate; discutirlo eleva el rigor de la formalización). ✓

### 2.3 Abstención y escalada (la "Zona Gris")

- **Chow, C.K. (1970). "On Optimum Recognition Error and Reject Trade-off".** IEEE Trans. on Information Theory. — el reject option clásico. ✓
- **Geifman, Y., El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks".** NeurIPS 2017.
  - **Citamos:** el fundamento de "escalar a humano/modelo cuando la confianza es baja" — exactamente el mecanismo que Microsoft describe ("use a low top probability as a signal to escalate"). ✓

---

## 3. Defensa perimetral determinista (Fase 1)

- **Roesch, M. (1999). "Snort — Lightweight Intrusion Detection for Networks".** LISA '99. — el motor de firmas canónico. ✓
- **Suricata** — Open Information Security Foundation (OISF), https://suricata.io. Motor de inspección multihilo, reglas compatibles con formato Snort. **Nuestra Fase 1 de referencia** (declarar en el paper como implementación de referencia, no como VectrayX). ✓✓ (software existente).
- **Paxson, V. (1999). "Bro: A System for Detecting Network Intruders in Real-Time".** Computer Networks 31(23-24). — Zeek, análisis comportamental de flujos. ✓
- **YARA** — VirusTotal, https://virustotal.github.io/yara/. — firmas de malware para la rama estática. ✓
- **Limitaciones documentadas** (rigidez, ceguera a día cero, explosión de reglas): tratar con surveys de NIDS (⚠️ elegir 1–2 surveys 2020+ y verificar) + McHugh/Mahoney & Chan (sección 5).

---

## 4. ML para detección de intrusos en red (NIDS)

- **Mirsky, Y., Doitshman, T., Pinkas, B., Elovici, Y. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection".** NDSS 2018.
  - **Citamos:** el ML rápido en la línea de fuego YA existe — Kitsune procesa features de paquetes en tiempo real sin inspección de payload. **Debemos diferenciarnos explícitamente:** Kitsune es no-supervisado, entrega scores de anomalía sin calibración garantizada ni decisiones tipadas, y no razona sobre contexto textual. Es el pariente académico más cercano y el baseline conceptual obligatorio. ✓
- **McHugh, J. (2000). "Testing Intrusion Detection Systems: A Critique of the 1998 and 1999 DARPA Off-line Intrusion Detection System Evaluations as Performed by Lincoln Laboratory".** ACM TISSEC 3(4).
  - **Citamos:** el precedente clásico del escrutinio de datasets sintéticos — establece la tradición crítica que seguimos al auditar CIC-IDS2017. ✓
- **Mahoney, M.P., Chan, P.K. (2003). "An Analysis of the 1999 DARPA/Lincoln Laboratory Evaluation Data for Network Anomaly Detection".** RAID 2003. ✓
- **Engels, G., et al. (2021). "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study".** IEEE. https://ieeexplore.ieee.org/abstract/document/9474286
  - **Citamos:** errores de etiquetado documentados en CICIDS2017 (archivos/flujos mal etiquetados). Justifica usar el dataset con auditoría de etiquetas o el disclaimer correspondiente. ✓✓ (título/venue verificados hoy; ⚠️ completar lista de autores antes de citar).
- **"Errors in the CICIDS2017 Dataset and the Significant Differences in Performance of Machine Learning Models Between the Original and Corrected Versions".** Springer, DOI 10.1007/978-3-031-31108-6_2 (HAL: hal-03775466).
  - **Citamos:** las diferencias de performance entre versión original y corregida — evidencia de que los resultados publicados sobre las etiquetas originales son cuestionables. ✓✓ (DOI verificado hoy; ⚠️ confirmar autores — creemos Rosay et al.).
- **Comunidad:** auditoría de corrección de etiquetas CICIDS2017 — https://github.com/sandesh20lamichhane/cicids-label-correction-audit (referencia práctica, no citable como académica). ⚠️

---

## 5. ML para malware estático

- **Anderson, H.S., Roth, P. (2018). "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models".** arXiv:1804.04737.
  - **Citamos:** el dataset EMBER clásico (1.1M PE files, features estáticas, benchmark LightGBM). Comparabilidad con literatura publicada. ✓
- **EMBER2024 (2025). "A Benchmark Dataset for Holistic Evaluation of Malware Detection".** arXiv:2506.05074 · DOI ACM 10.1145/3711896.3737431 · https://github.com/FutureComputing4AI/EMBER2024
  - CrowdStrike + FutureComputing4AI. 3.2M archivos, incluye malware evasivo, protocolo de evaluación propio.
  - **Recomendación para la fase de datos:** EMBER2024 como dataset primario (paper propio + actualidad) y EMBER2018 como secundario (comparabilidad con la literatura establecida). ✓✓ (verificado hoy; ⚠️ completar autores).
  - Cobertura: [blog de CrowdStrike](https://www.crowdstrike.com/en-us/blog/ember-2024-advancing-cybersecurity-ml-training-on-evasive-malware/).
- **Distribuciones del dataset:** GitHub de elastic/ember ya NO tiene releases (verificado hoy — arrays de releases/tags vacíos). Mirrors: [Kaggle ember-2018-v2-features](https://www.kaggle.com/datasets/dhoogla/ember-2018-v2-features) (features pre-extraídas), [HF sparrowaisolutions/ember-dataset](https://huggingface.co/datasets/sparrowaisolutions/ember-dataset). ✓✓

---

## 6. LLMs en ciberseguridad (los "System Two" de seguridad)

**Corrección crítica al draft (sección 1):** el draft llama "autorregresivos" a CyberBERT/SecureBERT con latencias de 1.5–3 s. **Es técnicamente incorrecto**: SecureBERT es un encoder tipo BERT (bidireccional, MLM), cuya inferencia de clasificación es una sola pasada de milisegundos en GPU. La crítica de la generación token-por-token aplica a LLMs decoder (GPT-style) y a agentes. La comparación honesta del paper debe ser de 4 niveles: (i) determinista, (ii) ML clásico, (iii) **encoders tipo BERT — rápidos Y semánticos** (por eso nuestro baseline fine-tuned DistilBERT es el comparador correcto, no un strawman lento), (iv) LLMs autoregresivos/agénticos (el problema real de latencia/alucinación/formateo).

- **Aghaei, E., et al. (2022). "SecureBERT: A Domain-Specific Language Model for Cybersecurity".** arXiv:2204.02685 · Springer DOI 10.1007/978-3-031-25538-0_3 · https://github.com/ehsanaghaei/SecureBERT
  - **Citamos:** el representante de modelos de lenguaje de dominio para ciberseguridad — Y la corrección anterior (el paper debe citarlo correctamente como encoder). ✓✓ (arXiv ID verificado hoy).
- **SecureBERT 2.0 (oct. 2025).** arXiv:2510.00240 (IEEE 11418060). "Advanced Language Model for Cybersecurity Intelligence". — versión actualizada; evaluar si citar 1.0, 2.0 o ambos. ✓✓
- **CyberBERT** — existe en la literatura (malicious URL/payload detection), pero NO verificamos la cita exacta hoy. ⚠️ Alternativa: no citarlo y usar SecureBERT + nuestro propio baseline DistilBERT.
- **Greshake, K., et al. (2023). "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection".** AISec @ ACM CCS 2023.
  - **Citamos:** el riesgo de prompt injection indirecto que el draft nombra en "trabajo futuro" (sanitización de telemetría con texto desestructurado — user_agent, URIs). Es la amenaza adversarial de nuestro campo `payload_context`. ✓

---

## 7. Sistemas en cascada y triage de dos fases

- **Viola, P., Jones, M. (2001). "Rapid Object Detection Using a Boosted Cascade of Simple Features".** CVPR 2001.
  - **Citamos:** el ancestro canónico de "filtro barato primero, modelo caro solo si hace falta" — la arquitectura de dos fases de VectrayX-One es una cascada de decisión. ✓
- **Literatura de alert triage en SOC** (priorización de alertas, workloads de analistas): ⚠️ elegir 1–2 referencias 2020+ y verificar.
- **Two-stage / cascada en detección** (inferencia con modelos en cascada, early-exit networks): ⚠️ opcional; si citamos, verificar (early-exit: BranchyNet/Big-Little Net son candidatos).

---

## 8. Datasets: resumen y decisión

| Dataset | Cita | Problemas documentados | Decisión propuesta |
|---|---|---|---|
| **CIC-IDS2017** | Sharafaldin, Lashkari & Ghorbani (2018), "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP ✓ | Etiquetado con errores (Engels 2021; Rosay et al.) | **Primario (red)** con etiquetas auditadas. Cubre bien los attack_vectors del draft: Slowloris ✓, DDoS/SYN ✓, DoS HTTP (Hulk/GoldenEye) ✓ |
| **CICDDoS2019** | ⚠️ (papel oficial por verificar; página UNB: unb.ca/cic/datasets/ddos-2019) | Mismo pipeline de generación que IDS2017 (CICFlowMeter) con críticas documentadas | **Secundario**: cubre amplification vectors con más profundidad; solo si es necesario para HTTP_Amplification |
| **UNSW-NB15** | Moustafa & Slay (2015), "UNSW-NB15: A Comprehensive Data set for Network Intrusion Detection Systems", MilCIS ✓ | Tráfico simulado, críticas de realismo (tradición McHugh) | **Primario (red)** — estándar establecido, 9 familias de ataque para LOAO |
| **EMBER 2018** | Anderson & Roth (2018), arXiv:1804.04737 ✓ | Dataset original con features del 2018 | Secundario (comparabilidad) |
| **EMBER2024** | arXiv:2506.05074, ACM 10.1145/3711896.3737431 ✓✓ | Nuevo (2025); menos literatura de terceros aún | **Primario (malware)** |

---

## 9. Mapa: claim del paper → literatura que lo respalda/contrasta

| Claim (draft) | Respaldo | Contraste/Límite a declarar |
|---|---|---|
| Trade-off latencia vs generalización semántica | Roesch 1999, Paxson 1999 (rigidez); SecureBERT (semántica de dominio) | Kitsune ya hace ML rápido en red — diferenciar (nosotros: tipado + calibrado + contexto textual) |
| Decisiones tipadas de una pasada | Gu et al. 2018 (NAR); docs Microsoft-Decision-1; Jev | BERT-encoders también son de una pasada — la novedad NO es "no-autorregresivo" per se, sino primitivas tipadas calibradas aplicadas al dominio |
| `Noul` = probabilidad calibrada | Platt 1999; Guo et al. 2017; conformal (Vovk; Angelopoulos & Bates) | Guo et al.: por defecto las redes están mal calibradas — hay que demostrarlo con ECE (experimento E3) |
| Zona gris / escalación | Chow 1970; Geifman & El-Yaniv 2017 | — |
| Arquitectura de dos fases | Viola-Jones 2001 (cascadas) | — |
| Generalización día cero | Protocolo LOAO (leave-one-attack-out) | Engels 2021 / Rosay et al.: resultados sobre etiquetas sin auditar son cuestionables → auditar y reportar |
| Vulnerabilidad a inyección en telemetría (trabajo futuro) | Greshake et al. 2023 | — |

---

## 10. Correcciones al draft derivadas de esta revisión

1. **Sección 1:** SecureBERT/CyberBERT NO son autorregresivos (ver sección 6). Reescribir la comparativa en 4 niveles (determinista / ML clásico / encoder BERT / LLM autoregresivo-agéntico).
2. **Sección 3 y 4.2:** `Noul` no es booleano — es probabilidad calibrada [0,1]; el umbral de política lo aplica el controlador compilado. Ajustar el ejemplo TypeScript (`requires_immediate_drop: Noul; // true`) a `noul: 0.94`.
3. **Tabla 5.2:** todas las cifras son aspiracionales hasta medirlas (E4/E5). Marcarlas como targets de diseño o sustituirlas por resultados.
4. **Inconsistencia interna:** "Acción en Microsegundos" (diagrama) vs 70–120 ms (tabla) — unificar: Fase 1 en µs, Fase 2 en ms medidos.
5. **Citas:** el draft tiene cero referencias; este documento es la base de la bibliografía. Además del origen del paradigma (Kahneman, Jev, Decision-1), citar a Kitsune obligatoriamente — es el trabajo más cercano y omitirlo sería un rechazo seguro.

## 11. Checklist de verificación pendiente (antes de escribir la bibliografía final)

- [ ] Lista completa de autores: Engels 2021, Rosay et al. (Springer), EMBER2024, SecureBERT (más allá de primer autor)
- [ ] Cita exacta de CyberBERT — o decidir omitir
- [ ] Paper oficial de CICDDoS2019 (si aparece en el paper)
- [ ] Docs oficiales de OpenAI "Decisions" (si se menciona el ecosistema)
- [ ] 1–2 surveys de NIDS recientes (2020+) para la sección de related work
- [ ] DOIs/páginas exactos de todos los clásicos marcados ✓
- [ ] Metodología de JevBench (si se menciona)
