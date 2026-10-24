# Revisión adversarial (Reviewer 2) — VX-One

## 1. VEREDICTO

**Reject (top-tier).** El trabajo es inusualmente transparente sobre resultados negativos y la auditoría Suricata con PCAP original es valiosa. Sin embargo, la contribución constructiva central de “honesty is teachable” no es auditable porque E9b contradice la línea base E2 en los mismos folds. Además, las afirmaciones de significancia usan semillas/bootstraps sin una prueba pareada definida, y la única evaluación de cascada deployable recupera sólo 0.339 de recall en un día y es una estimación de dos muestras.

## 2. FLAWS FATALES

### F1 — E9b no es consistente con su baseline E2; invalida la evidencia cuantitativa de la contribución constructiva

**Dónde.** `paper/main.tex:215` y Apéndice `:443`; `docs/EXPERIMENTS.md:963–1009`.

**Evidencia.** El paper presenta E9b como re-training de *los mismos* LOAO folds y afirma que la abstención media UNSW a `tau_abs=0.5` pasa de **0.513 a 0.950**. Pero el artefacto baseline de E2 para esos mismos nombres de corrida y ese umbral dice otra cosa:

* `results/e2/e2.json`: `unsw_small_oficial_loao_dos` = **0.0276**, `exploitation` = **0.0819**, `analysis` = **0.9232**, `backdoor` = **0.6346**, `recon` = **0.2823**.
* El sweep que sustenta el texto (`results/e2_rc/tau_sweep.json`) reproduce esa semántica a 0.5 (por ejemplo, CIC `dos_ddos` = 0.0000 y `bruteforce` = 0.000482).
* En cambio, `results/e9b_oe/e9b_oe.json`/`.md` etiqueta como `ref` del mismo experimento `dos` = **0.7102**, `exploitation` = **0.3203**, `analysis` = **0.9409**, `backdoor` = **0.9108**, `recon` = **0.2500**. El promedio simple de esos nueve `ref` es el 0.513 del paper; el promedio simple de E2 es aproximadamente **0.376** (el 37.5% que el propio paper reporta en `main.tex:215`).

No puede ser simultáneamente el mismo max-softmax calibrado, la misma población excluida y `tau=0.5`. Tampoco es una diferencia de redondeo (para `dos`, 2.76% frente a 71.02%). Sin una trazabilidad que identifique logits, temperatura, candidatos y filtro exactos, el efecto OE y sus costes (`-0.008/-0.033`) no son interpretables. **Acción requerida:** retirar E9b de abstract/conclusión/apéndice o regenerarlo desde exactamente los logits de E2 con hashes y un test de igualdad de baseline antes de publicar.

## 3. FLAWS MAYORES

### M1 — Inferencia estadística insuficiente y lenguaje de significancia no justificado

**Dónde.** Tabla 1/caption `main.tex:138`, texto `:167`, Limitations `:390`; resultados `results/e1_seed/seed_variance.json` y `results/e1e3/baseline_seed_variance.json`.

* Se dice “statistically ties” y “~3σ”, pero no hay hipótesis, estadístico, unidad experimental, emparejamiento por seed, ni *p-value*/intervalo de la **diferencia**. El bootstrap de 500 ejemplos para una sola seed (caption: VX `[.9828,.9841]`, MLP `[.9816,.9828]`) cuantifica incertidumbre de test condicional al modelo entrenado; no incertidumbre de entrenamiento ni comparación de algoritmos.
* Hay tres seeds, pero Table 1 enseña VX seed-42 y para baselines mezcla un valor seed-42 con desviaciones. Los datos disponibles sí permitirían comparar pareadamente seed 42/43/44; por ejemplo CIC-time VX = 0.7112/0.7541/0.7355 y XGBoost = 0.6372/0.6126/0.6663. Reportar “3σ” no sustituye un intervalo bootstrap pareado sobre predicciones para cada seed y un análisis jerárquico/por-seed.
* No hay CIs para FPR@95, macro-F1, ECE, Brier, ni para los cambios de ablation/LOAO/OE. Para ECE esto importa especialmente: es un estimador binned sesgado y sensible al binning; el propio JSON muestra MCE alto aun donde ECE es minúsculo (p. ej., CIC-base gate test ECE 0.00071 pero MCE 0.324, `results/e1e3/e1e3.json:116–213`).
* Se prueban muchos modelos × datasets × métricas × 16 LOAO folds × ablations, sin familia pre-registrada ni corrección FDR/Holm. Esto no invalida números descriptivos; invalida la retórica confirmatoria.

**No medido, no “mal medido”:** faltan CIs/contrastes para casi todas las métricas no-AUROC. La solución no es ocultar las cifras sino etiquetarlas como descriptivas y añadir inferencia pareada.

### M2 — La cascada “end-to-end” no está medida sobre los mismos flujos en el régimen deployment; el resultado operacional es débil

**Dónde.** `main.tex:278,295–299,392`; `results/e5/byday_phase2.json`; `results/e5_original/coverage_byday.json`.

El paper lo declara parcialmente, pero el framing de contribución (“measured cascade”, abstract/intro) sigue siendo más fuerte que la evidencia:

* `comb` usa una independencia entre el alertado de Fase 1 y Fase 2 evaluada en otro conjunto; no es un veredicto conjunto por flujo (`main.tex:278`, `:392`). Esa independencia es precisamente dudosa si Suricata falla en el subconjunto más sigiloso.
* La tabla principal usa Fase 2 estratificada, que ha visto todas las familias. El único punto estrictamente by-day evaluable es Friday: recall combinado **0.3394**, frente a **0.9958** en el ancla family-covered (`results/e5/byday_phase2.json:42–79`). Thursday no tiene F2 by-day y Mon–Thu no son computables bajo ese split (`:4–40`).
* El PCAP original sí rescata el claim acotado sobre Suricata: recall Fase 1 0.0205/0.0219/0.0208/0.0023 Tue–Fri (`results/e5_original/coverage_byday.json:19–118`). Pero no convierte la cascada completa en una medición deployable. Además el matching por 5-tupla diario con reutilización de puertos no produce una correspondencia unívoca alert-flow.

La contribución defendible es “medimos baja cobertura de este Suricata/ET sobre este threat mix; estimamos una cascada bajo supuestos.” No es una demostración de triage perimetral end-to-end.

### M3 — El objeto “unknown/honesty” no es uniforme y aún no se compara con selective classification/conformal risk de forma equivalente

**Dónde.** diseño `main.tex:98,114`; E2 `:213–246`; Discussion `:377`; E9b `:443`.

En EMBER, `unknown` es clase supervisada con 2.19M filas; en UNSW/CIC es max-softmax abstention. Son problemas distintos y el paper lo reconoce (`:246`), pero luego los agrupa como ventaja de las “typed heads”. En CIC, la semántica falla por completo: en LOAO recon AUROC = **0.4259**, FPR@95 = **0.8685**, abstención = **0%** (`results/e2/e2.json:88–103`). Por ello el claim de “event-level honesty” sólo está demostrado para la clase supervisada EMBER, no para open-world network triage.

Los baselines de max-prob, margin, logística y Mahalanobis son adecuados como controles mínimos, no como frontera. Falta comparar *risk–coverage* / AURC y una política con garantía de riesgo (conformal risk control o selective classifier calibrado) a la misma cobertura y matriz de costes; en temporal, reportar cobertura/riesgo empírico por ventana sin prometer exchangeability. El apéndice conformal binario de EMBER no resuelve la decisión multi-clase/abstención de E2.

### M4 — La historia de calibración mezcla métricas incompatibles y puede inducir al lector a interpretar “ECE” erróneamente

**Dónde.** Setup `main.tex:124`; E3 `:175`; E8/apéndice `:439`; `docs/EXPERIMENTS.md:885–912`; `results/e8_recal/e8_recal.json`.

El archivo E8 denomina `ece15` a una métrica que el documento admite no es ECE estándar: el *hard-prediction gap*. En PortScan el JSON llama ECE 0.5531→0.4452 (`results/e8_recal/e8_recal.json:40–107`), mientras el apéndice intenta corregirlo a posteriori. Aunque la corrección es honesta, tener en el mismo paper ECE estándar (E3) y “ECE15” no estándar (E8) deja una ruta fácil a una inferencia falsa de calibración. Renombrar todos los resultados E8/E9 a error duro/`hardGap`, recalcular standard ECE/NLL/Brier con CIs, y separar con tablas/figuras es indispensable.

### M5 — La comparación vendor es una prueba de prompt/representación zero-shot, no evidencia sobre el mérito de la tríada ni un baseline competitivo

**Dónde.** `main.tex:352–370,402`; `results/e7_vendors/*`.

El resultado descriptivo está respaldado: Jev completo 2021 filas AUROC 0.5677, FPR@95 0.932, ECE 0.265 (`opencode/vendor_jev_free_full.json`), y Azure AUROC 0.5436 (`vendor_azure_main.json:10–37`). Pero cada API recibe 39 features tabulares nombradas, un candidate set elegido para UNSW y un único protocolo zero-shot. No se reportan prompt alternatives, unidades/normalización semántica, ejemplos de salida, ni un modelo tabular/LLM *domain-adapted* comparable. Así, “generic triad zero-shot does not transfer under this serialization” está medido; “dos vendors de la triad están al azar” no fundamenta una crítica general de los productos ni la novedad arquitectónica.

Además hay una discrepancia de reproducibilidad: el texto dice 2,021 filas y 10,105 (=2021×5) requests (`main.tex:352`), pero `variance_azure.json:4–6` registra **2,041** filas. Para Jev, la repetición es incompleta: 451 filas, cobertura 0–5 por fila y AUROC 0.579–0.615 (`variance_opencode.json:4–17,62–86`), no una replicación completa N=5. Debe explicitarse y no presentarse como varianza comparable de ambos vendors.

### M6 — Novedad/posicionamiento: la arquitectura parece una combinación estándar; el paper no demuestra que “typing” cause la ganancia

**Dónde.** `main.tex:57,90–106,375–379`; bibliografía `paper/refs.bib`.

Shared MLP + BCE/softmax/CORN + temperature scaling + abstention es una composición razonable de componentes conocidos, tal como el texto reconoce. La ablación relevante no aísla la hipótesis: “single-heads” cambia arquitectura/entrenamiento pero en UNSW baja AUROC 0.9834→0.9784 y sube macro-F1 0.4571→0.4598 (`main.tex:316–321`); no prueba que la semántica tipada, frente a tres cabezas independientes o un backbone multitask sin controller, dé honestidad. En CIC, `unknown off` es un no-op (misma macro-F1 0.8826), porque allí `unknown` no es clase entrenada (`:323–327`).

Falta posicionar el método contra literatura moderna de **conformal risk control**, *selective classification under shift*, open-world/OE y decision-making learned policies. No pido un “Decision Transformer baseline” de paja para tabular IDS; pido explicar por qué una arquitectura de policy/trajectory no aplica al evento aislado y comparar contra el estándar de abstención que sí aplica. La bibliografía no contiene un trabajo posterior a 2023 de esos frentes (salvo EMBER2024/KDD’25), por lo que “first academic instantiation” reetiqueta una combinación conocida más que establecer novedad de seguridad.

### M7 — Amenazas de validez externas y reproducibilidad operacional aún limitan el claim de línea de fuego

**Dónde.** `main.tex:88,120–126,252–269,386–402`; `docs/EXPERIMENTS.md:239–255`.

* Sólo hay benchmarks offline; los flujos CIC son artefactos de capturas generadas y la by-day evaluación concentra Friday en DDoS/PortScan. No se midieron tráfico empresarial, ataques adaptativos/evasivos, throughput/concurrencia, backpressure, memoria ni efecto de FPs de Suricata sobre una operación real.
* Latencia declara “exact hardware recorded”, pero la documentación disponible sólo dice `local-cpu`, un hilo, 3,000 eventos (`EXPERIMENTS.md:241–255`): no CPU, frecuencia, OS/kernel, PyTorch/BLAS, warm-up, afinidad ni intervalo. Es una microbenchmark útil, no prueba de inline deployment ni comparación de coste.
* Score/severity tiene labels rubric-derived (`main.tex:386`), por lo que MAE de severidad y el gate derivado no validan un riesgo/severidad humano u operacional. Esto está declarado, pero contradice parte del tono de “ordinal score” calibrado.

## 4. FLAWS MENORES

1. **Claridad:** “unknown” alterna entre clase real, abstención max-softmax y pool OE; usar tres nombres/columnas separados en todas las tablas.
2. **Tabla 1:** combina seed-42 para VX con medias/desviaciones de baselines; mostrar media±CI para *todos* o una tabla pareada por seed. Incluir ECE/Brier/NLL en el PDF, no sólo artefactos.
3. **Tabla 5:** `accuracy` domina con benignos y la caption dice que es estimada; eliminarla o relegarla, poner cobertura, AURC y CIs de matching/estimación.
4. **E2 micro-folds:** CIC exploitation n=1 e infiltration n=5 no deben aparecer como AUROC comparables; separarlos de agregados y de la narrativa cuantitativa.
5. **Bibliografía:** predominan fuentes fundacionales (1999–2023). `angelopoulos2021` es un arXiv tutorial, no una referencia suficiente para la frontera de garantías operacionales.
6. **Texto/formalismo:** hay mucho espacio dedicado a lemas que no compensa los huecos experimentales; mover pruebas al suplemento para hacer sitio a protocolos, CIs y un esquema temporal reproducible.

## 5. EXPERIMENTOS FALTANTES PRIORIZADOS (máximo 6)

| Prioridad | Experimento | Claim que desbloquea | Coste/factibilidad |
|---|---|---|---|
| 1 | **Reconciliar y rerun E9b**: mismo checkpoint/logits que E2, IDs disjuntos train/test, hash de filas, misma T/candidatos; evaluar base y OE con 3 seeds y CIs pareados. | “OE enseña honesty” y coste conocido. | Alto impacto, bajo–medio coste; scripts/NPZ y runner OE ya existen. |
| 2 | **Inferencia pareada completa E1/E2/E6**: seed-paired bootstrap de diferencias AUROC/FPR@95/F1/Brier/ECE, 95% CIs; Holm/FDR sobre familia declarada. | “tie”, “wins under drift”, y efectos de ablación. | Alto impacto, bajo coste; logits de 3 seeds ya están presentes para E1 y parte de E6. |
| 3 | **Cascada por flujo en orden temporal**: alimentar F2 con exactamente los flujos originales que no alertó F1, sin independencia; train anterior al día y evaluar cada día posible, o declarar el único Friday. | Claim de cascade end-to-end y cobertura/riesgo de deployment. | Alto impacto, medio–alto; EVE, PCAP y matching ya existen, pero faltan artefactos temporales Mon–Thu. |
| 4 | **Selective/conformal risk baseline al mismo presupuesto de cobertura**: CRC/RCPS o selective classifier sobre gate/choice, AURC y riesgo empírico por ventana/familia; compararlo con max-softmax y OE. | Que la cabeza tipada aporta más que una política selectiva estándar. | Alto impacto, medio; logits y scripts de risk-coverage/conformal existen. |
| 5 | **Ablación causal de typing**: mismo trunk y capacidad: (a) tres heads independientes, (b) multitask sin score/controller, (c) single multiclass con reject/OE; todas con misma tuning budget. | Novedad causal del empaquetado typed/multitask. | Medio impacto, medio coste; arquitectura y entrenamiento ya existen. |
| 6 | **Vendor protocol robustness**: 3–5 serializaciones documentadas (unidades, z-score/percentiles, descripciones de features), prompt congelado antes de test, mismas filas y repeats completos. | Claim limitado sobre zero-shot APIs; no sobre vendors en general. | Medio impacto, bajo–medio coste; API evaluator existe. |

## 6. TRES A CINCO MEJORAS “FRONTERA” CONCRETAS

1. **Convertir “honesty” en una garantía operacional, no una tasa de abstención.** Añadir en §E2 una figura `risk–coverage–unknown-detection` por familia/ventana con objetivo de riesgo predeclarado (p.ej. ≤5%); evidencia: experimento 4 y CIs binomiales. Esto compara políticas a cobertura igual y evita que más abstención parezca automáticamente mejor.
2. **Protocolo open-world streaming genuino.** En §Setup definir una secuencia `detect → label delay → OE/retrain → evaluate next unseen family/window`, sin reutilizar filas de evaluación ni conocer la familia futura. Evidencia: rerun E9b con IDs, timestamps y dos o más episodios. Este cambio separa aprendizaje tras incidente de LOAO retrospectivo.
3. **Sustituir el headline de cascada por un estimador identificable.** En §E5 reportar una matriz por flujo: F1 alert/no-alert × F2 score × label, con bootstrap por conexión/time block; si no existe, conservar sólo F1 PCAP y declarar F2 como simulación contrafactual. Evidencia: experimento 3.
4. **Medición de calibración apta para decisiones.** Reemplazar ECE aislado por adaptive/calibration curves, NLL/Brier, intervalos y error de decisión bajo una matriz de coste explícita; renombrar E8 `hardGap` y prohibir “ECE” allí. Evidencia: recomputación offline de logits, factible ya.
5. **Reformular la novedad.** En abstract/related work cambiar “first instantiation” por “auditoría empírica de una interfaz de decisiones en tres tareas”; añadir una tabla que separa qué es estándar (CORN, temperature, OE, reject) de qué aporta el paper (contrato, evidencia negativa, cascada). Evidencia nueva: ablación causal de typing, no sólo redacción.

## 7. CITAS SOTA 2024–2026 QUE FALTAN

* **Angelopoulos, Bates, Candès, Jordan, Lei, “Conformal Risk Control,” ICLR 2024** — **[confianza alta]**. Es directamente pertinente: controla una función de riesgo de decisión/rechazo y es un baseline/mejor marco para el claim de selective triage, más cercano que sólo coverage split-conformal.

No incluyo citas 2024–2026 sobre LLM-based triage o decision transformers porque no puedo verificar con seguridad una referencia concreta y pertinente de venue/año sin riesgo de inventarla. Sí conviene discutir explícitamente *Decision Transformer* como antecedente de 2021 si se mantiene ese término en el posicionamiento, pero no satisface el rango solicitado.

## Consistencia paper ↔ artefactos verificada

* **Coinciden:** E1 seed-42 de Table 1 (VX CIC-time 0.7112; XGB 0.6372), E2 LOAO principal, E5 Friday by-day 0.9707/0.3394, E5 PCAP original, y E7 main Jev/Azure coinciden con sus JSON correspondientes.
* **No coincide:** E9b `ref` vs E2/E2-RC, y `variance_azure.json` registra 2,041 filas mientras §E7 afirma 2,021/10,105 requests. Estas discrepancias requieren resolución antes de que los artefactos puedan considerarse fuente de verdad.
