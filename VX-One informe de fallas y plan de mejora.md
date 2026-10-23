# VX-One: informe de fallas y plan de mejora

Este informe se basa solo en el PDF de la versión más reciente. No he visto el código ni los datos, así que los puntos marcados como "verificar" son sospechas a confirmar, no conclusiones.

## Veredicto

Es un estudio empírico honesto, con buenas prácticas (negativos reportados, semillas, auditoría de etiquetas, enlaces a código y pesos) y una línea E8/E9 cada vez más interesante. No es un resultado de frontera: lo mejor que puede ofrecer es una caracterización cuidadosa de dónde fallan las decisiones tipadas. Los problemas más serios (cascada E5, falta de baselines en LOAO, ruido en ablaciones) siguen abiertos.

## Fallas, por gravedad

### Críticas

**1. La cascada E5 probablemente mide la reconstrucción, no a Suricata.**

- En la Tabla 5, la accuracy de Fase 1 es casi exactamente 1 − prevalencia de ataque: 0.636 el miércoles y 0.589 el viernes. Es decir, Fase 1 detecta prácticamente cero ataques, incluidos PortScan y DDoS, que no dependen del payload. Eso apunta a que reconstruir PCAPs desde features de flujo destruyó la señal.
- No se explica por qué no se usó el tráfico original con payload.
- El abstract dice que la gray zone es "\~99.9% of the traffic", cuando el cuerpo habla de "daily attacks". Son cosas distintas.
- No se dice qué datos entrenaron a Fase 2, y las métricas parecen calculadas sobre días completos. Además, el AUROC por-día del modelo es 0.71, mientras que en E5 el viernes da precisión 0.998 y recall 0.996. Si es el mismo modelo, no cuadra; si es otro, hay que decirlo.

**2. "Honestidad ante lo desconocido" sin baselines ni costes.**

- LOAO no incluye XGBoost/RF/MLP ni métodos OOD estándar (energy, Mahalanobis). Un clasificador binario entrenado con el resto de familias probablemente detecta familias no vistas igual de bien, así que no se aísla el aporte del diseño tipado.
- En UNSW y CIC, "unknown" es max-softmax, no una clase entrenada, aunque el abstract dice que el choice "always includes" esa opción.
- La frontera honestidad/utilidad (Fig. 1) usa solo filas de familias excluidas. Falta la curva de abstención sobre familias conocidas para ver el coste real.
- Hay folds con n=1 y n=5, y todos los porcentajes de abstención de la Tabla 2 son enteros, incluso con n=44 (worms) y n=18,871. Verificar si hubo submuestreo.

**3. Las victorias de detección están dentro del ruido.**

- UNSW: 0.9834 vs 0.9818/0.9821, sin intervalos para los baselines. Sin la rama de texto (a1), VX-One baja a 0.9716, por debajo de todos ellos. No se confirma que los baselines reciban los campos proto/service/state.
- Por-día: 0.711 vs 0.637 con sd de 0.018. Que FPR@95 sea 1.0 con AUROC 0.637 sugiere empates o saturación de scores.
- El macro-F1 por-día de la Tabla 1 (0.4518) es el máximo del rango de semillas (media 0.344), y XGBoost saca 0.8336 sin explicación.

### Importantes

**4. Calibración bajo drift: comparaciones y cifras dudosas.**

- El abstract compara ECE 0.04 (UNSW) con 0.40 (CIC): son datasets distintos. La comparación válida es CIC estratificado (≤0.001) frente a por-día. Además, el texto cita la Tabla 1 para el ECE, pero esa tabla no lo incluye, ni tampoco hay ECE de los baselines.
- E8: el texto dice que un mapa constante cerca del prior queda "exactamente" en el plateau de ECE 0.445 con Brier 0.246. Para una constante c, Brier = p(1−p) + (c−p)²; con c≈p eso es ≤0.25 y el ECE tiende a 0. Brier 0.246 es compatible con un prior cercano a 0.5, pero no con un ECE de 0.445. Verificar cómo se calcula.
- El isotonic "day-self" con ECE 0.260 se ajusta y evalúa sobre los mismos datos, así que queda casi calibrado por construcción. Verificar si hay validación cruzada.
- La conclusión "el techo es la discriminación" mezcla dos cosas. Lo que se muestra es que los mapas ajustados en ventanas pasadas no cruzan el cambio de prior entre días, que es el caso típico de label shift. No se probó ninguna corrección (EM de Saerens, BBSE).

**5. Ablaciones por debajo del ruido de semillas.**

- En CIC estratificado el macro-F1 tiene sd de 0.074, con una semilla en 0.73. El efecto de a4 (+4.9) es menor que 1 sd, y a6 (−14.7) es casi idéntico a la semilla baja. El propio §6 los llama "noise-borderline", pero §5.5 los presenta como hallazgos que "sobreviven".
- El run a2 se cita pero no aparece en la Tabla 6.
- a5 compara 11 vs 10 clases. El 85% de wacatac atribuido falsamente corresponde al modelo con unknown, no al "benchmark-optimal" que se menciona en §5.7.
- No hay ablación del diseño central: multitarea vs cabezas separadas, ponderación de incertidumbre o cabeza ordinal.

**6. Baseline comercial (E7) poco comparable.**

- Es zero-shot, sobre features ya transformadas, contra un modelo supervisado en el dominio. No se nombra proveedor ni versión, no hay repeticiones ni intervalos, y la muestra es estratificada.
- El ECE de "Ours" difiere entre tablas (0.097 es del gate y 0.0496 del choice), lo cual puede confundir.
- §4 todavía dice que este baseline "was not executed this cycle", mientras §5.6 lo reporta.

### Menores

- **Teoría:** §3.4 cita "Prop. 4" para la composición, pero en el apéndice es la Prop. 3. Además, τ=0.5 corresponde a λ\_r=0.5, no a "unknown cuesta lo mismo que una atribución errónea" (eso sería λ\_r=1). El Lema 1 es poco informativo para el umbralizado, y la "certificación" de d\_TV es una estimación puntual sin intervalo.
- **Redacción:** la nota *a* de la Tabla 1 está sobre FPR@95; "our baselines include appropriate encoders" no corresponde a nada (no hay encoders); "matríces", "CIDS2017", "τ\_abs ∈ {0.05, 0.99}" (es un rango) y "D11-like discipline" sin definir.
- **Latencia:** no se reporta la de los baselines ni la del feature stage φ.
- **Alcance:** todo es benchmark público; el propio §6 aclara que LOAO no mide robustez ante ataques reales.

## Plan de mejora

### Fase 1 — Arreglos que cambian las conclusiones (prioridad alta)

1. **Repetir E5 con tráfico con payload.** Ejecutar Suricata sobre los PCAPs originales de CIC-IDS2017 y comparar contra la reconstrucción. Si Fase 1 sigue sin detectar PortScan/DDoS en tráfico original, el problema es la configuración. Aclarar los datos de entrenamiento de Fase 2 y por qué el viernes da 0.998/0.996 si el AUROC por-día es 0.71.
2. **Baselines en LOAO y en EMBER-drift.** Añadir un clasificador binario por familia excluida, energy/Mahalanobis y un modelo entrenado con clase unknown. Reportar curvas risk–coverage que incluyan familias conocidas.
3. **Más semillas y baselines con la misma disciplina.** Al menos 5 semillas para VX-One y para cada baseline, con intervalos. Confirmar que los baselines reciben las mismas features categóricas, y repetir la comparación sin la rama de texto de forma pareada.
4. **Reescribir el abstract** con las cifras correctas: gray zone "of attacks", comparar ECE dentro del mismo dataset, y no afirmar que choice "always includes" unknown cuando no es así en UNSW/CIC.

### Fase 2 — Calibración bajo drift

5. **Revisar el cálculo de E8.** Documentar cómo se obtiene el ECE de un predictor constante, y repetir el isotonic "day-self" con validación cruzada.
6. **Probar label shift explícitamente.** Estimar el nuevo prior con EM (Saerens) o BBSE y recalibrar. Si eso recupera el ECE, la conclusión cambia de "el techo es la discriminación" a "falta corregir el prior".
7. **Comparar ECE de todos los métodos** en el mismo split (estratificado vs por-día) y añadir los baselines a la tabla de calibración.

### Fase 3 — Ablaciones y baseline comercial

8. **Usar más semillas en las ablaciones** y reportar solo los efectos que superen el ruido. Mover a "inconcluso" los de a4 y a6 en CIC. Incluir el run a2 o quitar la referencia.
9. **Añadir ablaciones del diseño central:** multitarea vs cabezas separadas, ponderación de incertidumbre vs fija, y la cabeza ordinal. Medir también el modelo sin unknown con los mismos folds de drift para poder afirmar lo del 85%.
10. **Rehacer E7 con rigor:** nombrar modelo y versión, repetir las llamadas, dar intervalos, y comparar también con features con semántica (nombres de campo), no solo con el estado transformado. Corregir §4.

### Fase 4 — Escritura y teoría

11. **Corregir** la referencia a la Prop. 3, la equivalencia τ ↔ λ\_r, la nota *a* de la Tabla 1 y los errores de redacción listados.
12. **Reportar latencia** de baselines y del feature stage φ, para que la comparación de velocidad sea justa.
13. **Rebajar o reformular** los lemas informativos (Lema 1, d\_TV) y añadir intervalos a la cota de drift.

### Para acercarse a la frontera (opcional, mayor esfuerzo)

- Evaluación en tráfico real o en un dataset moderno con ataques recientes, y no solo en benchmarks con artefactos conocidos.
- Un método, no solo una caracterización: por ejemplo, una cabeza de abstención entrenada (con exposición a outliers) que sí funcione en el régimen saturado de CIC, donde hoy ningún umbral post-hoc sirve.
- Cerrar la pregunta abierta del paper: un Fase 1 con payload y una cascada medida de extremo a extremo.

## Orden sugerido si hay poco tiempo

1. Repetir E5 con payload (punto 1).
2. Baselines en LOAO (punto 2).
3. Label shift en E8 (punto 6).

Con estas tres ya se sabría cuáles de las conclusiones principales sobreviven.

## Advertencia

Este informe es una lectura del PDF, no una replicación. Algunas de mis sospechas (E5, E8) pueden tener una explicación en el código o los datos que el texto no recoge.

---

## Estado al 2026-10-22 (tras dos ciclos de corrección; quien escribe: el agente, no el autor del informe)

Resuelto y verificado en `results/` + paper (13 págs.):

| # | falla del informe | estado |
|---|---|---|
| 1 | E5 mide reconstrucción, no Suricata | **cerrado doble**: (a) PCAPs originales con payload (md5 verificados): ET resuelve 0.2–2.2% → la zona gris NO es artefacto de la reconstrucción; (b) Phase-2 honesto por día: recall combinado viernes 0.339 (= 44% share × 0.77 recall DoS/DDoS; PortScan es agregación temporal fuera de alcance por-evento). Matching validado: presencia 5-tupla = 1.000, cobertura con IP atacante = global. |
| 2 | Falta de baselines en LOAO | medido (9 UNSW + 7 CIC + logreg + Mahalanobis): **empate en AUROC** — VX gana 4/9 folds UNSW, MLP 3; en CIC XGBoost transfiere mejor en botnet/recon. Paper lo declara como tie. |
| 6 | Label shift en E8 | medido (BBSE-hard/soft, EM): fallan (q̂≤0.04 vs 0.555) — pero **reencuadrado**: el shift es open-set (violación del supuesto P(x|y)), no prueba de "prior no estimable". |
| 5 | ECE de constante / isotonic day-self | **causa raíz encontrada**: el `ece_bin` de las tablas E8 es el hard-gap `|mean(1[p≥.5])−mean(y)|`, no la ECE estándar; el "plateau 0.445" es 1−q (piso todo-unos), la iso day-self es soft-calibrada (ECE 0.000) y su 0.260 es el error duro del ranking. Apéndice E8/E9/limitaciones/conclusión reescritos. |
| 3 | Semillas de ablaciones | a2/a4 σ≤0.005 estables; **a6-CIC σ=0.128 (inestable)** — declarado. |
| 4 | Abstract/declaraciones | "unknown" → semántica híbrida; margen UNSW (0.983 vs 0.982) declarado; CIs bootstrap en Tablas 1-2; % de abstención exactos; folds n=1/n=5 marcados degenerados. |

Pendiente: varianza de E7 (fleet N=5), head de abstención con outlier exposure.
Veredicto de la tercera lectura (autor del informe): "estudio empírico serio de
dónde ayudan y dónde fallan las decisiones tipadas; defendible como paper de
medición en un taller o conferencia de seguridad".


### Actualización final del ciclo (2026-10-23)

- **E7 varianza (N=5)**: Decision-1 medida completa (10,205 llamadas):
  AUROC 0.540±0.002, macro-F1 0.066±0.001, MAE 0.960±0.001, σ gate por fila
  0.022 — el veredicto zero-shot es estructural. Jev: inviable hoy por el
  rate-limit del tier free (~250 req/día/IP, requeriría fleet) — único
  pendiente de E7.
- **E9b outlier exposure (el ítem opcional)**: resultó el resultado
  constructivo del ciclo — abstención media @τ=0.5 de 0.513→0.950 (UNSW,
  9 folds) y 0.00→0.94 (CIC saturado, absorción 94%→0%), gate CIC recuperada
  (recon 0.43→0.99), coste known-F1 −0.008/−0.033. "Freshness compra
  discriminación, OE compra honestidad; ninguno compra calibración".
- Paper final del ciclo: 14 páginas, Tabla 7 con dos vendedores + varianza,
  apéndice E9b, E8 con corrección de métrica (hard-gap vs ECE estándar),
  LOAO como empate declarado con CIs bootstrap.


### Cierre de frontera (2026-10-23, madrugada)

- **E1-bis**: baselines de la Tabla 1 con 3 semillas (24 corridas). Resulado
  honesto: MLP empata a VX en UNSW in-distribución (0.9835±0.0013 vs 0.9834);
  la ventaja de VX queda en by-day CIC (~3σ sobre el ruido de semilla de los
  baselines) y en las semánticas tipadas. Abstract/Tabla/§e1 actualizados.
- **E9b-bis**: λ sweep (0.2/0.5/1.0 → abst 0.916/0.950/0.956, sin cuchillo) +
  ablation pool sintético (ruido: abst 0.452 — el mecanismo requiere la
  distribución real de la familia, no regularización genérica).
- **E7-var Jev**: N=5 por fleet (9 nodos, 1,563 llamadas, submuestra 451):
  AUROC 0.597±0.017, σ fila 0.007, desacuerdo 1.1% — Jev casi determinista,
  Decision-1 estocástico (13% flips); ninguno escapa del azar. Queda solo la
  pasada full-sample de Jev (fleet, ~4 días de ventana).
- Estado: paper 14 páginas; pendientes reales = full-sample Jev variance +
  cross-domain outlier pools. Todo lo demás del informe está medido.
