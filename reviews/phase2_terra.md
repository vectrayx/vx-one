# Revisión de correcciones (Reviewer 2, fase 2)

## Veredicto actualizado: **Borderline**

La discrepancia fatal E9b/E2 fue corregida: el artefacto canónico usa el mismo estimador que E2 y reproduce sus referencias. La evidencia constructiva UNSW ahora es auditable y el resultado negativo CIC se formula correctamente como detección. Quedan problemas corregibles pero importantes de trazabilidad documental, atribución bibliográfica y sobreafirmación del protocolo estadístico.

| Verificación | Estado | Evidencia exacta |
|---|---|---|
| E9b bajo contrato E2 | **OK** | `e9b_oe_contract.json` declara `vx_calibrated (temperature + omega de-tilt)`. UNSW: abst. `0.3754→0.8789`; $\lambda=.2/.5/1$: `0.7590/0.8789/0.9001`; ruido `0.3541`; false-att. `0.5497→0.0925`; absorción `0.0749→0.0286`; F1 `0.4761→0.4690` ($-0.0071$); gate peor `-0.0449` (generic), mejor `+0.0190` (exploitation). Coincide con §E2/§E9b tras redondeo. |
| E9b CIC como detección | **OK en el paper/abstract; FALLA documental** | JSON: abst. `0.0001→0.0000`, gate `0.7826→0.9689`, recon `0.4259→0.9909`, F1 `+0.0392`, botnet `-0.2602`. `paper/main.tex:28,215,450` no conserva el claim como resultado; sólo lo identifica como artefacto. Pero `docs/EXPERIMENTS.md:985–1008` aún afirma `0.513→0.950`, CIC `0.00→0.94` y absorción `→0.00`. |
| Bootstrap y CIs E1 | **OK, con salvedad de inferencia** | El código usa el mismo `idx` para VX y baseline (`scripts/e1_paired_stats.py:171–188`). Los CIs textuales coinciden: UNSW–MLP AUROC `+0.0013 [+.0010,+.0016]`, FPR `-0.0068 [-.0116,-.0006]`, F1 `+.0124 [+.0030,+.0227]`; CIC-time–XGB AUROC `+.0740 [+.0729,+.0751]`, FPR `-.4270 [-.4292,-.4247]`; EMBER–XGB seed Δ `-.0017`. Separar bootstrap condicional por filas de variación por reentrenamiento es correcto; con $n=3$, el veredicto por semillas es descriptivo, no una prueba confirmatoria. |
| LOAO Wilcoxon/Holm | **OK** | Recalculado desde `e2.json` y ambos JSON baseline: UNSW VX–MLP $p_{exact}=0.734375\to0.73$, VX–XGB $p_{exact}=0.1640625\to0.16$, Holm `0.4921875→0.49`; CIC sin $n=1,5$: XGB gana `5/5`, mediana VX–XGB `-0.12318`, $p_{exact}=0.0625$. La exclusión está declarada (`main.tex:418`, tabla dagger `:225–238`) y es defendible por la inestabilidad extrema de AUROC con 1/5 positivos, aunque debe leerse como análisis de cinco folds, no evidencia fuerte. |
| Prop. 5 y uso E8 | **OK** | La identidad Bayesiana y la conclusión “en general” son correctas para soporte de posteriors no degenerado: label shift exige tilt vectorial; un $T$ escalar no reproduce un shift no trivial. El texto aplica correctamente la salvedad: open set viola la hipótesis $P_t(x\mid y)=P_s(x\mid y)$ (`main.tex:394,420–422`). |
| Protocolo/multiplicidad | **FALLA** | La separación row/seed es razonable, pero “pre-declared” y “No interval or test ... selected after” (`main.tex:418`) no es demostrable y el historial lo contradice en espíritu: `ee4151f` ya tenía CIs bootstrap $B=500$ por modelo; el bootstrap pareado $B=300$, Wilcoxon/Holm y criterio de veredicto se añadieron después de la revisión. Además, la familia declarada omite las cinco métricas: hay hasta $4\times3\times5=60$ contrastes de Tabla 1, sin ajuste entre métricas/configuraciones. |
| Consistencia global | **OK salvo docs obsoletos** | Abstract/Table 1 son seed-42 y texto son medias, como dice el caption. §E7 distingue correctamente principal `2,021` de varianza `2,041/10,205`. |
| `angelopoulos2024crc` | **FALLA** | El trabajo existe y es ICLR 2024 (OpenReview `33XGfHLtZg`), pero `refs.bib:370–374` atribuye autores incorrectos: son Angelopoulos, Bates, **Fisch, Lei, Schuster**, no Candès y Jordan. |

## Errores nuevos / overclaims (por gravedad)

1. **Mayor:** `docs/EXPERIMENTS.md:980–1008` conserva las cifras E9b del estimador inválido y contradice el JSON canónico y el paper; la documentación se presenta como fuente de verdad.
2. **Mayor:** el protocolo afirma pre-especificación y ausencia de selección post-resultado sin soporte histórico, y no controla la multiplicidad de las cinco métricas de Tabla 1.
3. **Menor pero factual:** la referencia CRC 2024 tiene dos coautores erróneos.

## Correcciones concretas (máximo 3)

1. Reemplazar o marcar como histórico inválido `docs/EXPERIMENTS.md:964–1009`; insertar los números de §25/`e9b_oe_contract.json` y eliminar las conclusiones `0.513→0.950` / `0.00→0.94`.
2. En `paper/main.tex:418`, cambiar “pre-declared”/“No interval...” por una descripción transparente post-review; o aportar un artefacto fechado previo. Delimitar los CIs/p-values de Tabla 1 como exploratorios o corregir la familia de $60$ contrastes.
3. Corregir `paper/refs.bib:370–374` a Angelopoulos, Bates, Fisch, Lei y Schuster (ICLR 2024).
