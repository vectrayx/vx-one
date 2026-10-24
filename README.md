# paper-fix-glm — Paper VX-One reconstruido desde los resultados medidos

Repo de trabajo que reconstruye el paper de **VX-One** (VectrayX-One) a partir de
los **resultados realmente medidos** del repo original (`~/projects/paper`),
no de los números aspiracionales del draft antiguo.

## Qué hay aquí

| Ruta | Contenido |
|---|---|
| `paper/main.tex` + `paper/refs.bib` | Paper final en formato **ACM (`acmart`, `sigconf`)**, en inglés, tag arXiv `cs.CL`. Compila con `pdflatex` + `bibtex` (x2). |
| `docs/EXPERIMENTS.md` | Explicación en español de los experimentos E1–E7, con las tablas copiadas de `results/` y el mapa de código. Es la fuente de verdad numérica del paper. |
| `docs/reference/` | Documentos legados congelados del repo original: draft antiguo (`VectrayX_One_Paper_Draft.md`), spec F0, lit review, spec F2, advisory Opus, nota de implementación, `data/SCHEMA.md`. **No editar** — solo referencia. |
| `src/`, `tests/`, `modal/`, `scripts/` | Código completo, copiado del repo original (idéntico, salvo `__pycache__`). ⚠️ Los tests requieren `data/` (parquets), que no se copia: para correrlos aquí, enlazar `data/` del repo original (`ln -s ../paper/data data`). |
| `results/` | Solo agregados JSON/MD por experimento (los `.npz` pesados y logs quedaron atrás). |
| `results/e6_a7/` + `scripts/a7_derived_gate_eval.py` | Ablación causal de *typing*: gate dedicado vs gate derivado de un softmax único de familia. Medias por semilla y deltas pareados por fila. |

## Decisiones editoriales tomadas (por qué este paper no es una copia del draft)

1. **Nombres de primitivas neutros.** El draft usa nombres de marca de terceros
   (`Noul`, del producto "Jev"). El paper define las primitivas formalmente:
   *gate* (probabilidad calibrada), *choice* (clasificación sobre conjunto
   cerrado con candidato `unknown`), *score* (escala ordinal, CORN). Los
   productos comerciales (Jev, Microsoft-Decision-1) se citan como motivación
   industrial, no como fundamento del formalismo.
2. **Solo números medidos.** La tabla 5.2 del draft antiguo (latencias 70–120 ms,
   `<$0.05` por millón de tokens) era aspiracional. Aquí: E4 real (p50 675–1652
   µs, batch=1, 1 hilo CPU), y el análisis de costo queda pendiente del billing
   real de Modal (declarado).
3. **Comparador honesto.** SecureBERT/CyberBERT NO son autorregresivos (son
   encoders); la comparación del paper es de 4 niveles: determinista / ML
   clásico (XGBoost, RF, MLP, Kitsune) / encoders / LLMs decoder-agénticos.
   XGBoost **gana** en CIC saturado y el paper lo dice.
4. **La novedad radica en la evaluación, no en los heads.** Los heads son
   estándar (sigmoide, softmax, CORN + temperature scaling). La ablación a7 lo
   mide: derivar el gate de un softmax único de familia empata al gate dedicado
   en AUROC/FPR@95 (UNSW ΔAUROC −0.0008±0.0013; CIC idéntico al cuarto
   decimal), así que el *typing* compra contrato y auditabilidad, no ranking.
   Bajo LOAO el empate se rompe solo donde la absorción benigna domina
   (`dos_ddos`, `botnet` en CIC); `exploitation` es inestable entre semillas.
   Los claims del paper: (a) honestidad ante ataques desconocidos métricamente (LOAO + opción
   `unknown` + abstención), (b) medición de la zona gris de una cascada real
   (Suricata+ET deja ~99.9% del tráfico en zona gris sin payload), (c)
   comportamiento de la calibración bajo drift (ECE 0.04 → 0.40 en CIC temporal).
5. **Limitaciones declaradas en el propio paper** (§ Limitations): score por
   rubrik, datasets con etiquetas auditadas, LOAO ≠ mundo real, E5 con supuesto
   de independencia documentado, Fase 1 = Suricata de referencia (no el motor
   C++ VectrayX), corridas principales con 3 seeds (LOAO/ablation/E4/E5 en 1 seed), costo pendiente.

## Estado

- [x] Experimentos E1–E6 ejecutados (ver `docs/EXPERIMENTS.md`).
- [x] **Paper `paper/main.tex` completo (ACM sigconf, con Apéndice A formal desde el advisory Opus) + `paper/refs.bib` (39 entradas; las 12 fuentes System One/productos con URLs verificadas 2026-10-16).**
- [x] **PDF compilado localmente**: `paper/main.pdf` (14 páginas, ACM sigconf, 0 citas/referencias sin resolver). Para recompilar:
  ```bash
  pdflatex -interaction=nonstopmode paper/main.tex   # desde la raíz del repo
  BIBINPUTS=paper: BSTINPUTS=paper: bibtex main
  pdflatex -interaction=nonstopmode paper/main.tex
  pdflatex -interaction=nonstopmode paper/main.tex   # los artefactos quedan en la raíz
  ```
  Los paquetes ACM que la TeX Live 2024 de Fedora no traía (`acmart`, `libertine`, `newtx`, `ec`, `cm-super`, etc.) están instalados en `~/texmf` (TEXMFHOME), fetched del histórico TL2024 — no hace falta repetir nada.
- [x] E7 (baseline comercial): corrida completa 2,021/2,021 filas por el passthrough
      System One del Zen de OpenCode (`jev-1.13-free`, mismo modelo que el SKU
      comercial, coste $0). Veredicto: vendor gate AUROC 0.568 / FPR@95 0.932 /
      macro-F1 0.069 / MAE sev 0.809 vs nuestro 0.981 / 0.157 / 0.534 / 0.333 —
      baseline comercial zero-shot a nivel de azar en gate. Resultados en
      `results/e7_vendors/opencode/vendor_jev_free_full.json`; detalle
      técnico/experimental en EXPERIMENTS.md §9 (2026-10-20).
- [x] **E7-long (ablation unknown PAIRED n=2,021 ✅ nocturna via fleet)**: las
      mismas filas del main con la opción `unknown` — gate AUROC 0.476,
      macro-F1 0.066, unknown-rate 0.35% (el hatch está disponible y el
      vendor no lo usa). El ablation subsample queda confirmado paired;
      resultados en `results/e7_vendors/opencode/unkfull/` +
      `vendor_jev_free_unkfull.json`.
- [x] E8 (recalibración por ventana walk-forward bajo drift): corrida en los
      logits del by-day CIC (2 ventanas reales: Friday-DDos → Friday-PortScan)
      con protocolo congelado `scripts/e8_window_recal.py` — walk/platt/walkacc
      + fila oracle (techo). Veredicto: recalibrar calibra pero NO detects
      — oracle ECE del gate queda en 0.445 con day-AUROC 0.56; choice ECE
      oracle 0.056 (sí recalibrable). Resultados en `results/e8_recal/`;
      integrado al paper (Limitations + Apéndice E8 + Conclusion).
- [x] E9 (per-window retraining bajo drift): continue-train del checkpoint
      oficial by-day sobre train ∪ Friday-DDos (labels retrasadas del día-1),
      protocolo congelado `scripts/e9_window_retrain.py` (local CPU, 1.4
      min/época; sensibilidad ep2/ep4). Veredicto día-2 PortScan: gate
      AUROC 0.560→0.800 (1 ep) →0.8392 (2 ep) →0.760 (4 ep, degradación);
      as-run ECE 0.553→0.554 sin cambio (el prior de composición del día
      nuevo no se infiere de labels previas, b fit −0.3…−0.7 vs +9.5
      requerido); el techo oracle-recal colapsa 0.445→0.117/0.100 con el
      ranking recuperado. Resultados en `results/e9_retrain/`
      (e9_vs_static.md, e9_oracle.json, e9_walk*.json); integrado al paper
      (Limitations + Apéndice E9 + Conclusion "five next steps").
- [x] Pendientes mínimos del advisory que reusan logits ya computados (§12 de `docs/EXPERIMENTS.md`): sweep de τ_abs ✅, reliability por condicional CORN ✅, classwise calibration ✅, diagnóstico Bayes-vs-Chow ✅, protocolo conformal EMBER (12 semanas + weighted + cota d_TV) ✅ (2026-10-20).
- [x] 3 seeds en las 4 corridas principales de tablas (Modal cuenta 2, 12/12 ok): seed-42
  reproduce la fila oficial UNSW al 4º decimal; AUROC se mueve ≤0.002 salvo CIC by-day
  (0.7336±0.0176); macro-F1 de CIC es inestable a cualquier escala (strat 0.831±0.074,
  by-day 0.344±0.085). Resultados en `results/e1_seed/`, integrados al paper (E1 + Limitations).
- [x] Resolver los ⚠ de `paper/refs.bib` y el venue de CORN (verificado contra
  fuentes primarias 2026-10-20): `engels2021`→`engelen2021` (SPW 2021 — el
  primer autor es Engelen, no Engels), `rosay2022errors`→`lanvin2022errors`
  (CRiSIS 2022, Lanvin/Gimenez/Han/Majorczyk/Mé/Totel — autores Rosay eran de
  otro paper), `botacin2024ember`→`joyce2025ember` (KDD 2025, Joyce et al. —
  no Botacin), SecureBERT = SecureComm 2022 (Aghaei/Niu/Shadid/Al-Shaer) y
  CORN = PAA 26(3):941–955, 2023, DOI 10.1007/s10044-023-01181-9.

## Mejoras 2026-10-22 (informe de fallas externo → ciclo de medidas)

Aplicado sobre el plan `VX-One informe de fallas y plan de mejora.md`; detalle en `docs/EXPERIMENTS.md` §§14–17:

- **E8b — label shift explícito** (`scripts/e8_labelshift.py` → `results/e8_recal/e8_labelshift.*`): BBSE/EM no pueden estimar el prior del día nuevo desde un modelo confidente-equivocado (q̂ ≤ 0.04 vs 0.555); el "plateau 0.445" del apéndice E8 era la constante en la media del score crudo, no del prior. El techo desplegable es discriminación **y estimabilidad del prior**.
- **E5c — Phase-2 por-día honesto** (`scripts/e5_phase2_byday.py` → `results/e5/byday_phase2.*`): con el artefacto por-día (que nunca vio las familias del viernes) el recall combinado del viernes cae de 0.996 a **0.339** (P 0.971); DoS/DDoS se recupera (0.77), PortScan no existe como candidato. La tabla 5 queda anclada como cota superior family-covered.
- **E5d — Fase 1 con payload real** (`scripts/e5_original_match.py` → `results/e5_original/`): pcaps ORIGINALES de CICIDS2017 (md5 verificados) + Suricata 7.0.10 + ET Open 52,706 reglas: resuelve 0.2–2.2% del volumen diario de ataques (DDoS 0.45%, PortScan 0.06% el viernes; Heartbleed/Infiltration ~100%). **La zona gris sobrevive a payload completo** — no era artefacto de la reconstrucción. Corrida en Modal cuenta 2.
- **E2d — LOAO baselines + OOD** (`src/baselines/run_tabular --exclude-family` + `scripts/e2_loao_ood.py` + `scripts/e2_loao_baselines_eval.py` → `results/e2_baselines/`): 9 folds UNSW (local) + 7 folds CIC (Modal cpu=32). Resultado honesto (corregido tras revisión): **empate en AUROC** — VX gana 4/9 folds UNSW, MLP 3 (medianas MLP 0.991 vs VX 0.987); en CIC XGBoost transfiere mejor en los folds débiles (botnet 0.905 vs 0.750; recon 0.830 vs 0.426) y los OOD clásicos colapsan. Todo integrado al paper (§e2, §e5, limitaciones, conclusión).
- **E7 replicado con 2º vendedor (2026-10-22)**: acceso Azure restaurado →
  Microsoft-Decision-1 (preview, deployment `decision-1` vía AI Foundry
  `services.ai.azure.com/.../v1/systemone`, mismo payload System One que Jev)
  sobre la MISMA muestra seed-42 de 2,021 filas: gate AUROC 0.544 (Jev 0.568),
  choice macro-F1 0.066, severity MAE 0.969 — **dos vendedores independientes,
  mismo veredicto zero-shot**. `results/e7_vendors/vendor_azure_main.json`.
