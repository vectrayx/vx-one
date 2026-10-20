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
| `src/`, `tests/`, `modal/`, `scripts/` | Código completo, copiado del repo original (idéntico, salvo `__pycache__`; 60/60 tests verdes en el repo original). ⚠️ Los tests requieren `data/` (parquets), que no se copia: para correrlos aquí, enlazar `data/` del repo original (`ln -s ../paper/data data`). |
| `results/` | Solo agregados JSON/MD por experimento (los `.npz` pesados y logs quedaron atrás). |

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
   estándar (sigmoide, softmax, CORN + temperature scaling). El claims del
   paper: (a) honestidad ante ataques desconocidos métricamente (LOAO + opción
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
- [x] **PDF compilado localmente**: `paper/main.pdf` (9 páginas, ACM sigconf, 0 citas/referencias sin resolver). Para recompilar:
  ```bash
  pdflatex -interaction=nonstopmode paper/main.tex   # desde la raíz del repo
  BIBINPUTS=paper: BSTINPUTS=paper: bibtex main
  pdflatex -interaction=nonstopmode paper/main.tex
  pdflatex -interaction=nonstopmode paper/main.tex   # los artefactos quedan en la raíz
  ```
  Los paquetes ACM que la TeX Live 2024 de Fedora no traía (`acmart`, `libertine`, `newtx`, `ec`, `cm-super`, etc.) están instalados en `~/texmf` (TEXMFHOME), fetched del histórico TL2024 — no hace falta repetir nada.
- [ ] E7 (baseline comercial vía API) — no ejecutado; queda como pendiente barato (~$2).
- [ ] Pendientes mínimos del advisory que reusan logits ya computados (§12 de `docs/EXPERIMENTS.md`): sweep de τ_abs + risk-coverage, reliability por condicional CORN, classwise calibration, diagnóstico Bayes-vs-Chow.
- [x] 3 seeds en las 4 corridas principales de tablas (Modal cuenta 2, 12/12 ok): seed-42
  reproduce la fila oficial UNSW al 4º decimal; AUROC se mueve ≤0.002 salvo CIC by-day
  (0.7336±0.0176); macro-F1 de CIC es inestable a cualquier escala (strat 0.831±0.074,
  by-day 0.344±0.085). Resultados en `results/e1_seed/`, integrados al paper (E1 + Limitations).
- [ ] Resolver los ⚠ de `paper/refs.bib` (listas de autores: Engels 2021, Rosay 2022, EMBER2024, SecureBERT) y el venue de CORN/shi2023 antes de la bibliografía final.
