# CONTINUE.md — Auditoría adversarial y elevación a frontera (VX-One)

> Estado vivo del pipeline multi-agente. Última actualización: 2026-10-23 (cierre de Fase 3).
> Coordinación en español; código, docstrings y artefactos técnicos en inglés.

## Estado global

| Fase | Responsable | Estado | Artefacto |
|---|---|---|---|
| 0. Ingesta y estado | Sol (orquestador) | ✅ | `CONTINUE.md` |
| 1. Vulnerabilidad + SOTA | GPT-5.6 Terra (fallback de Kimi/5.4/Claude/Grok) | ✅ | `reviews/phase1_kimi.md` |
| 2. Endurecimiento teórico | Sol + revisión | ✅ | `paper/main.tex` (Prop. 5, protocolo estadístico) |
| 3. Validación empírica | DeepSeek 4.1 / Sol | ✅ | `results/e9b_oe/`, `results/e1e3/paired_stats.json` |
| 4. Reescritura frontera | Sol | ✅ | `paper/main.tex`, abstract/§E1/§E2/§E9b |

## Fallback de modelos (incidencias)

- Kimi K2.6: rate limit. GPT-5.4 Pro: rate limit. Claude Sonnet 5.5: sin
  suscripción. Grok 4.7: sin capacidad. **GPT-5.6 Terra: OK** (reviewer).
- DeepSeek V4 Pro: OK. Mistral: no aparece en el catálogo del proveedor Azure.

## Hallazgos de Fase 1 y resolución

### V1 (CRÍTICO, resuelto) — E9b usaba otro estimador que E2
- E2 aplica `vx_calibrated` (temperatura + de-tilt ω); E9b aplicaba solo
  temperatura. Mismo checkpoint, abstención distinta (UNSW/dos 0.028 vs 0.710).
- Recalculado todo bajo el contrato único (`scripts/e9b_oe_eval.py`,
  `--selftest` reproduce E2 al cuarto decimal):
  - UNSW: abst@0.5 **0.375 → 0.879**; absorción 0.075→0.029; F1 0.476→0.469.
  - λ 0.2/0.5/1.0 → 0.759/0.879/0.900; pool sintético 0.354.
  - CIC: abst **0.000→0.000**; absorción 0.945→1.000. **El claim
    0.000→0.94 era artefacto y se retiró.** Lo que sí mejora es el gate:
    0.783→0.969 (recon 0.426→0.991; 7/7 folds).
- Paper: §E2 (sweep + contraparte), §E9b, apéndice y abstract corregidos.

### V2 (resuelto) — recuento de filas E7
- La varianza de Decision-1 usó 2.041 filas (10.205 llamadas), no 2.021.
  Corregido en §E7 con la aclaración del draw principal.

### V3 (resuelto) — CIC E9b sin artefacto
- `e9b_oe.json` solo tenía `"cic": true`; ahora `e9b_oe_contract.json`
  contiene ambos datasets y `e9b_oe.md` se regenera desde el JSON.

### V4 (resuelto) — inferencia estadística
- `scripts/e1_paired_stats.py`: bootstrap pareado de filas (B=300) para
  AUROC/FPR@95/F1/ECE/Brier + semillas 42/43/44; `--selftest` reproduce Tabla 1.
- UNSW VX−MLP ΔAUROC +0.0013 [+0.0010,+0.0016] pero a nivel semilla
  0.9837±0.0002 vs 0.9835±0.0013 → **empate** (criterio declarado).
- CIC by-day +0.074 [+0.073,+0.075]; semillas 0.734±0.022 vs 0.639±0.027.
- EMBER −0.0017 (pérdida pequeña, reportada).
- LOAO: Wilcoxon por fold + Holm. UNSW empate (p=0.73 vs MLP; 0.16 vs XGB).
  CIC sin folds degenerados: XGB 5/5, Δ mediana −0.123, p=0.0625 → **pérdida**.

### V5 (resuelto) — E8 `ECE15`
- El apéndice ya define `hardGap` y prohíbe llamarlo ECE; se mantiene.

### V6 (declarado, no ejecutado) — ablación causal de typing
- El control “tres cabezas independientes con el mismo trunk” requiere
  reentrenamiento; queda como limitación y trabajo futuro. No se afirma
  causalidad del empaquetado tipado.

### V7 (resuelto) — SOTA/posicionamiento
- Añadida cita y discusión de Conformal Risk Control (ICLR 2024) y
  Proposición 5 (label shift es un vector, no un escalar).

## Fase 2 — verificación adversarial de las correcciones (GPT-5.6 Terra)

Informe: `reviews/phase2_terra.md`. Veredicto actualizado: **Borderline**
(antes Reject). Verificó contra los JSON: E9b bajo contrato E2 (0.3754→0.8789,
λ 0.7590/0.8789/0.9001, ruido 0.3541, false-att 0.5497→0.0925, absorción
0.0749→0.0286, F1 −0.0071, gate −0.0449/+0.0190), CIC (abst 0.0001→0.0000,
gate 0.7826→0.9689, recon 0.4259→0.9909, F1 +0.0392, botnet −0.2602),
bootstrap pareado (mismo `idx` para ambos modelos), CIs del texto, y
Wilcoxon/Holm recalculado (UNSW 0.7344/0.1641→Holm 0.4922; CIC 5/5, mediana
−0.12318, p=0.0625). Prop. 5 correcta.

Tres correcciones aplicadas:
1. `docs/EXPERIMENTS.md` §22 marcada como histórica inválida y sustituida por
   las cifras canónicas de §25 (se eliminaron las conclusiones 0.513→0.950 y
   0.00→0.94).
2. Protocolo estadístico reescrito sin claim de pre-registro: declara que los
   tests se añadieron tras la revisión externa, que los p-valores por fila son
   exploratorios (hasta 60 contrastes sin ajuste global) y que los veredictos
   descansan sólo en la regla por semillas y en Holm por fold.
3. Autores de CRC 2024 corregidos: Angelopoulos, Bates, **Fisch, Lei,
   Schuster** (verificado en ICLR/arXiv 2208.02814).

## Fase 3 — ablación causal de typing (a7)

Pregunta del revisor de Fase 1: ¿la ganancia viene del *typing* (gate/choice/
score como primitivas separadas) o del multi-task sharing? Ablación a7:

- **Diseño.** Un solo softmax de familia (misma CE ponderada por ω, mismo
  de-tilt analítico `z − log ω`, misma cabeza CORN de score, mismo trunk,
  mismo presupuesto y early stopping). El gate se deriva como
  `1 − p(normal)` de la distribución de choice; no hay cabeza de gate.
  Implementado en `src/models/network.py` (`HeadSpec.flat`, `_flat_forward`),
  `src/models/train.py` (`ablation == "a7"`), y registrado en el artefacto
  (`head_meta["flat"]`) para que `dump_outputs` reconstruya la red.
- **Corridas.** UNSW-Small y CIC-strat-Small, semillas 42/43/44, Modal L4.
  Artefactos/npz en el volumen `paper-results` (`outputs/*_abl_a7*`); los npz
  no se versionan (`.gitignore`), los metadatos y el informe sí.
- **Evaluador.** `scripts/a7_derived_gate_eval.py` (pareado por semilla y por
  fila, bootstrap B=300 descriptivo, métricas canónicas vía `eval_vx_run`).
  Salidas: `results/e6_a7/a7.json` y `a7.md`.
- **Documentación.** `docs/EXPERIMENTS.md` §26 (protocolo, tabla, alcance y
  comando de reproducción); `README.repro.md` con la sección de la ablación.
- **Repo público.** `public-sync` → `origin/main`: código a7 en inglés, los
  dos scripts (`a7_derived_gate_eval.py`, `a7_loao_eval.py`), resultados
  in-distribution y LOAO, paper y zip actualizados.

Resultados (semilla 42 en Tabla 6; medias±sd de 3 semillas en el texto):

| dataset | AUROC a7 vs typed | ΔAUROC | m-F1 a7 vs typed | Δm-F1 | ECE-ch a7 vs typed |
|---|---|---|---|---|---|
| UNSW | 0.9829±0.0014 vs 0.9837±0.0002 | −0.0008±0.0013 (1/3) | 0.4615±0.0121 vs 0.4560±0.0011 | +0.0055±0.0109 (2/3) | 0.0435 vs 0.0544 |
| CIC strat | 0.9999±0.0000 vs 0.9999±0.0000 | −0.0000 (2/3) | 0.8981±0.0348 vs 0.8312±0.0902 | +0.0669±0.0995 (2/3) | 0.0004 vs 0.0007 |

**Veredicto in-distribution:** nulo en detección. El gate dedicado no compra
AUROC ni FPR@95; la diferencia de atribución en CIC está dentro de la
inestabilidad de semillas ya documentada (typed 0.727–0.884). Lo que compra el
typing es el contrato: BCE binaria sin tilt, temperatura propia y semántica
`unknown` explícita. Esto acota el claim de novedad y responde la objeción
causal del revisor.

**Extensión LOAO (16 folds, seed 42; semillas 43/44 en tres folds CIC).**
`scripts/a7_loao_eval.py` + `results/e6_a7/loao.json`/`loao.md` y
`loao_seeds.json`/`loao_seeds.md`. UNSW: empate (mediana ΔAUROC −0.0009, 2/9,
Wilcoxon p=0.30). CIC: patrón no uniforme; el gate derivado pierde de forma
consistente en `dos_ddos` (0.60–0.63 en 3 semillas vs 0.85) y `botnet`
(0.57–0.59 vs 0.75), gana ligeramente en `recon`/`webattack`, y en
`exploitation` el seed 42 daba 0.42 vs 0.97 pero 43/44 dan 0.94/0.87: el
colapso era de una semilla. El test de 7 folds no tiene potencia (p=0.30).
Lectura: el typing importa en open-set solo donde la absorción benigna domina;
sin la regla de semillas habríamos reportado un −0.55 que no replica.

## Estado del paper

- 14 páginas, 0 errores LaTeX, 0 citas rotas, abstract actualizado.
- Todas las cifras del paper trazables a `results/` y `docs/EXPERIMENTS.md` §25.
- Paquete arXiv regenerado tras los cambios (5 archivos, 903 KB).
- `pytest -q`: **60 passed** (174 s).
- Self-tests: `scripts/e9b_oe_eval.py --selftest` reproduce E2 al cuarto
  decimal; `scripts/e1_paired_stats.py --selftest` reproduce la Tabla 1.

## Pendiente antes de someter

1. Commit/push del ciclo: `661b656` en `origin/master` (corrección E9b,
   inferencia pareada, revisión adversarial). La ablación a7 se commitea en
   el mismo ciclo tras regenerar PDF/zip.
2. Rellenar `\country{}` y `\acmConference{}` desde el portal ACM (no arXiv).
3. Decidir si el resultado CIC de E9b (detección, no abstención) cambia el
   título/claim de portada: el abstract ya lo dice con precisión.
