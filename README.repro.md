# VX-One — Typed, Calibrated Decision Heads for Real-Time Perimeter Triage

Reproducible code and ACM paper for VX-One.

## Reproduce the paper
```bash
# Paper
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper compiles with `acmart` sigconf. Figures and tables are in `paper/`.

## Experiments
Code in `src/` and `scripts/`. Tests require public datasets:
- UNSW-NB15
- CIC-IDS2017
- EMBER2024

Link the data directory from the original repo:
```bash
ln -s ../paper/data data
pytest tests/
```

Key scripts:
- `scripts/e1_seed.py`, `scripts/e8_window_recal.py`, `scripts/e9_window_retrain.py`
- `scripts/e1_paired_stats.py` (row-paired bootstrap + seed rule; `--selftest`
  reproduces Table 1)
- `scripts/e9b_oe_eval.py` (outlier-exposure evaluator under the
  `vx_calibrated` contract; `--selftest` reproduces E2)
- `scripts/a7_derived_gate_eval.py` (typing ablation: dedicated gate vs gate
  derived from a single family softmax; `--selftest` is synthetic)
- `scripts/a7_e9b_eval.py` (same ablation under outlier exposure; pairs the
  derived gate with the dedicated E9b run fold by fold)

## Typing ablation (a7)

The a7 ablation asks whether the dedicated gate head matters for accuracy, or
whether the typed triad is a semantic contract. It retrains UNSW-Small and
CIC-strat-Small with a single family softmax and derives the gate as
`1 - p(normal)`; trunk, weighted CE, analytic de-tilt, CORN score head and
budget are identical.

```bash
for src in unsw cic; do
  sch=oficial; [ "$src" = cic ] && sch=strat
  for s in 42 43 44; do
    modal run modal/app.py::train_vx --source $src --scale small \
      --scheme $sch --ablation a7 --seed $s
  done
done
```

Then pull the logits from the Modal volume and evaluate:

```bash
python scripts/a7_derived_gate_eval.py --a7 results/e6_a7 \
  --typed results/outputs_seed --out results/e6_a7
```

Result: in-distribution detection is a tie (UNSW ΔAUROC -0.0008±0.0013 across
seeds; CIC identical to the fourth decimal), so the dedicated gate buys a
contract (separate binary calibration, explicit `unknown`), not ranking. Under
LOAO the null splits by regime: UNSW stays a tie, while saturated CIC loses
consistently on `dos_ddos` and `botnet` (the folds where the choice head
absorbs the unseen family into `normal`); `exploitation` is seed-unstable.
The LOAO extension is reproducible with:

```bash
modal run modal/app.py::loao_a7 --source unsw
modal run modal/app.py::loao_a7 --source cic
python scripts/a7_loao_eval.py --a7 results/e6_a7/loao \
  --typed results/e6_a7/loao_typed --out results/e6_a7
```

Under outlier exposure (E9b protocol: the excluded family returns as an
unlabeled pool, lambda=0.5), the derived gate matches the dedicated one on
abstention (0.909 vs 0.879, p=0.15) and false attribution, but is
significantly worse at detection ranking (gate AUROC 0.922 vs 0.965,
p=0.012). Abstention lives in the choice head; the dedicated gate buys
detection ranking. Reproduce with:

```bash
modal run modal/app.py::oe_loao_a7 --source unsw --oe-weight 0.5
python scripts/a7_e9b_eval.py --a7 results/e6_a7/oe \
  --typed results/outputs_oe \
  --typed-meta results/e9b_oe/meta/oe_json --out results/e6_a7
```

See `results/e6_a7/a7.md`, `loao.md`, `loao_seeds.md`, `oe_a7.md` and
`docs/EXPERIMENTS.md` §26--§27.

Results aggregated in `results/`. No raw parquets are shipped.

## License
CC-BY-4.0 for paper, Apache-2.0 for code.
