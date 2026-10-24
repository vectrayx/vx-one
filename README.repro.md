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

Result: detection is a tie (UNSW ΔAUROC -0.0008±0.0013 across seeds; CIC
identical to the fourth decimal), so the dedicated gate buys a contract
(separate binary calibration, explicit `unknown`), not ranking. See
`results/e6_a7/a7.md` and `docs/EXPERIMENTS.md` §26.

Results aggregated in `results/`. No raw parquets are shipped.

## License
CC-BY-4.0 for paper, Apache-2.0 for code.
