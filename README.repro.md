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

Results aggregated in `results/`. No raw parquets are shipped.

## License
CC-BY-4.0 for paper, Apache-2.0 for code.
