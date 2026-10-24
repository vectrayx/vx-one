# A7 under LOAO - derived gate vs dedicated gate

One-vs-normal detection on the excluded family (same folds, seed 42).
Delta = A7 - typed; positive is better. Bootstrap intervals are
descriptive (B=300); the fold-level sign test is the verdict.

| fold | AUROC a7 | AUROC typed | delta | 95% CI |
|---|---|---|---|---|
| cic/botnet | 0.5874 | 0.7499 | -0.1624 | [-0.1751,-0.1494] |
| cic/bruteforce | 0.9422 | 0.9674 | -0.0252 | [-0.0265,-0.0238] |
| cic/dos_ddos | 0.6326 | 0.8524 | -0.2198 | [-0.2221,-0.2172] |
| cic/exploitation | 0.4211 | 0.9703 | -0.5491 | [-0.5499,-0.5485] |
| cic/infiltration | 0.6762 | 0.5329 | +0.1523 | [-0.1839,+0.5057] |
| cic/recon | 0.4391 | 0.4259 | +0.0131 | [+0.0101,+0.0159] |
| cic/webattack | 0.9926 | 0.9792 | +0.0131 | [+0.0078,+0.0184] |
| unsw/analysis | 0.9989 | 0.9992 | -0.0003 | [-0.0006,-0.0001] |
| unsw/backdoor | 0.9975 | 0.9989 | -0.0014 | [-0.0023,-0.0006] |
| unsw/dos | 0.9936 | 0.9941 | -0.0005 | [-0.0011,+0.0001] |
| unsw/exploitation | 0.9492 | 0.9550 | -0.0059 | [-0.0069,-0.0050] |
| unsw/fuzzing | 0.8308 | 0.8519 | -0.0210 | [-0.0225,-0.0197] |
| unsw/generic | 0.9882 | 0.9957 | -0.0075 | [-0.0081,-0.0069] |
| unsw/recon | 0.9721 | 0.9523 | +0.0198 | [+0.0183,+0.0212] |
| unsw/shellcode | 0.9679 | 0.9688 | -0.0011 | [-0.0057,+0.0044] |
| unsw/worms | 0.9901 | 0.9871 | +0.0030 | [-0.0006,+0.0073] |

## Summary

### all

- folds: 16
- median delta: -0.0012
- mean delta: -0.0501 ± 0.1546
- A7 better on 5/16 folds
- exact Wilcoxon signed-rank p = 0.2522

### cic

- folds: 7
- median delta: -0.0252
- mean delta: -0.1124 ± 0.2270
- A7 better on 3/7 folds
- exact Wilcoxon signed-rank p = 0.2969

### unsw

- folds: 9
- median delta: -0.0009
- mean delta: -0.0016 ± 0.0107
- A7 better on 2/9 folds
- exact Wilcoxon signed-rank p = 0.3008

