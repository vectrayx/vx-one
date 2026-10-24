# E9b - outlier exposure under the paper's calibration contract

Estimator: `vx_calibrated` (temperature + omega de-tilt), the same
contract as E2. Reference and OE rows therefore share one estimator.
Supersedes the earlier temperature-only readout.

## UNSW

| fold | abst ref | abst oe | false-att ref | false-att oe | absorb ref | absorb oe | F1 ref | F1 oe | gate ref | gate oe |
|---|---|---|---|---|---|---|---|---|---|---|
| analysis | 0.923 | 0.917 | 0.075 | 0.081 | 0.002 | 0.002 | 0.508 | 0.506 | 0.999 | 0.999 |
| backdoor | 0.635 | 0.981 | 0.365 | 0.019 | 0.000 | 0.000 | 0.497 | 0.499 | 0.999 | 0.998 |
| dos | 0.028 | 0.610 | 0.967 | 0.384 | 0.005 | 0.006 | 0.508 | 0.502 | 0.994 | 0.994 |
| exploitation | 0.082 | 0.883 | 0.830 | 0.097 | 0.088 | 0.019 | 0.558 | 0.506 | 0.955 | 0.974 |
| fuzzing | 0.284 | 0.683 | 0.184 | 0.093 | 0.531 | 0.223 | 0.451 | 0.490 | 0.852 | 0.865 |
| generic | 0.500 | 0.974 | 0.500 | 0.026 | 0.001 | 0.000 | 0.380 | 0.383 | 0.996 | 0.951 |
| recon | 0.282 | 0.954 | 0.687 | 0.045 | 0.031 | 0.001 | 0.414 | 0.406 | 0.952 | 0.952 |
| shellcode | 0.418 | 0.976 | 0.566 | 0.018 | 0.016 | 0.005 | 0.462 | 0.462 | 0.969 | 0.965 |
| worms | 0.227 | 0.932 | 0.773 | 0.068 | 0.000 | 0.000 | 0.506 | 0.468 | 0.987 | 0.990 |

Means: ref abst 0.375, F1 0.476, gate 0.967; oe0.2 abst 0.759, F1 0.465, gate 0.966; oe0.5 abst 0.879, F1 0.469, gate 0.965; oe1.0 abst 0.900, F1 0.456, gate 0.964; noise0.5 abst 0.354, F1 0.469, gate 0.968

## CIC

| fold | abst ref | abst oe | false-att ref | false-att oe | absorb ref | absorb oe | F1 ref | F1 oe | gate ref | gate oe |
|---|---|---|---|---|---|---|---|---|---|---|
| botnet | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.960 | 0.700 | 0.750 | 0.988 |
| bruteforce | 0.001 | 0.000 | 0.291 | 0.000 | 0.708 | 1.000 | 0.637 | 0.899 | 0.967 | 0.996 |
| dos_ddos | 0.000 | 0.000 | 0.027 | 0.000 | 0.973 | 1.000 | 0.606 | 0.730 | 0.852 | 0.999 |
| exploitation | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.777 | 0.889 | 0.970 | 0.998 |
| infiltration | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.941 | 0.778 | 0.533 | 0.812 |
| recon | 0.000 | 0.000 | 0.002 | 0.001 | 0.998 | 0.999 | 0.865 | 0.824 | 0.426 | 0.991 |
| webattack | 0.000 | 0.000 | 0.067 | 0.000 | 0.933 | 1.000 | 0.641 | 0.881 | 0.979 | 0.999 |

Means: ref abst 0.000, F1 0.775, gate 0.783; oe0.5 abst 0.000, F1 0.815, gate 0.969
