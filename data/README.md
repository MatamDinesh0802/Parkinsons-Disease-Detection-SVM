# Data

## UCI Parkinson's Dataset

The CSV at `raw/parkinsons.csv` is the UCI Machine Learning Repository's
[Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons),
collected by Max Little (University of Oxford) in collaboration with the
National Centre for Voice and Speech, Denver, CO.

### Summary

| | |
|---|---|
| Rows | 195 (31 subjects × ~6 voice recordings) |
| Features | 22 voice measures + `status` (target) + `name` (id) |
| Target | `status` — 1 = Parkinson's, 0 = healthy |
| Class balance | ~75% Parkinson's, ~25% healthy |

### Feature glossary

| Group | Features | Meaning |
|---|---|---|
| Fundamental frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` | Avg / max / min vocal fundamental frequency |
| Jitter | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` | Variation in fundamental frequency |
| Shimmer | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` | Variation in amplitude |
| Noise/Harmonics | `NHR`, `HNR` | Noise-to-harmonics, harmonics-to-noise |
| Nonlinear dynamics | `RPDE`, `D2`, `DFA` | Recurrence period density entropy, correlation dim., detrended fluctuation |
| Pitch nonlinearity | `spread1`, `spread2`, `PPE` | Fundamental frequency variation measures |

### Citation

> Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A., & Moroz, I. M. (2007).
> Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.
> *BioMedical Engineering OnLine*, 6(1), 23.
