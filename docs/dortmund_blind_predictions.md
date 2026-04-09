# Blind Predictions for Dortmund Vital Study Replication
## Written BEFORE downloading or analyzing the data
## Date: 2026-03-21

### Dataset
- Dortmund Vital Study (ds005385, OpenNeuro)
- 608 subjects, ages 20-70, continuous age distribution
- 64-channel EEG, resting state (3 min EC + 3 min EO)
- Different lab (IfADo Dortmund) from LEMON (MPI Leipzig)

### Source Findings (LEMON dataset)
- N = 215 subjects (153 young 20-35, 74 old 59-77)
- Broadband PTA computed per channel per band
- Key finding: beta prop_positive ~ age: r = -0.33 (bimodal groups)

### BLIND PREDICTIONS (recorded before any analysis)

#### Prediction 1: PTA age correlation (PRIMARY)
- **beta prop_positive ~ age: r in range [-0.40, -0.20]**
- Point estimate: r = -0.30
- Falsification criterion: |r| < 0.15 → effect not replicated

#### Prediction 2: σ(A) age correlation
- **σ(A)_beta ~ age: r > 0 (positive)**
- Point estimate: r = +0.20
- Falsification criterion: r < 0 → coarsening not replicated

#### Prediction 3: Band specificity
- **Beta band shows strongest age effect (largest |r|) among all bands**
- Secondary: alpha also shows significant r, but weaker
- Falsification criterion: if delta or theta show stronger age effect → band specificity not replicated

#### Prediction 4: Prop_positive > 0.5 at all ages
- **Mean prop_positive > 0.5 for all age decades**
- Baseline asymmetry (A > 0) is biological, not aging
- Falsification criterion: any decade with prop_positive < 0.45

#### Prediction 5: Coarsening triplet
- **σ(A) increases AND H(sign pattern) decreases with age**
- This is the EEG coarsening signature
- Falsification criterion: if σ does not increase OR H does not decrease

#### Prediction 6 (BONUS, from longitudinal follow-up):
- **Within-subject change over 5 years: Δ(beta prop_positive) < 0**
- Subjects measured at t1 and t1+5yr should show decrease
- This is the strongest possible test (within-subject, longitudinal)
- Falsification criterion: Δ(beta prop_positive) not significantly < 0

### Analysis Protocol (identical to LEMON)
1. Preprocessing: bandpass 0.5-100 Hz, notch 50 Hz, average reference, ICA
2. Broadband PTA: 1-45 Hz bandpass, cycle detection, classify by frequency
3. Power thresholding: per-band median relative power filter
4. Minimum cycles: 30 per channel per band
5. Module score: prop_positive per band per subject
6. Spatial metrics: σ(A), H(sign pattern)
7. Age correlation: Pearson/Spearman, controlled for sex

### Criteria for success
- PRIMARY: beta prop_positive ~ age: |r| > 0.15, p < 0.01
- SECONDARY: σ(A) ~ age: r > 0, p < 0.05
- TERTIARY: coarsening triplet confirmed (σ↑, H↓)
- BONUS: longitudinal within-subject confirmation

### If ALL primary+secondary fail:
- Framework is not replicated
- LEMON finding was dataset-specific
- Paper must either be fundamentally revised or shelved

### If primary succeeds but secondary fails:
- Core PTA-age association replicated
- Coarsening interpretation not supported
- Paper focuses on PTA as biomarker, drops coarsening narrative

### If all succeed:
- Two-dataset, two-lab replication
- Proceed to quantitative model (coupled oscillators)
- Paper gains major strength
