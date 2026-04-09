# Mechanism Revision: From Coarsening to E/I Drift
## Date: 2026-03-21

## What was killed

### Ising Coarsening Model ✗
- Prediction: coupling J decreases with age → spatial domains form
- Reality: coupling INCREASES with age (0.031 → 0.076)
- Spatial structure flattens (distance-correlation r: -0.383 → -0.232)
- Opposite of coarsening: dedifferentiation, not domain formation

### Coupled Oscillator / Langevin Model ✗
- Fitted J = 0 to young data (coupling not needed for summary stats)
- When pairwise structure included: coupling exists but grows with age
- Cannot reproduce young→old transition by reducing J alone

### "Noise increase" as primary driver ✗
- σ(A) increase accounts for only 3% of Δ(prop_positive)
- 68% comes from mean shift (bias change)
- 30% from non-Gaussian effects (heavy left tail)

## What survives

### Empirical PTA-age association ✓
- LEMON (N=215): r = -0.330
- Dortmund (N=606): r = -0.316, p = 1.5e-15
- Blind prediction matched: r ∈ [-0.40, -0.20]
- Band specificity: beta strongest (confirmed)
- σ(A) ~ age: r = +0.091, p = 0.026 (confirmed)
- Test-retest: r = 0.451 (reliable trait)

### TMS coarsening signatures (scRNA-seq) ✓
- σ(Module Score) increases with age (p = 6.4e-4, NF-κB specific)
- H(pattern) decreases with age (p = 1.6e-3, NF-κB specific)
- These are DESCRIPTIVE findings, not mechanistically linked to EEG

## New hypothesis: Regional E/I Drift

### Core observation
Beta PTA aging is driven by a systematic shift in waveform asymmetry
toward more inhibitory (negative A) values. This shift is:
- **Region-specific**: Central (-0.008) and Parietal (-0.008) strongest
- **Not global**: Temporal shows OPPOSITE direction (+0.006)
- **Not noise**: Mean shift accounts for 68% of effect

### Mechanism (hypothesis, not proven)
1. GABA concentration declines with age (MRS literature: frontal > parietal)
2. Remaining GABAergic synapses compensate via receptor upregulation
3. This creates more synchronous but less diverse inhibitory transients
4. Beta waveform becomes more stereotypically inhibitory (negative PTA)
5. Central/parietal affected most (motor cortex, strongest GABA decline)
6. Temporal spared or shows opposite pattern (different GABA dynamics)

### Testable predictions
- P1: Regional PTA shift should correlate with regional GABA decline (MRS)
- P2: Benzodiazepine (GABA-A agonist) should acutely shift PTA negative
- P3: Cortical thickness in central/parietal should correlate with PTA
- P4: Motor cortex PTA should predict motor performance (grip strength, RT)

### What's needed
- EEG+MRS dataset with aging (not currently available as open data)
- LEMON MRI data for cortical thickness correlation
- Pharmacological EEG study (benzodiazepine challenge)

## Status of each paper

### EEG Paper
- Core finding (PTA ~ age): REPLICATED, strong
- Mechanism section: must be rewritten from "coarsening" to "E/I drift"
- Quantitative model: killed; replace with descriptive regional analysis
- Strength: two-dataset replication, blind predictions matched

### Transcriptomic Paper (coupling atlas)
- π_tissue near-invariant: robust (7/7 Tier 1 tests)
- Coarsening signatures in scRNA-seq: confirmed but descriptive
- Cross-system analogy (EEG↔scRNA-seq): weakened
  - Both show σ↑ and H↓ with aging, but mechanisms different
  - EEG: E/I drift, not coupling decay
  - scRNA-seq: genuine NF-κB pathway coarsening
  - Analogy is phenomenological, not mechanistic

## Lessons learned
1. Replication before mechanism: always replicate first
2. Quantitative models expose qualitative hand-waving
3. "σ↑ and H↓" is too generic — any degrading system shows this
4. Spatial structure analysis is essential: summary stats hide mechanism
5. Kill your darlings: Ising was beautiful but wrong
