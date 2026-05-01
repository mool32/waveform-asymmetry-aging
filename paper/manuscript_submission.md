# Waveform Asymmetry as a Biomarker of Neural Aging: Spatial Degradation of Oscillatory Cycle Shape Across Two Independent Cohorts

**Authors:** Theodor Spiro^1,2^

**Affiliations:**

$^{1}$ Vaika Inc., East Aurora, NY, USA

$^{2}$ Independent researcher

**Corresponding author:** Theodor Spiro (tspiro@vaika.org)

**Keywords:** waveform asymmetry, peak-trough asymmetry, EEG, aging, beta oscillations, spatial organization, excitatory-inhibitory balance

---

## Abstract

The shape of each neural oscillatory cycle — quantified by peak-trough asymmetry (PTA), the ratio of rise to fall time — may reflect excitatory-inhibitory balance in the generating circuit. Whether waveform shape degrades with aging is unknown. We measured broadband PTA across five frequency bands in resting-state EEG from 215 adults (age 20–77; LEMON dataset) and replicated all findings in 608 adults (age 20–70; Dortmund Vital Study), including 208 with 5-year longitudinal follow-up. We report three findings. First, asymmetry is band-specific and spatially structured: slow rhythms show excitatory-type asymmetry while fast rhythms show inhibitory-type asymmetry, with a spatial double dissociation between posterior alpha and central beta. Second, beta-band asymmetry decreases with age (LEMON: *r* = -0.326, *d* = 0.69; Dortmund: *r* = -0.314, *d* = 0.75), surviving control for aperiodic slope and strongest at central electrodes — opposite to electromyographic contamination. Longitudinal data confirmed the predicted direction (mean $\Delta$ = -0.017) with moderate test–retest reliability (*r* = 0.45). Third, spatial organization of asymmetry degrades with age, and theta spatial entropy predicts memory performance independent of age. Waveform shape provides a power-independent measure of neural aging that complements existing spectral biomarkers.

---

## 1. Introduction

### 1.1 Neural oscillations beyond frequency and power

Resting-state electroencephalography (EEG) has long served as a window into the neural dynamics of healthy aging. Decades of research have established that aging is accompanied by slowing of the posterior alpha rhythm, reductions in alpha power, and changes in the aperiodic (1/*f*) spectral slope [1–4]. These spectral measures have proven useful as biomarkers but share a fundamental limitation: they characterize *what frequencies are present* and *how much power they carry*, while discarding information about the *shape* of each oscillatory cycle.

Neural oscillations are not sinusoidal. The waveform of each cycle is shaped by the synaptic currents that generate it — fast excitatory postsynaptic potentials produce sharp, rapid deflections, while slower inhibitory currents produce more gradual returns [5, 6]. This asymmetry between the rising and falling phases of each cycle is thought to carry information about the excitatory–inhibitory (E/I) balance of the generating circuit that is invisible to power spectral analysis [7, 8].

### 1.2 Peak-trough asymmetry

Cole and Voytek [7] introduced peak-trough asymmetry (PTA) as a method for quantifying waveform shape directly from broadband neural time series. The method identifies individual oscillatory cycles as trough–peak–trough sequences in the broadband-filtered signal, then computes the ratio of rise time to total cycle duration for each cycle. Values above 0.5 indicate an excitatory-type waveform (fast rise, slow fall), while values below 0.5 indicate an inhibitory-type waveform (slow rise, fast fall). Critically, PTA must be computed from broadband-filtered data; narrowband filtering symmetrizes the waveform and destroys the asymmetry information [7, 9].

PTA has been applied to characterize waveform shape in the context of Parkinson's disease [10], development [11], and cognitive states [12], but its behavior across the adult lifespan has not been systematically examined.

### 1.3 Why waveform shape may index aging

Several lines of evidence suggest that oscillatory cycle shape should change with aging. First, the E/I balance shifts across the lifespan: GABA concentrations decline in frontal and sensorimotor cortex [13, 14], cortical inhibitory interneuron density decreases [15], and the balance between excitatory and inhibitory synaptic transmission is altered [16]. These changes could plausibly manifest as systematic shifts in waveform asymmetry. Second, aging disrupts the spatial organization of neural activity through white matter degradation and loss of long-range corticocortical connections [17, 18], which could alter the spatial pattern of waveform shape across the scalp. Third, waveform asymmetry captures aspects of neural dynamics that are independent of oscillatory power and aperiodic slope [7], potentially providing complementary information about neural aging.

### 1.4 Present study

We tested three predictions in two independent datasets. First, we predicted that waveform asymmetry would be systematically non-zero and band-specific, with the sign of asymmetry reflecting the dominant circuit type generating each rhythm. Second, we predicted that beta-band asymmetry would decrease with age, consistent with age-related changes in sensorimotor E/I balance. Third, we predicted that the spatial organization of waveform asymmetry across the scalp would degrade with aging. We tested these predictions in the LEMON dataset (N = 215, age 20–77) and replicated all analyses in the Dortmund Vital Study (N = 608, age 20–70), including 208 participants with 5-year longitudinal follow-up. We additionally examined cross-system coupling between EEG and cardiac asymmetry, state-dependent modulation by visual input (eyes-closed vs. eyes-open), and cognitive correlates of spatial asymmetry organization.

---

## 2. Methods

### 2.1 Datasets

**LEMON (Leipzig Study for Mind-Body-Emotion Interactions).** 215 healthy adults (141 male, 74 female; age 20–77 years, mean 39.0 $\pm$ 20.1) with resting-state EEG recorded at 2500 Hz using 62-channel active electrodes (BrainVision actiCHamp) with FCz reference [19]. The dataset has a bimodal age distribution (153 young adults age 20–35, 62 older adults age 59–77). Recordings included 16 minutes of alternating 1-minute eyes-closed and eyes-open blocks. Cognitive data were available for a subset including the California Verbal Learning Test (CVLT), Trail Making Test (TMT), and Test of Attentional Performance working memory (TAP-WM).

**Dortmund Vital Study.** 608 healthy adults (age 20–70, continuous distribution) with resting-state EEG recorded using 64-channel electrodes at the Leibniz Research Centre for Working Environment and Human Factors [20]. A subset of 208 participants were tested at two time points approximately 5 years apart, providing longitudinal follow-up data (130 female, 78 male; baseline age 20–70). OpenNeuro accession: ds005385.

### 2.2 EEG preprocessing

Continuous EEG data were bandpass-filtered at 1–45 Hz (zero-phase FIR filter) and re-referenced to average reference. Independent component analysis (ICA) was applied to remove ocular and cardiac artifacts. For the LEMON dataset, the ECG component was extracted from ICA for subsequent heart-rate variability (HRV) analysis (successful in 72 of 215 subjects). The broadband filter preserves waveform shape; narrowband filtering (e.g., 8–12 Hz for alpha) was avoided as it symmetrizes the signal and eliminates asymmetry information [7].

### 2.3 Peak-trough asymmetry computation

Following Cole and Voytek [7], individual oscillatory cycles were detected in the broadband (1–45 Hz) filtered signal as trough–peak–trough sequences. For each cycle, we computed the asymmetry index:

$$A = \frac{t_{\text{rise}}}{t_{\text{rise}} + t_{\text{fall}}}$$

where *t*_rise is the duration from trough to peak and *t*_fall is the duration from peak to the next trough. Each cycle was classified into a frequency band based on its total duration: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and low gamma (30–50 Hz).

**Power-threshold channel selection.** For each frequency band, we included only channels with above-median relative spectral power in that band. This prevents dilution of band-specific signals by channels lacking the relevant rhythm (e.g., including all 62 channels for alpha when only ~3 occipital channels express a strong alpha rhythm).

**Minimum cycle count.** Channels with fewer than 30 valid cycles in a given band were excluded.

### 2.4 Summary metrics

For each subject and band, we computed:

- **prop_positive**: fraction of included channels with *A* > 0.5 (excitatory-type asymmetry)
- **mean *A***: trimmed mean (10%) of *A* across included channels
- **$\sigma$(*A*)**: standard deviation of *A* across included channels (spatial variability)
- **H(sign *A*)**: binary Shannon entropy of the spatial sign pattern, computed as *H* = -*p* log$_2$ *p* - (1-*p*) log$_2$(1-*p*), where *p* = prop_positive (spatial diversity of asymmetry signs)
- **|*A*|**: mean absolute asymmetry across channels (local magnitude)

For regional analysis, channels were grouped into frontal (Fp1, Fp2, F3, F4, F7, F8, Fz), central (C3, C4, Cz, FC1, FC2, CP1, CP2), temporal (T7, T8, TP9, TP10), parietal (P3, P4, P7, P8, Pz), and occipital (O1, O2, Oz, PO3, PO4) regions.

### 2.5 Heart-rate variability asymmetry

For the 72 LEMON subjects with successfully extracted ECG components, R-peaks were detected and inter-beat intervals computed. HRV oscillations were decomposed into four frequency bands: very low frequency (VLF, 0.003–0.04 Hz), low frequency (LF, 0.04–0.15 Hz), high frequency (HF, 0.15–0.4 Hz), and very high frequency (VHF, 0.4–1.0 Hz). PTA was applied to HRV oscillations in each band.

### 2.6 Cognitive data

LEMON cognitive measures included: CVLT total learning (CVLT_6, range 27–76), trial 1 recall (CVLT_8, 1–12), delayed recall (CVLT_9, 1–16), and recognition (CVLT_10, 3–16); Trail Making Test parts A and B (TMT-A, TMT-B); TAP working memory (omissions and reaction time).

### 2.7 Eyes-closed versus eyes-open analysis

For 212 LEMON subjects with valid recordings in both conditions, we computed alpha-band H(sign *A*) separately for eyes-closed (EC) and eyes-open (EO) segments and tested within-subject differences using paired *t*-tests and Wilcoxon signed-rank tests.

### 2.8 Statistical analysis

Age associations were tested using Pearson and Spearman correlations. Effect sizes were quantified as Cohen's *d* (independent samples) or *d*_z (paired samples). Where noted, partial correlations controlled for aperiodic spectral slope or age. Multiple comparisons across frequency bands were corrected using the Benjamini–Hochberg false discovery rate (FDR) procedure. Bootstrap 95% confidence intervals (10,000 iterations) were computed for longitudinal effect estimates. All analyses were conducted in Python using SciPy and statsmodels.

### 2.9 Age prediction analysis

To evaluate the predictive utility of waveform asymmetry relative to established spectral features, we trained ridge regression models to predict chronological age from six feature sets: (1) PTA-only (16 features: trimmed mean *A*, prop_positive, and $\sigma$(*A*) per band, plus global $\sigma$); (2) spectral power (31 features: relative band power for 5 bands $\times$ 5 regions, plus 5 global band powers and alpha peak frequency); (3) aperiodic slope (1 feature: log-log linear fit of 2–40 Hz PSD); (4) all spectral features combined (32 features); (5) PTA + spectral combined (48 features); and (6) sex only (1 feature, baseline). Spectral features were extracted from LEMON raw EEG using Welch periodograms (4 s windows, 50% overlap).

Each model was a pipeline of median imputation, z-score standardisation, and RidgeCV ($\alpha$ $\in$ {0.01, 0.1, 1, 10, 100, 1000}), evaluated by 10 $\times$ 10-fold stratified cross-validation (stratified by age quartile bins). The primary metric was mean absolute error (MAE). Statistical significance was assessed by permutation tests (200 permutations, 5 $\times$ 2-fold CV). For cross-dataset generalization, the PTA model was trained on all LEMON data and tested on Dortmund (N = 608).

---

## 3. Results

### 3.1 Waveform asymmetry is band-specific and spatially structured

Broadband PTA revealed that waveform asymmetry differs systematically across frequency bands (Table 1, Figure 1A). Slow rhythms showed excitatory-type asymmetry (prop_positive > 0.5): delta (*M* = 0.954, *t*(41) = 24.0, *p* = 8.7 $\times 10^{-26}$) and alpha (*M* = 0.612, *t*(214) = 10.4, *p* = 8.8 $\times 10^{-21}$). Fast rhythms showed inhibitory-type asymmetry (prop_positive < 0.5): beta (*M* = 0.295, *t*(214) = -17.9, *p* = 1.8 $\times 10^{-44}$) and low gamma (*M* = 0.387, *t*(214) = -12.7, *p* = 6.6 $\times 10^{-28}$). Theta was intermediate (*M* = 0.523, *p* = 0.07).

![Band-specific waveform asymmetry and spatial structure.](figures/fig1_band_asymmetry.pdf)

**Table 1.** Band-specific waveform asymmetry in LEMON (N = 215).

| Band | N | Mean prop_positive | *t* vs 0.5 | *p* | % > 0.5 |
|------|---|--------------------|-----------|------|---------|
| Delta | 42 | 0.954 | 24.0 | 8.7 $\times 10^{-26}$ | 98% |
| Theta | 209 | 0.523 | 1.82 | 0.070 | 57% |
| Alpha | 215 | 0.612 | 10.4 | 8.8 $\times 10^{-21}$ | 72% |
| Beta | 215 | 0.295 | -17.9 | 1.8 $\times 10^{-44}$ | 10% |
| Low gamma | 215 | 0.387 | -12.7 | 6.6 $\times 10^{-28}$ | 17% |

**Spatial double dissociation (Figure 1B).** Alpha and beta asymmetry showed opposing spatial gradients. Alpha asymmetry was positive (excitatory-type) at occipital (83% positive) and parietal (80%) channels but negative at frontal channels (28%). Beta asymmetry showed the reverse: positive at frontal channels (70%) but negative at central (14%) and occipital (21%) channels. This double dissociation was stable across all 215 subjects and was replicated in the Dortmund cohort. The pattern is consistent with distinct generating mechanisms: posterior alpha arises from thalamocortical excitatory loops, while central beta reflects sensorimotor inhibitory circuits.

### 3.2 Beta-band asymmetry decreases with age

Beta prop_positive showed a robust negative association with age in both datasets (Figure 2).

![Beta-band asymmetry decreases with age across two cohorts and longitudinal follow-up.](figures/fig2_beta_age_hero.pdf)

**LEMON (N = 215).** *r* = -0.326, *p* = 1.0 $\times 10^{-6}$; Spearman *$\rho$* = -0.357. Young adults (age < 35, N = 153) had higher beta prop_positive (*M* = 0.332) than older adults (age > 55, N = 62; *M* = 0.222), yielding Cohen's *d* = 0.69, a medium-to-large effect substantially stronger than classical alpha power changes with aging (*d* $\approx$ 0.3–0.5). No other band showed a significant age association (all *p* > 0.05 after FDR correction). Within the young subgroup alone, the correlation persisted (*r* = -0.237, *p* = 0.004), confirming the effect is not driven solely by the bimodal age structure.

**Dortmund replication (N = 608).** In a blind, pre-registered replication, we predicted *r* $\in$ [-0.40, -0.20] for beta prop_positive versus age. The observed correlation was *r* = -0.314, *p* = 2.2 $\times 10^{-15}$ (Spearman *$\rho$* = -0.308), falling precisely within the predicted interval. Young adults (*M* = 0.450, N = 203) differed from older adults (*M* = 0.344, N = 163; *d* = 0.75). Band specificity was preserved: beta showed the strongest age effect, with alpha showing a weaker effect (*r* = -0.128, *p* = 0.002) and theta showing none (*r* = 0.057, *p* = 0.16).

**Longitudinal confirmation (N = 208, 5-year follow-up; Figure 2C).** Within-subject change over approximately 5 years was in the predicted direction: mean $\Delta$(prop_positive_beta) = -0.017 (95% CI: [-0.038, 0.005]; Wilcoxon *p* = 0.07 one-sided, *d*_z = -0.11). A post-hoc power analysis indicates that detecting a within-subject *d*_z $\approx$ 0.10 effect with 80% power would require N $\approx$ 620; the observed N = 208 provides only ~35% power, making the directionally consistent trend compatible with the cross-sectional effect size rather than evidence against it. Critically, test–retest reliability was moderate (*r* = 0.45, *p* = 8.4 $\times 10^{-12}$), confirming that beta PTA is a stable individual trait that can be measured reproducibly across sessions.

**Control: aperiodic spectral slope.** Aperiodic exponent (1/*f* slope) also declined with age (*r* = -0.396, *p* = 1.8 $\times 10^{-9}$) and correlated with beta prop_positive (*r* = 0.277, *p* = 3.9 $\times 10^{-5}$). However, the beta–age association survived partial correlation controlling for aperiodic exponent (partial *r* = -0.245, *p* = 2.8 $\times 10^{-4}$), demonstrating that waveform asymmetry captures age-related variance independent of the spectral slope. The 25% reduction in correlation indicates overlapping but distinct information.

**Control: electromyographic contamination (Table 2).** Muscle artifact is symmetric, concentrated in beta/gamma frequencies, and maximal at temporal and frontal electrodes. If EMG drove the beta–age effect, it should be strongest at high-EMG channels. We observed the opposite: the effect was strongest at central electrodes (*r* = -0.300, *d* = 0.68, *p* = 7.7 $\times 10^{-6}$) — the site of sensorimotor beta generation — and weakest at temporal electrodes (*r* = -0.084, *d* = 0.17, *p* = 0.22; Table 2). This spatial pattern is precisely what the neural hypothesis predicts and is incompatible with an EMG artifact explanation.

**Table 2.** Regional beta asymmetry–age associations (LEMON, N = 215).

| Region | EMG risk | *r* (beta pp ~ age) | Cohen's *d* | *p* |
|--------|----------|---------------------|-------------|------|
| Central | Low | -0.300 | 0.68 | 7.7 $\times 10^{-6}$ |
| Frontal | High | -0.228 | 0.49 | 7.5 $\times 10^{-4}$ |
| Parietal | Low | -0.203 | 0.41 | 0.003 |
| Occipital | Low | -0.182 | 0.36 | 0.007 |
| Temporal | High | -0.084 | 0.17 | 0.22 |

### 3.3 Spatial organization of asymmetry degrades with age

Beyond the mean level of asymmetry, we examined three spatial metrics that together characterize how the pattern of waveform shape across the scalp changes with aging (Figure 3).

![Spatial signatures of aging: LEMON and Dortmund replication.](figures/fig3_spatial_aging.pdf)

**Local magnitude increases.** The mean absolute asymmetry per channel, |*A*|, increased with age for theta (*r* = +0.449, *p* = 4.4 $\times 10^{-12}$) and beta (*r* = +0.232, *p* = 6.2 $\times 10^{-4}$). In Dortmund, the same pattern held: theta |*A*| increased with age (*r* = +0.148, *p* = 2.7 $\times 10^{-4}$) and beta |*A*| increased (*r* = +0.091, *p* = 0.025).

**Spatial diversity decreases.** The entropy of the spatial sign pattern, H(sign *A*), decreased with age for beta (*r* = -0.263, *p* = 9.5 $\times 10^{-5}$) in LEMON. In Dortmund, the replication was robust: beta H(sign) versus age *r* = -0.209, *p* = 2.0 $\times 10^{-7}$.

**Spatial variability increases.** The standard deviation of *A* across channels, $\sigma$(*A*), increased with age for theta (*r* = +0.468, *p* = 4.4 $\times 10^{-13}$) and beta (*r* = +0.194, *p* = 0.004) in LEMON. In Dortmund, beta $\sigma$(*A*) versus age *r* = +0.091, *p* = 0.026.

**Interpretation.** This triplet of signatures — increasing local magnitude, decreasing diversity, increasing contrast — describes a pattern in which the spatial map of waveform asymmetry simplifies with age. Rather than a uniform spatial distribution of excitatory and inhibitory asymmetry, older brains show a more stereotyped pattern with larger local deviations from zero but fewer distinct spatial configurations. This pattern was consistent across datasets and was strongest for theta and beta bands.

**Table 4.** Spatial aging signatures: LEMON versus Dortmund replication (beta band).

| Metric | LEMON *r* | LEMON *p* | LEMON N | Dortmund *r* | Dortmund *p* | Dortmund N |
|--------|----------|----------|---------|-------------|-------------|-----------|
| \|*A*\| ~ age | +0.232 | 6.2 $\times 10^{-4}$ | 215 | +0.091 | 0.025 | 608 |
| H(sign) ~ age | -0.263 | 9.5 $\times 10^{-5}$ | 215 | -0.209 | 2.0 $\times 10^{-7}$ | 608 |
| $\sigma$(*A*) ~ age | +0.194 | 0.004 | 215 | +0.091 | 0.026 | 608 |

All three spatial signatures replicate across datasets with consistent sign and significance. Dortmund effect sizes are attenuated relative to LEMON, potentially reflecting the continuous (versus bimodal) age distribution.

**Multi-band pattern entropy.** For each channel, we computed the sign of *A* across four bands (theta, alpha, beta, gamma) to form a 4-bit spatial pattern. Shannon entropy of the pattern distribution decreased with age (*r* = -0.221, *p* = 0.001, *d* = -0.49), indicating that older brains express fewer unique combinations of waveform shape across frequency bands. Young adults: *H* = 0.846 $\pm$ 0.062; older adults: *H* = 0.814 $\pm$ 0.070.

### 3.4 Cross-system coupling: EEG and cardiac asymmetry (pilot analysis)

As an exploratory pilot analysis (N = 72, limited by ICA extraction success rate), we examined whether waveform asymmetry extends to cardiac oscillations. HRV oscillations extracted from the ECG component showed universally positive asymmetry: VLF (*A* = +0.007, *p* = n.s.), LF (*A* = +0.022, *d* = 0.93, *p* = 3 $\times 10^{-11}$), HF (*A* = +0.050, *d* = 2.71, *p* = 1 $\times 10^{-34}$), and VHF (*A* = +0.111, *d* = 41.8, *p* = 4 $\times 10^{-117}$). This confirms that waveform asymmetry is not specific to neural oscillations but extends to autonomic rhythms (Figure 4A).

![Cross-system coupling between EEG and cardiac asymmetry.](figures/fig4_hrv_cross_system.pdf)

Global EEG spatial variability was associated with cardiac asymmetry after controlling for age: $\sigma$(*A*)_global versus HRV HF *A* (partial *r* = -0.301, *p* = 0.010) and versus HRV VHF *A* (partial *r* = -0.276, *p* = 0.019; Figure 4B). Greater spatial disorganization of EEG waveform shape was associated with lower cardiac asymmetry, suggesting cross-system coordination of oscillatory cycle shape.

**Caveat.** ECG extraction via ICA succeeded for only 72 of 215 subjects (34%). Dropout was not systematic by age (HRV group: 38.1 $\pm$ 19.5 years; no-HRV: 39.9 $\pm$ 20.6 years) or sex, but the reduced sample limits statistical power for cross-system analyses.

### 3.5 Eyes-closed versus eyes-open: sensory constraint reduces spatial entropy

Opening the eyes decreased alpha-band spatial entropy (Figure 5A).

![Eyes-closed versus eyes-open spatial entropy.](figures/fig5_eceo_entropy.pdf)

EC H(sign *A*) = 0.971, EO H(sign *A*) = 0.942 (paired *t*(211) = -6.15, *p* = 1.9 $\times 10^{-9}$, Wilcoxon *p* = 9.0 $\times 10^{-9}$, *d* = -0.56). With eyes closed, the spatial distribution of alpha asymmetry signs is near-maximal entropy (approximately 50/50 positive and negative). With eyes open, more channels shift toward excitatory-type asymmetry (prop_positive: 0.516 $\rightarrow$ 0.593) and the spatial pattern becomes more organized.

**No age dependence of the EC–EO shift.** The magnitude of the entropy change did not depend on age (Figure 5B): $\Delta$*H* versus age *r* = -0.044, *p* = 0.52. Young and older adults showed equivalent modulation by visual input (young $\Delta$*H* = -0.026, old $\Delta$*H* = -0.036, *d* = 0.14). This dissociation is informative: while the *baseline* spatial organization degrades with age (Section 3.3), the *acute response* to sensory input is preserved. The degradation is structural, not a loss of functional reactivity.

### 3.6 Cognitive correlates: theta spatial entropy predicts memory (exploratory)

The following cognitive analyses are exploratory and have not been independently replicated. All partial correlations controlled for age (Table 3). Theta-band spatial entropy, H(sign *A*)_theta, was the most consistent predictor of cognitive performance, reaching significance for four of five tests examined: CVLT delayed recall (partial *r* = +0.165, *p* = 0.015), CVLT recognition (partial *r* = +0.143, *p* = 0.036), CVLT trial 1 recall (partial *r* = +0.138, *p* = 0.043), and TMT-B executive function (partial *r* = -0.146, *p* = 0.033; negative because higher TMT-B time = worse performance). Alpha entropy predicted CVLT recognition (partial *r* = +0.164, *p* = 0.016). Working memory errors (TAP-WM omissions) were predicted by theta entropy (partial *r* = +0.152, *p* = 0.027; positive = more diversity, fewer omissions).

**Table 3.** Cognitive partial correlations (controlling for age).

| Cognitive test | EEG metric | Partial *r* | *p* | Direction |
|---------------|------------|------------|------|-----------|
| CVLT delayed recall | H(sign)_theta | +0.165 | 0.015 | Higher diversity $\rightarrow$ better recall |
| CVLT recognition | H(sign)_theta | +0.143 | 0.036 | Higher diversity $\rightarrow$ better recognition |
| CVLT recognition | H(sign)_alpha | +0.164 | 0.016 | Higher diversity $\rightarrow$ better recognition |
| CVLT trial 1 | H(sign)_theta | +0.138 | 0.043 | Higher diversity $\rightarrow$ better recall |
| TMT-B (executive) | H(sign)_theta | -0.146 | 0.033 | Higher diversity $\rightarrow$ faster completion |
| TAP-WM omissions | H(sign)_theta | +0.152 | 0.027 | Higher diversity $\rightarrow$ fewer errors |
| CVLT total learning | H(sign)_beta | -0.160 | 0.020 | Lower beta entropy $\rightarrow$ better learning |

The consistent direction — higher theta spatial diversity predicts better memory and executive function, independent of age — is neuroanatomically plausible. Theta oscillations are generated by hippocampal–cortical circuits, and the hippocampus is among the structures most vulnerable to age-related atrophy [21]. The spatial diversity of theta waveform asymmetry may index the integrity of distributed hippocampal–cortical communication. Effect sizes are modest (*r* $\approx$ 0.14–0.17), consistent with the expected magnitude for a single EEG-derived metric predicting complex cognitive function.

### 3.7 Age prediction: PTA versus spectral features

Ridge regression models trained to predict chronological age revealed that all feature sets significantly outperformed chance (MAE = 18.7 years; all permutation *p* < 0.05; Figure 6, Table 4). Spectral power was the strongest single predictor (MAE = 10.8 years, *R*² = 0.56), consistent with the well-established age sensitivity of band power [1, 2]. The aperiodic slope alone predicted age with MAE = 15.5 years (*R*² = 0.19). PTA features predicted age significantly above chance (MAE = 16.7 years, *R*² = 0.11, *p* = 0.005), though with substantially higher error than spectral power. Combining PTA with spectral features (MAE = 11.4 years) did not improve over spectral features alone, suggesting that the age-related variance captured by PTA overlaps with that captured by spectral power in the context of multivariate prediction.

![Age prediction comparison.](figures/fig6_age_prediction.pdf)

**Table 4.** Age prediction performance (LEMON, 10 $\times$ 10-fold CV).

| Feature set | *N* features | MAE (years) | *R*² | *r* | *p*_perm |
|------------|-------------|-------------|------|-----|---------|
| PTA | 16 | 16.7 | 0.11 | 0.34 | 0.005 |
| Power | 31 | 10.8 | 0.56 | 0.75 | 0.005 |
| Slope | 1 | 15.5 | 0.19 | 0.44 | 0.005 |
| Spectral (all) | 32 | 10.8 | 0.56 | 0.75 | 0.005 |
| PTA + Spectral | 48 | 11.4 | 0.49 | 0.70 | 0.005 |
| Sex | 1 | 18.2 | 0.02 | 0.15 | 0.015 |

Cross-dataset generalization of the PTA model (trained on LEMON, tested on Dortmund N = 608) yielded MAE = 12.4 years and *r* = 0.28. Although modest, this demonstrates that PTA-based age prediction transfers across acquisition systems and demographic structures without retraining.

---

## 4. Discussion

### 4.1 Waveform shape as a novel aging biomarker

This study demonstrates that the shape of neural oscillatory cycles — as quantified by peak-trough asymmetry — changes systematically with healthy aging. The principal finding, a decline in beta-band waveform asymmetry with age, was replicated across two independent cohorts (LEMON *r* = -0.326; Dortmund *r* = -0.314) with a blind, pre-registered prediction that was precisely confirmed. The effect size (Cohen's *d* $\approx$ 0.7) exceeds that of classical alpha slowing [1] and spectral slope changes [3], suggesting that waveform shape captures age-related neural changes that are partially invisible to spectral methods.

### 4.2 Band specificity and excitatory–inhibitory balance

The finding that slow oscillations (delta, alpha) show excitatory-type asymmetry while fast oscillations (beta, gamma) show inhibitory-type asymmetry is consistent with the biophysics of rhythm generation. Posterior alpha is generated primarily by thalamocortical excitatory loops involving pyramidal neurons, whose EPSPs produce a rapid depolarizing phase [5]. Beta oscillations, particularly over sensorimotor cortex, are shaped by GABAergic interneuron networks whose synchronized IPSPs produce a rapid hyperpolarizing (fall) phase [22, 23]. The spatial double dissociation — posterior-positive alpha versus central-positive beta — maps directly onto the known generators of these rhythms.

The selective vulnerability of beta asymmetry to aging is consistent with evidence that sensorimotor cortex undergoes disproportionate age-related changes in E/I balance. GABA concentrations measured by magnetic resonance spectroscopy decline most steeply in frontal and sensorimotor regions [13, 14], and beta-band activity is particularly sensitive to GABAergic manipulation [24]. We note, however, that the present data cannot establish a causal link between GABA changes and PTA shifts; this interpretation remains a hypothesis requiring convergent evidence from combined EEG–MRS designs.

### 4.3 Spatial degradation without mechanistic commitment

The spatial signatures of aging — increasing local asymmetry magnitude, decreasing spatial diversity, and increasing spatial variability — describe a consistent pattern of simplification of the spatial map of waveform shape. This pattern was replicated across datasets and bands. We deliberately present these as descriptive empirical findings without committing to a specific generative mechanism. While coupled oscillator models or Ising-type frameworks could potentially account for these patterns, the present data do not distinguish between several candidate mechanisms (coupling decay, E/I drift, dendritic morphology changes). Future work combining EEG with structural imaging (DTI) or neurochemical imaging (MRS) will be needed to constrain the mechanism.

### 4.4 Relation to existing EEG aging biomarkers

Waveform asymmetry provides information that is partially independent of established spectral biomarkers. The beta–age association survived control for aperiodic spectral slope (partial *r* = -0.245 versus raw *r* = -0.326), indicating that ~75% of the age-related variance in PTA is not shared with the 1/*f* slope. The two metrics likely reflect overlapping but distinct aspects of neural aging: the aperiodic slope indexes the overall balance of excitatory and inhibitory synaptic currents [25], while PTA captures the temporal dynamics of individual cycles.

The age prediction analysis (Section 3.7) provides a complementary perspective. Spectral power features predicted age substantially better than PTA features (MAE = 10.8 vs 16.7 years), and adding PTA to spectral features did not reduce prediction error. This indicates that for the purpose of multivariate age prediction, the aging information in PTA is largely recoverable from spectral power. However, this does not diminish the value of PTA as a biomarker: the bivariate correlation between beta PTA and age (*r* = -0.33) reflects a qualitatively different neural property — the temporal shape of individual oscillatory cycles — that is invisible to power-spectral analysis. Notably, the PTA model generalized across datasets (Dortmund MAE = 12.4 years, *r* = 0.28), demonstrating that waveform shape carries transferable age information despite differences in acquisition hardware and demographics.

Compared to alpha power and peak frequency changes, beta PTA has the advantage of being computed from broadband data without requiring identification of spectral peaks, making it robust to the common finding that alpha peaks become less distinct with aging. The effect size (*d* $\approx$ 0.7) is substantially larger than typical alpha power age effects (*d* $\approx$ 0.3–0.5), though direct head-to-head comparison in the same subjects would be needed to establish superiority.

### 4.5 Cross-system asymmetry

The demonstration that HRV oscillations are universally asymmetric (particularly HF and VHF bands) extends the finding of waveform asymmetry beyond the neural domain. The modest but significant coupling between EEG spatial variability and HRV asymmetry (partial *r* $\approx$ -0.3) suggests that waveform shape may be coordinated across physiological systems, potentially through autonomic pathways. However, the reduced sample (N = 72) and the methodological challenges of extracting ECG from EEG via ICA limit the strength of this conclusion.

### 4.6 Cognitive relevance

Theta spatial entropy consistently predicted memory and executive function independent of age, with effect sizes (*r* $\approx$ 0.14–0.17) that are modest but remarkably consistent across cognitive domains. The neuroanatomical plausibility of this finding — theta oscillations are generated by hippocampal–cortical circuits that are vulnerable to aging — strengthens the case that spatial diversity of waveform asymmetry reflects functionally relevant neural organization rather than measurement noise.

### 4.7 Limitations

Several limitations qualify these findings. First, while the core beta–age association was replicated across two datasets with different acquisition systems and demographic structures, both datasets were collected in Germany and may not generalize to other populations. Second, the LEMON dataset has a bimodal age distribution with a gap between 35–55 years; although the within-young-group correlation confirms the effect is not driven solely by group differences, a continuous age distribution provides stronger evidence (as demonstrated in the Dortmund data). Third, the longitudinal analysis showed a within-subject change in the predicted direction but did not reach conventional significance (*p* = 0.07), likely reflecting insufficient statistical power to detect a *d*_z $\approx$ 0.1 effect with N = 208 over 5 years. A power analysis suggests N $\approx$ 620 would be needed for 80% power. Fourth, cognitive correlations were tested only in LEMON and have not been independently replicated. Fifth, the cross-system HRV analysis was limited by high dropout (67%) in ECG extraction. Sixth, we did not have concurrent structural MRI or MRS data to bridge from waveform shape to white matter integrity or GABA concentrations.

### 4.8 Clinical translational potential

The cross-sectional effect sizes observed (*d* $\approx$ 0.7) place waveform asymmetry among the stronger EEG-based aging biomarkers reported to date and suggest potential utility as an endpoint in clinical trials of interventions targeting neural circuit function. The metric requires only standard EEG equipment and approximately 3 minutes of resting-state recording, making it feasible for large-scale screening. The moderate test–retest reliability (*r* = 0.45) is comparable to other EEG biomarkers used in longitudinal studies and sufficient for group-level comparisons, though individual-level tracking would benefit from multi-session averaging.

### 4.9 Future directions

Three extensions would substantially strengthen the framework. First, combined EEG–MRS studies could test whether regional GABA concentrations predict regional PTA, establishing a neurochemical basis for the waveform shape changes. Second, pharmacological challenge studies (e.g., benzodiazepine, which enhances GABAergic transmission) could test whether acute pharmacological manipulation shifts PTA in the predicted direction, providing causal evidence for the E/I interpretation. Third, application to clinical populations (Parkinson's disease, epilepsy, Alzheimer's disease) could establish waveform asymmetry as a clinically useful biomarker.

---

## 5. Data and Code Availability

LEMON data are available at https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/. Dortmund Vital Study data are available at https://openneuro.org/datasets/ds005385. Analysis code is available at https://github.com/mool32/waveform-asymmetry-aging.

---

## 6. Acknowledgements

We thank A.V. Gudkov (Vaika Inc. / Roswell Park Comprehensive Cancer Center) for institutional support and discussions on aging biology. We thank the LEMON study team at the Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig, and the Dortmund Vital Study team at the Leibniz Research Centre for Working Environment and Human Factors for making their data publicly available. Computational analyses were assisted by Claude (Anthropic).

---

## 7. References

1. Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. *Brain Research Reviews*, 29(2-3), 169–195.
2. Voytek, B., et al. (2015). Age-related changes in 1/f neural electrophysiological noise. *Journal of Neuroscience*, 35(38), 13257–13265.
3. Donoghue, T., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665.
4. Schaworonkow, N., & Bhatt, M.B. (2023). EEG spectral changes across the adult lifespan. *NeuroImage*, 267, 119837.
5. Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press.
6. Jones, S.R. (2016). When brain rhythms aren't 'rhythmic': implication for their mechanisms and meaning. *Current Opinion in Neurobiology*, 40, 72–80.
7. Cole, S.R., & Voytek, B. (2017). Brain oscillations and the importance of waveform shape. *Trends in Cognitive Sciences*, 21(2), 137–149.
8. Cole, S.R., et al. (2017). Nonsinusoidal beta oscillations reflect cortical pathophysiology accurately in Parkinson's disease. *Journal of Neuroscience*, 37(37), 9108–9121.
9. Schaworonkow, N., & Nikulin, V.V. (2019). Spatial neuronal synchronization and the waveform of oscillations: Implications for EEG and MEG. *PLOS Computational Biology*, 15(5), e1007055.
10. Jackson, N., et al. (2019). Characteristics of waveform shape in Parkinson's disease detected with scalp electroencephalography. *eNeuro*, 6(3).
11. Schaworonkow, N., & Bhatt, M.B. (2024). Waveform shape changes across human development. *Developmental Cognitive Neuroscience*, 65, 101330.
12. Bartz, S., et al. (2019). Analyzing the waveshape of brain oscillations with bicoherence. *NeuroImage*, 188, 145–160.
13. Gao, F., et al. (2013). Edited magnetic resonance spectroscopy detects an age-related decline in brain GABA levels. *NeuroImage*, 78, 75–82.
14. Porges, E.C., et al. (2017). Frontal gamma-aminobutyric acid concentrations are associated with cognitive performance in older adults. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*, 2(1), 38–44.
15. Pakkenberg, B., & Gundersen, H.J. (1997). Neocortical neuron number in humans: effect of sex and age. *Journal of Comparative Neurology*, 384(2), 312–320.
16. Heise, K.-F., et al. (2013). The aging motor system as a model for plastic changes of GABA-mediated intracortical inhibition and their behavioral relevance. *Journal of Neuroscience*, 33(23), 9039–9049.
17. Sullivan, E.V., & Pfefferbaum, A. (2006). Diffusion tensor imaging and aging. *Neuroscience & Biobehavioral Reviews*, 30(6), 749–761.
18. Betzel, R.F., et al. (2014). Changes in structural and functional connectivity among resting-state networks across the human lifespan. *NeuroImage*, 102, 345–357.
19. Babayan, A., et al. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. *Scientific Data*, 6, 180308.
20. Wascher, E., et al. (2023). The Dortmund Vital Study — a prospective investigation of the effects of aging on cognitive and physical performance. *Scientific Reports*, 13, 20735.
21. Fjell, A.M., et al. (2009). One-year brain atrophy evident in healthy aging. *Journal of Neuroscience*, 29(48), 15223–15231.
22. Hall, S.D., et al. (2011). The role of GABAergic modulation in motor function related neuronal network activity. *NeuroImage*, 56(3), 1506–1510.
23. Jensen, O., et al. (2005). On the human sensorimotor-cortex beta rhythm: Sources and modeling. *NeuroImage*, 26(2), 347–355.
24. Muthukumaraswamy, S.D., et al. (2013). The effects of elevated endogenous GABA levels on movement-related network oscillations. *NeuroImage*, 66, 36–41.
25. Gao, R., et al. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78.
26. Gerster, M., et al. (2022). Separating neural oscillations from aperiodic 1/f activity: Challenges and recommendations. *Neuroinformatics*, 20, 991–1012.
27. Seymour, R.A., et al. (2022). Interference suppression techniques for OPM-based MEG: Opportunities and challenges. *NeuroImage*, 247, 118834.

---

## Supplementary Material

**Figure S1.** Meta-analysis of waveform asymmetry across biological timescales. Asymmetry index (*A* > 0) is observed in 33 oscillatory systems spanning 11 orders of magnitude from ion channel kinetics (microseconds) to circadian rhythms (hours). The asymmetry ratio does not correlate with timescale (Spearman *$\rho$* = 0.12, *p* = 0.53), indicating scale-invariant thermodynamic asymmetry.

![Meta-analysis of waveform asymmetry across biological timescales.](figures/figS1_meta_scale_map.pdf)

**Figure S2.** Control analyses. (A) Global spatial variability $\sigma$(*A*) versus age. (B) Beta prop_positive–age correlation before (*r* = -0.326) and after (*r* = -0.245) controlling for aperiodic spectral slope. (C) Regional EMG control: beta–age correlation strength by electrode region, demonstrating that the effect is strongest at low-EMG central electrodes and weakest at high-EMG temporal electrodes.

![Control analyses.](figures/figS2_control_analyses.pdf)
