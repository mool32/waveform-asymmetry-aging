# Waveform Asymmetry as a Biomarker of Neural Aging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19912202.svg)](https://doi.org/10.5281/zenodo.19912202)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Preprint](https://img.shields.io/badge/preprint-PDF-red.svg)](paper/preprint.pdf)

Code and analysis pipeline for:

> **Waveform Asymmetry as a Biomarker of Neural Aging: Spatial Degradation of Oscillatory Cycle Shape Across Two Independent Cohorts**
> Theodor Spiro (Vaika Inc., East Aurora, NY, USA)
> *Under review at Frontiers in Aging Neuroscience*
> Preprint: [paper/preprint.pdf](paper/preprint.pdf) · DOI: [10.5281/zenodo.19912202](https://doi.org/10.5281/zenodo.19912202)
> Contact: tspiro@vaika.org

---

## Paper summary

The shape of each neural oscillatory cycle - quantified by **peak-trough asymmetry (PTA)**, the ratio of rise to fall time - may reflect excitatory-inhibitory balance in the generating circuit. We measured broadband PTA across five frequency bands in resting-state EEG from **215 adults** (LEMON, age 20-77) and replicated all findings in **608 adults** (Dortmund Vital Study, age 20-70), including **208 with 5-year longitudinal follow-up**.

**Three findings:**

1. **Asymmetry is band-specific and spatially structured** - slow rhythms show excitatory-type asymmetry, fast rhythms show inhibitory-type asymmetry, with a spatial double dissociation between posterior alpha and central beta.
2. **Beta-band asymmetry decreases with age** (LEMON: *r* = -0.326, *d* = 0.69; Dortmund: *r* = -0.314, *d* = 0.75), surviving control for aperiodic slope. Effect size exceeds classical alpha slowing. Longitudinal data confirmed the predicted direction (mean Δ = -0.017, *r* = 0.45 test-retest).
3. **Spatial organization degrades with age**, and **theta spatial entropy predicts memory performance independent of age**.

**Bottom line:** Waveform shape provides a power-independent measure of neural aging that complements existing spectral biomarkers, accessible from standard 3-minute EEG recordings.

---

## Quick start

```bash
git clone https://github.com/mool32/waveform-asymmetry-aging.git
cd waveform-asymmetry-aging
pip install -r requirements.txt

# Main analysis script - PTA computation across LEMON dataset
python src/phase1/run_pta.py

# Verify every statistic cited in the paper
python scripts/verify_numbers.py
```

### Main analysis entry points

| Script | Purpose |
|--------|---------|
| **[`src/phase1/run_pta.py`](src/phase1/run_pta.py)** | **Main PTA pipeline** (LEMON, broadband across 5 bands) |
| [`scripts/analyze_dortmund_correlations.py`](scripts/analyze_dortmund_correlations.py) | Dortmund replication (N=608) |
| [`scripts/analyze_longitudinal.py`](scripts/analyze_longitudinal.py) | 5-year longitudinal analysis (N=208) |
| [`scripts/run_age_prediction.py`](scripts/run_age_prediction.py) | Age-prediction Ridge models (PTA vs spectral) |
| [`scripts/generate_paper_figures.py`](scripts/generate_paper_figures.py) | Regenerate all 6 main + 2 supplementary figures |
| [`scripts/verify_numbers.py`](scripts/verify_numbers.py) | Cross-check every number in the manuscript |

---

## Repository structure

```
.
├── README.md                         # This file
├── paper/
│   ├── preprint.pdf                  # Full preprint with figures (24 pp)
│   ├── manuscript_submission.md      # Source markdown
│   └── frontiers/                    # Frontiers submission package
├── src/
│   ├── phase1/
│   │   ├── run_pta.py                # ★ Main PTA pipeline
│   │   ├── run_eceo_entropy.py       # Eyes-closed vs open
│   │   ├── statistics.py             # Statistical tests
│   │   └── visualize.py              # Plotting
│   ├── utils/
│   │   └── asymmetry.py              # Core PTA computation
│   └── phase4/
│       └── meta_analysis.py          # Cross-scale meta-analysis (33 systems)
├── scripts/
│   ├── analyze_longitudinal.py
│   ├── analyze_dortmund_correlations.py
│   ├── run_age_prediction.py
│   ├── generate_paper_figures.py
│   └── verify_numbers.py
└── results/
    ├── phase1_full/                  # LEMON outputs (summary.json)
    ├── dortmund_replication/         # Dortmund cross-sectional
    ├── dortmund_longitudinal/        # 5-year follow-up
    ├── eceo_entropy/                 # Eyes-closed/open
    └── figures/                      # Generated figures (PDF/PNG)
```

---

## Datasets

- **LEMON** ([MPI Leipzig Mind-Brain-Body](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)) - N=215, age 20-77
- **Dortmund Vital Study** ([OpenNeuro ds005385](https://openneuro.org/datasets/ds005385)) - N=608, age 20-70, including N=208 with 5-year longitudinal follow-up

Both datasets are openly licensed; see source links for terms.

---

## Citation

If you use this code or build on these findings, please cite:

```bibtex
@article{Spiro2026waveform,
  author  = {Spiro, Theodor},
  title   = {Waveform Asymmetry as a Biomarker of Neural Aging:
             Spatial Degradation of Oscillatory Cycle Shape Across
             Two Independent Cohorts},
  year    = {2026},
  doi     = {10.5281/zenodo.19912202},
  url     = {https://github.com/mool32/waveform-asymmetry-aging}
}
```

---

## Requirements

Python 3.10+. Dependencies in `requirements.txt`:
`numpy`, `scipy`, `matplotlib`, `mne`, `pandas`, `statsmodels`, `scikit-learn`.

---

## Contact

**Theodor Spiro** - Vaika Inc., East Aurora, NY, USA
Email: tspiro@vaika.org

---

## License

[MIT License](LICENSE) - free to reuse with attribution.
