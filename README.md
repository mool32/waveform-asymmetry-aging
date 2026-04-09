# Waveform Asymmetry as a Biomarker of Neural Aging

Code and analysis pipeline for:

> **Waveform asymmetry as a biomarker of neural aging: spatial degradation of oscillatory cycle shape across two independent cohorts**

## Key finding

Beta-band peak-trough asymmetry (PTA) decreases with age in two independent EEG datasets (LEMON N=215, r=-0.326; Dortmund N=608, r=-0.314), with effect size d~0.7 — stronger than classical alpha slowing. The spatial organization of waveform asymmetry also degrades with age, and theta spatial entropy predicts memory performance independent of age.

## Repository structure

```
src/
  phase1/           # EEG preprocessing and PTA computation
    run_pta.py      # Main broadband PTA pipeline
    run_eceo_entropy.py  # Eyes-closed vs eyes-open analysis
    statistics.py   # Statistical tests for all predictions
    visualize.py    # Plotting functions
  utils/
    asymmetry.py    # Core asymmetry computation functions
  phase4/
    meta_analysis.py  # Cross-scale meta-analysis (33 systems)

scripts/
  analyze_longitudinal.py       # 5-year longitudinal analysis (N=208)
  analyze_dortmund_correlations.py  # Dortmund replication statistics
  generate_paper_figures.py     # Regenerate all paper figures
  verify_numbers.py             # Cross-check all cited statistics

paper/
  manuscript.md     # Full manuscript (BioRxiv format)
  figures/          # All publication figures (PDF, 300 DPI)

results/
  phase1_full/      # LEMON results (summary.json)
  dortmund_replication/  # Dortmund cross-sectional results
  dortmund_longitudinal/ # Longitudinal follow-up data
  eceo_entropy/     # Eyes-closed/open results
  figures/          # Development figures
```

## Datasets

- **LEMON:** [MPI Leipzig Mind-Brain-Body](https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/) (N=215, age 20-77)
- **Dortmund Vital Study:** [OpenNeuro ds005385](https://openneuro.org/datasets/ds005385) (N=608, age 20-70, including N=208 longitudinal)

## Reproducing the analysis

```bash
pip install -r requirements.txt

# Run main PTA pipeline on LEMON data (requires downloaded EEG files)
python src/phase1/run_pta.py

# Compute Dortmund age correlations (from pre-computed summary)
python scripts/analyze_dortmund_correlations.py

# Longitudinal analysis
python scripts/analyze_longitudinal.py

# Verify all statistics cited in the paper
python scripts/verify_numbers.py

# Regenerate all figures
python scripts/generate_paper_figures.py
```

## Requirements

Python 3.10+. See `requirements.txt` for dependencies (numpy, scipy, matplotlib, mne-python, pandas, statsmodels).

## License

MIT
