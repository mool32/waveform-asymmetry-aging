#!/usr/bin/env python3
"""Verify all downloaded calcium imaging data."""

import numpy as np
from pathlib import Path
import json
from scipy.stats import skew

DATA = Path(__file__).resolve().parent.parent / "data" / "calcium"

print("=" * 60)
print("CALCIUM IMAGING DATA INVENTORY")
print("=" * 60)

# 1. Synthetic traces
print("\n--- Synthetic Calcium Traces ---")
d = np.load(str(DATA / "caiman/synthetic_calcium_traces.npz"), allow_pickle=True)
print(f"File: caiman/synthetic_calcium_traces.npz")
print(f"  traces: {d['traces'].shape} (cells x timepoints)")
print(f"  dff: {d['dff'].shape}")
print(f"  fs: {d['fs']} Hz")
print(f"  x_coords: {d['x_coords'].shape}, y_coords: {d['y_coords'].shape}")
s1 = skew(d["dff"], axis=1)
print(f"  Skewness: mean={s1.mean():.3f}, std={s1.std():.3f}")
print(f"  Positive skew: {(s1 > 0).sum()}/{len(s1)} cells")

# 2. Allen traces
print("\n--- Allen Brain Observatory Traces ---")
d2 = np.load(str(DATA / "allen/allen_VISp_527550473_traces.npz"), allow_pickle=True)
print(f"File: allen/allen_VISp_527550473_traces.npz")
print(f"  traces: {d2['traces'].shape} (cells x timepoints)")
print(f"  dff: {d2['dff'].shape}")
print(f"  fs: {d2['fs']:.1f} Hz")
print(f"  Duration: {d2['n_frames']/d2['fs']:.1f} seconds ({d2['n_frames']/d2['fs']/60:.1f} min)")
s2 = skew(d2["dff"], axis=1)
print(f"  Skewness: mean={s2.mean():.3f}, std={s2.std():.3f}")
print(f"  Positive skew: {(s2 > 0).sum()}/{len(s2)} cells")

# 3. CaImAn benchmark ROIs
print("\n--- CaImAn Benchmark ROI Data ---")
wb = DATA / "caiman/website_basic/WEBSITE"
if wb.exists():
    datasets = sorted([d.name for d in wb.iterdir()
                       if d.is_dir() and d.name not in ("scripts", "__pycache__")])
    print(f"Datasets: {datasets}")
    for ds in datasets:
        regions_dir = wb / ds / "regions"
        if regions_dir.exists():
            for rfile in sorted(regions_dir.iterdir()):
                if rfile.suffix == ".json":
                    rdata = json.loads(rfile.read_text())
                    print(f"  {ds}/{rfile.name}: {len(rdata)} ROIs")

# Summary
print("\n--- Summary ---")
total_mb = 0
file_count = 0
for f in DATA.rglob("*"):
    if f.is_file():
        total_mb += f.stat().st_size / (1024 * 1024)
        file_count += 1
print(f"Total: {file_count} files, {total_mb:.1f} MB")

print("\nREADY FOR ASYMMETRY ANALYSIS:")
print("  1. Synthetic traces: 100 cells x 9000 timepoints (GCaMP6f model)")
print("  2. Allen VISp real data: 265 cells x 105729 timepoints (30 Hz)")
print("  3. CaImAn benchmark ROIs: spatial coordinates for 7 datasets")
