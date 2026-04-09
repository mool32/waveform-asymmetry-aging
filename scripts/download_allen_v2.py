#!/usr/bin/env python3
"""
Download pre-extracted fluorescence traces from Allen Brain Observatory.
Uses direct REST API — no allensdk required.
"""

import json
import urllib.request
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLEN_DIR = PROJECT_ROOT / "data" / "calcium" / "allen"
ALLEN_DIR.mkdir(parents=True, exist_ok=True)

API = "http://api.brain-map.org/api/v2"


def main():
    print("=" * 60)
    print("Allen Brain Observatory — Download Traces")
    print("=" * 60)

    # Step 1: Get an experiment in VISp
    print("\n1. Finding ophys experiment in VISp...")
    url = (f"{API}/data/query.json?"
           "criteria=model::OphysExperiment,"
           "rma::criteria,[targeted_structure_id$eq385],"
           "rma::options[num_rows$eq5]")
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    experiments = data['msg']
    print(f"   Found {len(experiments)} experiments")

    exp_id = experiments[0]['id']
    print(f"   Using experiment: {exp_id}")

    # Step 2: Get well-known files
    print("\n2. Finding NWB file...")
    url2 = (f"{API}/data/query.json?"
            f"criteria=model::WellKnownFile,"
            f"rma::criteria,[attachable_id$eq{exp_id}]"
            f"[attachable_type$eqOphysExperiment],"
            f"rma::options[num_rows$eq10]")
    response2 = urllib.request.urlopen(url2)
    wkf_data = json.loads(response2.read())

    nwb_file = None
    for w in wkf_data['msg']:
        fsize = w.get('file_size', 0) / (1024 * 1024)
        path = w.get('filename', w.get('path', '?'))
        print(f"   File: id={w['id']} size={fsize:.0f}MB type_id={w.get('well_known_file_type_id')}")
        # NWB files are type 375
        if w.get('well_known_file_type_id') in [375, 570] or (isinstance(path, str) and '.nwb' in path):
            if nwb_file is None or fsize < nwb_file[1]:
                nwb_file = (w['id'], fsize, path)

    if nwb_file is None:
        print("   No NWB file found! Trying direct download URL pattern...")
        # Allen has a known pattern
        nwb_url = f"{API}/well_known_file_download/{exp_id}"
    else:
        wkf_id, fsize_mb, path = nwb_file
        print(f"\n   Selected: id={wkf_id} ({fsize_mb:.0f} MB)")
        nwb_url = f"{API}/well_known_file_download/{wkf_id}"

    # Step 3: Download
    dest = ALLEN_DIR / f"ophys_experiment_{exp_id}.nwb"
    if dest.exists():
        print(f"\n3. Already downloaded: {dest.name} ({dest.stat().st_size/(1024*1024):.1f} MB)")
    else:
        print(f"\n3. Downloading NWB file...")
        try:
            urllib.request.urlretrieve(nwb_url, str(dest))
            print(f"   Done: {dest.stat().st_size/(1024*1024):.1f} MB")
        except Exception as e:
            print(f"   FAILED: {e}")
            return

    # Step 4: Extract traces
    print(f"\n4. Extracting fluorescence traces...")
    import h5py

    with h5py.File(str(dest), 'r') as f:
        # Print structure
        def show_structure(name, obj):
            if isinstance(obj, h5py.Dataset) and any(d > 100 for d in obj.shape):
                print(f"   {name}: {obj.shape} {obj.dtype}")
        f.visititems(show_structure)

        # Allen BOb NWB v1 structure:
        # processing/brain_observatory_pipeline/DfOverF/dff_traces
        # processing/brain_observatory_pipeline/Fluorescence/traces

        dff = None
        fluor = None
        timestamps = None

        # Walk all datasets to find trace-like data
        all_ds = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_ds[name] = obj.shape
        f.visititems(collect)

        # Find 2D datasets that look like traces (cells x time or time x cells)
        for name, shape in all_ds.items():
            if len(shape) == 2 and min(shape) > 5 and max(shape) > 500:
                if 'dff' in name.lower() or 'dfoverf' in name.lower():
                    dff = f[name][:]
                    print(f"\n   Found dF/F: {name} {dff.shape}")
                elif 'fluorescence' in name.lower() or 'fluor' in name.lower():
                    fluor = f[name][:]
                    print(f"\n   Found fluorescence: {name} {fluor.shape}")
                elif 'trace' in name.lower():
                    if dff is None:
                        dff = f[name][:]
                        print(f"\n   Found traces: {name} {dff.shape}")

            # Timestamps
            if len(shape) == 1 and shape[0] > 500:
                if 'timestamp' in name.lower():
                    timestamps = f[name][:]

        # If nothing found with keywords, take the largest 2D float dataset
        if dff is None and fluor is None:
            best_name = None
            best_size = 0
            for name, shape in all_ds.items():
                if len(shape) == 2 and min(shape) > 5:
                    size = shape[0] * shape[1]
                    if size > best_size:
                        dtype = f[name].dtype
                        if np.issubdtype(dtype, np.floating):
                            best_size = size
                            best_name = name
            if best_name:
                dff = f[best_name][:]
                print(f"\n   Using largest 2D float dataset: {best_name} {dff.shape}")

    traces = dff if dff is not None else fluor
    if traces is None:
        print("   ERROR: No trace data found in NWB!")
        return

    # Ensure (n_cells, n_timepoints) orientation
    if traces.shape[0] > traces.shape[1]:
        traces = traces.T
        print(f"   Transposed to (cells, time): {traces.shape}")

    n_cells, n_frames = traces.shape

    # Compute dF/F if we only have raw fluorescence
    if fluor is not None and dff is None:
        f0 = np.percentile(traces, 10, axis=1, keepdims=True)
        dff_computed = (traces - f0) / f0
    else:
        dff_computed = traces

    # Determine sampling rate
    if timestamps is not None and len(timestamps) > 1:
        fs = 1.0 / np.median(np.diff(timestamps))
    else:
        fs = 30.0  # Allen standard

    # Save
    outpath = ALLEN_DIR / f"allen_VISp_{exp_id}_traces.npz"
    np.savez(
        str(outpath),
        traces=traces,
        dff=dff_computed,
        timestamps=timestamps if timestamps is not None else np.arange(n_frames) / fs,
        fs=fs,
        n_cells=n_cells,
        n_frames=n_frames,
        metadata=np.array(json.dumps({
            'source': 'Allen Brain Observatory',
            'experiment_id': int(exp_id),
            'structure': 'VISp',
            'n_cells': int(n_cells),
            'n_frames': int(n_frames),
            'fs': float(fs),
        }))
    )

    print(f"\n   SAVED: {outpath}")
    print(f"   {n_cells} cells, {n_frames} timepoints, {fs:.1f} Hz")
    print(f"   Duration: {n_frames/fs:.1f} seconds")

    # Quick asymmetry stats
    from scipy.stats import skew
    s = skew(dff_computed, axis=1)
    print(f"   Trace skewness: mean={s.mean():.3f}, std={s.std():.3f}")
    print(f"   (Positive skew expected from calcium indicator dynamics)")


if __name__ == "__main__":
    main()
