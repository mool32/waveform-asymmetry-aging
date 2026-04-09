#!/usr/bin/env python3
"""
Download pre-extracted fluorescence traces from Allen Brain Observatory
using direct REST API (no allensdk required).

Downloads a small NWB file and extracts dF/F traces using h5py.
"""

import json
import urllib.request
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLEN_DIR = PROJECT_ROOT / "data" / "calcium" / "allen"
ALLEN_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "http://api.brain-map.org/api/v2"


def api_query(endpoint):
    """Query Allen API and return parsed JSON."""
    url = f"{API_BASE}/{endpoint}"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


def download_file(url, dest, desc=""):
    """Download file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest.name} ({dest.stat().st_size/(1024*1024):.1f} MB)")
        return True
    print(f"  Downloading {desc or dest.name}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Done: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def find_small_experiment():
    """Find an ophys experiment with a small NWB file."""
    print("  Searching for experiments in VISp...")

    # Query ophys experiments in primary visual cortex
    # The API returns JSON with 'msg' containing experiment records
    data = api_query(
        "data/query.json?"
        "criteria=model::OphysExperiment,"
        "rma::criteria,[targeted_structure_id$eq385],"  # VISp
        "rma::include,well_known_files(well_known_file_type[name$eq'NWBOphys']),"
        "rma::options[num_rows$eq20][order$eq'id']"
    )

    experiments = data.get('msg', [])
    print(f"  Found {len(experiments)} experiments")

    # Find one with a small NWB file
    best_exp = None
    best_size = float('inf')

    for exp in experiments:
        wkfs = exp.get('well_known_files', [])
        for wkf in wkfs:
            fsize = wkf.get('file_size', 0)
            fsize_mb = fsize / (1024 * 1024)
            if 50 < fsize_mb < 500 and fsize_mb < best_size:
                best_exp = exp
                best_size = fsize_mb
                best_wkf = wkf

    if best_exp:
        print(f"  Selected experiment {best_exp['id']}: NWB = {best_size:.0f} MB")
        return best_exp, best_wkf
    else:
        # Fallback: just use first experiment
        if experiments:
            exp = experiments[0]
            wkfs = exp.get('well_known_files', [])
            if wkfs:
                return exp, wkfs[0]
    return None, None


def extract_traces_from_nwb(nwb_path, exp_id):
    """Extract fluorescence traces from Allen NWB file."""
    import h5py

    print(f"\n  Extracting traces from {nwb_path.name}...")

    with h5py.File(str(nwb_path), 'r') as f:
        # Print top-level structure
        print("  NWB top-level groups:")
        for key in f.keys():
            print(f"    /{key}")

        # Find processing pipeline
        if 'processing' in f:
            for pname in f['processing'].keys():
                print(f"    /processing/{pname}")
                proc = f['processing'][pname]
                if hasattr(proc, 'keys'):
                    for dname in proc.keys():
                        obj = proc[dname]
                        print(f"      /{dname}", end="")
                        if hasattr(obj, 'shape'):
                            print(f" shape={obj.shape} dtype={obj.dtype}")
                        elif hasattr(obj, 'keys'):
                            print(f" (group with {list(obj.keys())})")
                        else:
                            print()

        # Try various paths for fluorescence data
        dff_data = None
        raw_fluor = None
        timestamps = None
        roi_ids = None

        # Common paths in Allen NWB files
        search_paths = {
            'dff': [
                'processing/brain_observatory_pipeline/DfOverF/dff_traces',
                'processing/brain_observatory_pipeline/DfOverF/DfOverF/data',
                'processing/brain_observatory_pipeline/DfOverF/data',
            ],
            'fluorescence': [
                'processing/brain_observatory_pipeline/Fluorescence/traces',
                'processing/brain_observatory_pipeline/Fluorescence/Fluorescence/data',
                'processing/brain_observatory_pipeline/Fluorescence/data',
            ],
            'timestamps': [
                'processing/brain_observatory_pipeline/DfOverF/dff_timestamps',
                'processing/brain_observatory_pipeline/DfOverF/DfOverF/timestamps',
                'processing/brain_observatory_pipeline/Fluorescence/timestamps',
            ],
        }

        def find_datasets(group, prefix=""):
            """Recursively find all datasets."""
            results = []
            if hasattr(group, 'items'):
                for name, item in group.items():
                    path = f"{prefix}/{name}"
                    if isinstance(item, h5py.Dataset):
                        results.append((path, item.shape, item.dtype))
                    elif isinstance(item, h5py.Group):
                        results.extend(find_datasets(item, path))
            return results

        print("\n  All datasets in file:")
        all_datasets = find_datasets(f)
        for path, shape, dtype in all_datasets:
            if any(dim > 100 for dim in shape):  # Only show substantial datasets
                print(f"    {path}: {shape} {dtype}")

        # Try to find dF/F
        for path in search_paths['dff']:
            if path in f:
                dff_data = f[path][:]
                print(f"\n  Found dF/F at {path}: {dff_data.shape}")
                break

        # Try raw fluorescence
        if dff_data is None:
            for path in search_paths['fluorescence']:
                if path in f:
                    raw_fluor = f[path][:]
                    print(f"\n  Found fluorescence at {path}: {raw_fluor.shape}")
                    break

        # Try timestamps
        for path in search_paths['timestamps']:
            if path in f:
                timestamps = f[path][:]
                break

        # If we still don't have data, search more broadly
        if dff_data is None and raw_fluor is None:
            print("\n  Searching for large 2D datasets (likely traces)...")
            for path, shape, dtype in all_datasets:
                if len(shape) == 2 and min(shape) > 10 and max(shape) > 1000:
                    print(f"  Loading {path}: {shape}")
                    dff_data = f[path][:]
                    break

        # Save what we found
        traces = dff_data if dff_data is not None else raw_fluor
        if traces is not None:
            # Ensure shape is (n_cells, n_timepoints)
            if traces.shape[0] > traces.shape[1]:
                traces = traces.T
                print(f"  Transposed to: {traces.shape}")

            n_cells, n_frames = traces.shape

            save_dict = {
                'traces': traces,
                'n_cells': n_cells,
                'n_frames': n_frames,
            }

            if dff_data is not None:
                save_dict['dff'] = traces
            if raw_fluor is not None:
                save_dict['fluorescence'] = raw_fluor
            if timestamps is not None:
                save_dict['timestamps'] = timestamps
                fs = 1.0 / np.median(np.diff(timestamps))
                save_dict['fs'] = fs
                print(f"  Sampling rate: {fs:.2f} Hz")
            else:
                # Allen standard is ~30 Hz for 2P
                save_dict['fs'] = 30.0

            save_dict['metadata'] = np.array(json.dumps({
                'source': 'Allen Brain Observatory',
                'experiment_id': int(exp_id),
                'structure': 'VISp',
                'n_cells': int(n_cells),
                'n_frames': int(n_frames),
            }))

            outpath = ALLEN_DIR / f"allen_VISp_{exp_id}_traces.npz"
            np.savez(str(outpath), **save_dict)
            print(f"\n  SAVED: {outpath.name}")
            print(f"    {n_cells} cells, {n_frames} timepoints")

            # Quick stats
            from scipy.stats import skew
            trace_skew = skew(traces, axis=1)
            print(f"    Trace skewness: mean={trace_skew.mean():.3f} +/- {trace_skew.std():.3f}")

            return outpath
        else:
            print("  Could not find trace data in NWB file")
            return None


def main():
    print("=" * 60)
    print("Allen Brain Observatory - Download Fluorescence Traces")
    print("=" * 60)

    exp, wkf = find_small_experiment()
    if exp is None:
        print("No suitable experiment found!")
        return

    exp_id = exp['id']

    # Download NWB
    wkf_id = wkf.get('id')
    fname = wkf.get('filename', f'ophys_experiment_{exp_id}.nwb')
    dl_url = f"http://api.brain-map.org/api/v2/well_known_file_download/{wkf_id}"
    fsize_mb = wkf.get('file_size', 0) / (1024 * 1024)

    dest = ALLEN_DIR / fname
    if download_file(dl_url, dest, f"{fname} ({fsize_mb:.0f} MB)"):
        result = extract_traces_from_nwb(dest, exp_id)
        if result:
            print(f"\nSuccess! Traces saved to: {result}")
        else:
            print("\nFailed to extract traces from NWB file.")
    else:
        print("\nFailed to download NWB file.")


if __name__ == "__main__":
    main()
