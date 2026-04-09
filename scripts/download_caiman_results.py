#!/usr/bin/env python3
"""
Download CaImAn pre-processed benchmark results from Zenodo.
WEBSITE_basic.zip contains analysis results with extracted traces.

Also attempts to download from the Allen Brain Observatory API
without requiring allensdk — uses direct REST API calls.
"""

import os
import json
import urllib.request
import zipfile
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "calcium"
CAIMAN_DIR = DATA_DIR / "caiman"
ALLEN_DIR = DATA_DIR / "allen"

for d in [CAIMAN_DIR, ALLEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_with_progress(url, dest_path, desc=""):
    """Download with simple progress."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return True
    print(f"  Downloading {desc or dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Done: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


# =========================================================================
# 1. CaImAn WEBSITE_basic.zip from Zenodo (80 MB, has analysis results)
# =========================================================================
def download_caiman_website_basic():
    print("\n=== CaImAn WEBSITE_basic.zip (pre-processed results) ===")

    zenodo_url = "https://zenodo.org/api/records/1659149"
    try:
        response = urllib.request.urlopen(zenodo_url)
        record = json.loads(response.read())

        for f in record.get('files', []):
            if f['key'] == 'WEBSITE_basic.zip':
                url = f['links']['self']
                dest = CAIMAN_DIR / 'WEBSITE_basic.zip'
                if download_with_progress(url, dest, "WEBSITE_basic.zip (80 MB)"):
                    # Extract
                    print("  Extracting...")
                    with zipfile.ZipFile(str(dest), 'r') as zf:
                        # List contents first
                        names = zf.namelist()
                        print(f"  Archive contains {len(names)} files")
                        for n in names[:20]:
                            print(f"    {n}")
                        if len(names) > 20:
                            print(f"    ... and {len(names)-20} more")

                        # Extract to subdirectory
                        extract_dir = CAIMAN_DIR / "website_basic"
                        extract_dir.mkdir(exist_ok=True)
                        zf.extractall(str(extract_dir))
                        print(f"  Extracted to: {extract_dir}")

                    # Look for HDF5/NPZ files with traces
                    for root, dirs, files in os.walk(str(extract_dir)):
                        for fname in files:
                            fpath = Path(root) / fname
                            fsize = fpath.stat().st_size / (1024*1024)
                            if fname.endswith(('.hdf5', '.h5', '.npz', '.mat', '.npy')):
                                print(f"  Found data file: {fname} ({fsize:.1f} MB)")
                return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


# =========================================================================
# 2. Allen Brain Observatory — direct API without allensdk
# =========================================================================
def download_allen_traces_api():
    """
    Use the Allen Brain Observatory API directly to get fluorescence traces.
    No allensdk required — just REST API + NWB file download.
    """
    print("\n=== Allen Brain Observatory (direct API) ===")

    api_base = "http://api.brain-map.org/api/v2"

    try:
        # Step 1: Find an ophys experiment
        # Get experiments from primary visual cortex with natural scenes
        query = (f"{api_base}/data/query.json?"
                 "criteria=model::OphysExperiment,"
                 "rma::criteria,[targeted_structure_id$eq385],"  # VISp
                 "rma::options[num_rows$eq5][order$eq'id']")

        print("  Querying Allen API for experiments...")
        response = urllib.request.urlopen(query)
        data = json.loads(response.read())

        if not data.get('msg'):
            print("  No experiments found")
            return False

        exp = data['msg'][0]
        exp_id = exp['id']
        print(f"  Found experiment: {exp_id}")

        # Step 2: Get the NWB file URL
        # Allen provides processed data via well-known URL patterns
        nwb_url = (f"http://api.brain-map.org/api/v2/well_known_file_download/"
                   f"{exp_id}")

        # The NWB files are large. Instead, use the cell-level API
        # to get fluorescence trace data directly

        # Step 3: Get cell specimens for this experiment
        cell_query = (f"{api_base}/data/query.json?"
                      f"criteria=model::OphysCellSpecimen,"
                      f"rma::criteria,[ophys_experiment_id$eq{exp_id}],"
                      f"rma::options[num_rows$eq200]")

        print(f"  Querying cells for experiment {exp_id}...")
        response = urllib.request.urlopen(cell_query)
        cell_data = json.loads(response.read())

        cells = cell_data.get('msg', [])
        print(f"  Found {len(cells)} cells")

        if cells:
            # Save cell metadata
            cell_info = []
            for c in cells:
                cell_info.append({
                    'cell_id': c.get('id'),
                    'x': c.get('x'),
                    'y': c.get('y'),
                    'width': c.get('width'),
                    'height': c.get('height'),
                    'valid_roi': c.get('valid_roi'),
                    'max_correlation': c.get('max_correlation'),
                })

            # Save cell metadata
            with open(str(ALLEN_DIR / f"allen_exp{exp_id}_cells.json"), 'w') as f:
                json.dump(cell_info, f, indent=2)
            print(f"  Saved cell metadata to allen_exp{exp_id}_cells.json")

            # For actual traces, we need the NWB file
            # Let's download a small one
            # First check what well-known files are available
            wkf_query = (f"{api_base}/data/query.json?"
                         f"criteria=model::WellKnownFile,"
                         f"rma::criteria,[attachable_id$eq{exp_id}],"
                         f"[attachable_type$eq'OphysExperiment'],"
                         f"rma::options[num_rows$eq10]")

            response = urllib.request.urlopen(wkf_query)
            wkf_data = json.loads(response.read())

            for wkf in wkf_data.get('msg', []):
                print(f"    File: {wkf.get('filename', 'unknown')} "
                      f"({wkf.get('file_size', 0)/(1024*1024):.1f} MB) "
                      f"type: {wkf.get('well_known_file_type_id')}")

            # Download the NWB file (contains traces)
            # These are typically 100-500 MB
            for wkf in wkf_data.get('msg', []):
                if wkf.get('filename', '').endswith('.nwb'):
                    fsize_mb = wkf.get('file_size', 0) / (1024*1024)
                    if fsize_mb < 300:  # Only if under 300 MB
                        dl_url = f"http://api.brain-map.org/api/v2/well_known_file_download/{wkf['id']}"
                        dest = ALLEN_DIR / wkf['filename']
                        print(f"  Downloading NWB file ({fsize_mb:.0f} MB)...")
                        if download_with_progress(dl_url, dest, f"NWB ({fsize_mb:.0f} MB)"):
                            # Extract traces from NWB
                            extract_traces_from_nwb(dest, exp_id)
                        break
            return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_traces_from_nwb(nwb_path, exp_id):
    """Extract fluorescence traces from an Allen NWB file using h5py."""
    print(f"  Extracting traces from {nwb_path.name}...")
    try:
        import h5py

        with h5py.File(str(nwb_path), 'r') as f:
            # Allen NWB structure
            # /processing/brain_observatory_pipeline/Fluorescence/
            #   DfOverF/data  — dF/F traces
            #   DfOverF/timestamps
            # /processing/brain_observatory_pipeline/ImageSegmentation/

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"    {name}: {obj.shape} {obj.dtype}")

            print("  NWB structure (datasets):")
            f.visititems(print_structure)

            # Try to find fluorescence data
            trace_paths = [
                'processing/brain_observatory_pipeline/DfOverF/DfOverF/data',
                'processing/brain_observatory_pipeline/Fluorescence/Fluorescence/data',
                'processing/brain_observatory_pipeline/DfOverF/dff/data',
            ]

            timestamps_paths = [
                'processing/brain_observatory_pipeline/DfOverF/DfOverF/timestamps',
                'processing/brain_observatory_pipeline/Fluorescence/Fluorescence/timestamps',
            ]

            dff = None
            timestamps = None

            for tp in trace_paths:
                if tp in f:
                    dff = f[tp][:]
                    print(f"  Found traces at {tp}: {dff.shape}")
                    break

            for tp in timestamps_paths:
                if tp in f:
                    timestamps = f[tp][:]
                    print(f"  Found timestamps at {tp}: {timestamps.shape}")
                    break

            if dff is not None:
                outpath = ALLEN_DIR / f"allen_VISp_{exp_id}_traces.npz"
                save_dict = {'dff': dff}
                if timestamps is not None:
                    save_dict['timestamps'] = timestamps
                    fs = 1.0 / np.median(np.diff(timestamps))
                    save_dict['fs'] = fs
                    print(f"  Sampling rate: {fs:.2f} Hz")

                np.savez(str(outpath), **save_dict)
                print(f"  Saved extracted traces: {outpath.name}")
                print(f"  Shape: {dff.shape}")
                return True

    except Exception as e:
        print(f"  Error extracting: {e}")
        import traceback
        traceback.print_exc()
    return False


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Download Real Calcium Imaging Traces")
    print("=" * 60)

    # Try CaImAn pre-processed results
    download_caiman_website_basic()

    # Try Allen direct API
    download_allen_traces_api()

    print("\n" + "=" * 60)
    print("Final data inventory:")
    print("=" * 60)
    for root, dirs, files in os.walk(str(DATA_DIR)):
        for fname in sorted(files):
            fpath = Path(root) / fname
            fsize = fpath.stat().st_size / (1024*1024)
            rel = fpath.relative_to(DATA_DIR)
            print(f"  {rel} ({fsize:.1f} MB)")
