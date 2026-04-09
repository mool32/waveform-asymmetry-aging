#!/usr/bin/env python3
"""
Download calcium imaging benchmark datasets for asymmetry analysis.

Sources:
  1. CaImAn demo data (Flatiron Institute) — raw 2-photon movies
  2. NeuroFinder datasets — ROI coordinates + images
  3. Allen Brain Observatory — extracted fluorescence traces via AllenSDK
  4. CRCNS datasets — various calcium imaging experiments

Run this script from the project root:
    python scripts/download_calcium_data.py

Requirements:
    pip install h5py numpy scipy requests tqdm
    # For Allen data: pip install allensdk
"""

import os
import sys
import json
import urllib.request
import zipfile
import tempfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "calcium"
CAIMAN_DIR = DATA_DIR / "caiman"
NEUROFINDER_DIR = DATA_DIR / "neurofinder"
ALLEN_DIR = DATA_DIR / "allen"

for d in [CAIMAN_DIR, NEUROFINDER_DIR, ALLEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path, desc=None):
    """Download a file with progress reporting."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return True
    desc = desc or dest_path.name
    print(f"  Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Done: {dest_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


# =========================================================================
# 1. CaImAn demo data — pre-extracted results (HDF5 with traces)
# =========================================================================
def download_caiman_data():
    """
    Download CaImAn demo results. The key file is the analysis result HDF5
    which contains extracted temporal traces (C, S matrices) and spatial
    footprints (A matrix).

    We use the pre-computed results from the CaImAn paper benchmarks.
    """
    print("\n=== CaImAn Demo Data ===")

    base_url = "https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/"

    # These are analysis results with extracted traces
    files = {
        # Pre-computed CNMF results with traces
        "analysis_results_nf00.00.npz": "analysis results neurofinder 00.00",
        # The classic Sue demo (raw movie - large but useful)
        # "Sue_2x_3000_40_-46.tif": "Sue 2-photon demo movie (163 MB)",
    }

    # Also try to get pre-extracted results from CaImAn paper
    paper_urls = {
        # CaImAn paper benchmark results hosted on Zenodo
        # These contain extracted C (calcium traces), S (spikes), A (spatial)
    }

    for fname, desc in files.items():
        download_file(base_url + fname, CAIMAN_DIR / fname, desc)

    # Download the small demo data that ships with CaImAn on GitHub
    demo_urls = [
        # Ground truth data used in CaImAn paper (small, has traces)
        ("https://raw.githubusercontent.com/flatironinstitute/CaImAn/main/"
         "caiman/tests/comparison/comparison.py", "comparison_code.py"),
    ]

    print("\n  Trying CaImAn paper benchmark data from Zenodo...")
    # CaImAn paper DOI: 10.7554/eLife.38173
    # Benchmark data: https://zenodo.org/record/1659149
    zenodo_url = "https://zenodo.org/api/records/1659149"
    try:
        import json
        response = urllib.request.urlopen(zenodo_url)
        record = json.loads(response.read())
        print(f"  Found Zenodo record: {record.get('metadata', {}).get('title', 'unknown')}")
        for f in record.get('files', []):
            fname = f['key']
            fsize = f['size'] / (1024*1024)
            print(f"    Available: {fname} ({fsize:.1f} MB)")
            # Only download small files with extracted traces
            if fsize < 100 and ('.npz' in fname or '.mat' in fname or '.hdf5' in fname):
                download_file(
                    f['links']['self'],
                    CAIMAN_DIR / fname,
                    f"{fname} ({fsize:.1f} MB)"
                )
    except Exception as e:
        print(f"  Could not access Zenodo: {e}")


# =========================================================================
# 2. NeuroFinder — ROI coordinates (we need to extract traces ourselves)
# =========================================================================
def download_neurofinder_data():
    """
    Download a small NeuroFinder training dataset.
    Contains images + ground truth ROI regions.
    """
    print("\n=== NeuroFinder Data ===")

    # NeuroFinder datasets are hosted on S3
    # Format: images (TIFF stack) + regions (JSON with coordinates)
    # The smallest dataset is neurofinder.00.00 (~900 MB) - too large for quick download

    # Instead, get just the ground truth regions (coordinates only, small)
    # These are in the neurofinder-datasets repo
    regions_url = ("https://raw.githubusercontent.com/codeneuro/neurofinder-datasets/"
                   "master/00/00.train/regions/regions.json")

    download_file(regions_url, NEUROFINDER_DIR / "nf_00.00_regions.json",
                  "NeuroFinder 00.00 ROI regions")

    print("  Note: Full NeuroFinder datasets (~1 GB each) contain TIFF stacks.")
    print("  For extracted traces, use Allen Brain Observatory instead.")


# =========================================================================
# 3. Allen Brain Observatory — pre-extracted fluorescence traces (BEST)
# =========================================================================
def download_allen_data():
    """
    Download pre-extracted calcium traces from the Allen Brain Observatory.
    This is the BEST source for our purposes: real fluorescence traces
    with spatial coordinates, already extracted and quality-controlled.

    Uses the AllenSDK if available, otherwise provides download instructions.
    """
    print("\n=== Allen Brain Observatory Data ===")

    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache

        manifest_file = str(ALLEN_DIR / "boc_manifest.json")
        boc = BrainObservatoryCache(manifest_file=manifest_file)

        # Get experiment containers for a specific area and Cre line
        containers = boc.get_experiment_containers(
            targeted_structures=['VISp'],  # Primary visual cortex
            cre_lines=['Cux2-CreERT2'],    # Layer 2/3 excitatory
        )

        if not containers:
            print("  No containers found, trying broader search...")
            containers = boc.get_experiment_containers(
                targeted_structures=['VISp']
            )

        if containers:
            container_id = containers[0]['id']
            print(f"  Using container: {container_id}")

            # Get experiments for this container
            exps = boc.get_ophys_experiments(
                experiment_container_ids=[container_id],
                stimuli=['natural_scenes']
            )

            if exps:
                exp_id = exps[0]['id']
                print(f"  Downloading experiment {exp_id}...")

                # Get the NWB data (contains traces)
                data_set = boc.get_ophys_experiment_data(exp_id)

                # Extract traces
                timestamps, traces = data_set.get_fluorescence_traces()
                _, dff = data_set.get_dff_traces()
                cell_ids = data_set.get_cell_specimen_ids()

                # Get ROI coordinates
                rois = data_set.get_roi_mask_array()

                import numpy as np

                # Save extracted traces
                np.savez(
                    str(ALLEN_DIR / f"allen_VISp_{exp_id}_traces.npz"),
                    timestamps=timestamps,
                    fluorescence=traces,
                    dff=dff,
                    cell_ids=cell_ids,
                    roi_masks=rois,
                    metadata={
                        'experiment_id': exp_id,
                        'container_id': container_id,
                        'structure': 'VISp',
                        'source': 'Allen Brain Observatory',
                    }
                )

                print(f"  Saved: {len(cell_ids)} cells, {traces.shape[1]} timepoints")
                print(f"  Sampling rate: {1.0/(timestamps[1]-timestamps[0]):.2f} Hz")
                return True

        print("  No suitable experiments found.")
        return False

    except ImportError:
        print("  AllenSDK not installed. Install with: pip install allensdk")
        print("  Then re-run this script.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# =========================================================================
# 4. Alternative: Download pre-processed calcium traces from DANDI archive
# =========================================================================
def download_dandi_data():
    """
    Download NWB files with extracted calcium traces from DANDI Archive.
    DANDI hosts standardized neurophysiology data in NWB format.
    """
    print("\n=== DANDI Archive Data ===")

    # DANDI dataset 000003: Allen Institute openscope calcium imaging
    # Small NWB files with extracted fluorescence traces
    dandi_api = "https://api.dandiarchive.org/api"

    try:
        # Get dandiset info
        response = urllib.request.urlopen(f"{dandi_api}/dandisets/000003/versions/draft/")
        info = json.loads(response.read())
        print(f"  Found: {info.get('name', 'unknown')}")

        # List assets (NWB files)
        response = urllib.request.urlopen(
            f"{dandi_api}/dandisets/000003/versions/draft/assets/?page_size=5"
        )
        assets = json.loads(response.read())

        for asset in assets.get('results', [])[:2]:
            fname = asset['path'].replace('/', '_')
            fsize = asset.get('size', 0) / (1024*1024)
            print(f"    Available: {fname} ({fsize:.1f} MB)")

            if fsize < 500:  # Only download files under 500 MB
                blob_url = asset.get('contentUrl', [''])[0] if asset.get('contentUrl') else None
                if not blob_url:
                    # Get download URL from asset detail
                    asset_url = f"{dandi_api}/dandisets/000003/versions/draft/assets/{asset['asset_id']}/download/"
                    download_file(asset_url, ALLEN_DIR / fname, f"{fname} ({fsize:.1f} MB)")

    except Exception as e:
        print(f"  Could not access DANDI: {e}")


# =========================================================================
# 5. Synthetic calcium traces (always works, for immediate testing)
# =========================================================================
def generate_synthetic_traces():
    """
    Generate realistic synthetic calcium traces using established models.
    Uses exponential calcium dynamics with Poisson spike trains.

    This gives us data to start analysis immediately while real data downloads.
    """
    print("\n=== Generating Synthetic Calcium Traces ===")

    import numpy as np
    from scipy.signal import lfilter

    np.random.seed(42)

    # Parameters matching GCaMP6f dynamics
    n_cells = 100
    duration_s = 300       # 5 minutes
    fs = 30.0              # 30 Hz (typical 2-photon)
    n_frames = int(duration_s * fs)

    tau_rise = 0.05        # GCaMP6f rise time constant (seconds)
    tau_decay = 0.4        # GCaMP6f decay time constant (seconds)
    noise_std = 0.1        # Measurement noise level

    dt = 1.0 / fs
    t = np.arange(n_frames) * dt

    # Calcium indicator kernel (double exponential)
    kernel_t = np.arange(0, 3.0, dt)  # 3 second kernel
    kernel = (np.exp(-kernel_t / tau_decay) - np.exp(-kernel_t / tau_rise))
    kernel = kernel / kernel.max()

    traces = np.zeros((n_cells, n_frames))
    spike_trains = np.zeros((n_cells, n_frames))

    # Different firing rates for different "cell types"
    base_rates = np.concatenate([
        np.random.uniform(0.5, 2.0, n_cells // 2),   # Low-rate cells
        np.random.uniform(2.0, 8.0, n_cells // 2),    # High-rate cells
    ])

    for i in range(n_cells):
        # Generate Poisson spike train
        rate = base_rates[i]
        spikes = np.random.poisson(rate * dt, n_frames)
        spike_trains[i] = spikes

        # Convolve with calcium kernel
        calcium = np.convolve(spikes, kernel)[:n_frames]

        # Add baseline and noise
        baseline = 1.0 + 0.1 * np.sin(2 * np.pi * t / 60)  # Slow drift
        noisy_trace = baseline + calcium + noise_std * np.random.randn(n_frames)

        traces[i] = noisy_trace

    # Compute dF/F
    f0 = np.percentile(traces, 10, axis=1, keepdims=True)
    dff = (traces - f0) / f0

    # Generate spatial coordinates (circular FOV, ~500 um)
    fov_radius = 250  # microns
    angles = np.random.uniform(0, 2 * np.pi, n_cells)
    radii = fov_radius * np.sqrt(np.random.uniform(0, 1, n_cells))
    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)

    # Save
    outpath = DATA_DIR / "caiman" / "synthetic_calcium_traces.npz"
    np.savez(
        str(outpath),
        traces=traces,
        dff=dff,
        spike_trains=spike_trains,
        timestamps=t,
        x_coords=x_coords,
        y_coords=y_coords,
        cell_ids=np.arange(n_cells),
        fs=fs,
        n_cells=n_cells,
        tau_rise=tau_rise,
        tau_decay=tau_decay,
        kernel=kernel,
        metadata=np.array(json.dumps({
            'type': 'synthetic',
            'model': 'Poisson spikes + GCaMP6f double-exponential kernel',
            'n_cells': n_cells,
            'duration_s': duration_s,
            'fs': fs,
            'tau_rise_s': tau_rise,
            'tau_decay_s': tau_decay,
            'noise_std': noise_std,
            'note': ('Synthetic data for testing asymmetry analysis pipeline. '
                     'Calcium transients have inherent rise/decay asymmetry. '
                     'Replace with real data from Allen Brain Observatory.'),
        }))
    )

    print(f"  Saved: {outpath.name}")
    print(f"    {n_cells} cells, {n_frames} frames ({duration_s}s at {fs} Hz)")
    print(f"    Traces shape: {traces.shape}")
    print(f"    dF/F shape: {dff.shape}")
    print(f"    Spatial coords: ({x_coords.min():.0f}, {y_coords.min():.0f}) to "
          f"({x_coords.max():.0f}, {y_coords.max():.0f}) um")

    return outpath


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Calcium Imaging Data Download Script")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")

    # Always generate synthetic data first (instant, no network needed)
    synth_path = generate_synthetic_traces()

    # Try real data sources
    download_caiman_data()
    download_neurofinder_data()
    download_allen_data()
    download_dandi_data()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for root, dirs, files in os.walk(str(DATA_DIR)):
        level = root.replace(str(DATA_DIR), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in sorted(files):
            fpath = Path(root) / file
            fsize = fpath.stat().st_size / (1024*1024)
            print(f"{subindent}{file} ({fsize:.1f} MB)")

    print(f"\nSynthetic data ready at: {synth_path}")
    print("\nFor real data, install AllenSDK:")
    print("  pip install allensdk")
    print("  python scripts/download_calcium_data.py")
