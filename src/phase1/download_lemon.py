"""
Download LEMON dataset.

LEMON (Leipzig Study for Mind-Body-Emotion Interactions)
- Paper: Babayan et al., Scientific Data, 2019 (doi:10.1038/sdata.2018.308)
- ~227 subjects (153 young 20-35, 74 older 59-77)
- 62-channel EEG (BrainVision format, 2500 Hz) + ECG channel
- Resting state: eyes closed (EC) and eyes open (EO)

Data access options:
1. OpenNeuro: openneuro download --dataset ds000221
2. DataLad: datalad install https://github.com/OpenNeuroDatasets/ds000221.git
3. AWS S3: s3://fcp-indi/data/Projects/INDI/MPI-LEMON/
4. NITRC: https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html

Usage:
    python -m src.phase1.download_lemon --method datalad --output data/lemon/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_via_datalad(output_dir, n_subjects=None):
    """Download LEMON via DataLad (recommended)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Install dataset (metadata only, no bulk download)
    dataset_url = "https://github.com/OpenNeuroDatasets/ds000221.git"
    print(f"Installing LEMON dataset from {dataset_url}...")
    subprocess.run(
        ["datalad", "install", "-s", dataset_url, str(output_dir)],
        check=True,
    )

    if n_subjects is not None:
        # Get list of subjects
        subjects = sorted([
            d.name for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])[:n_subjects]

        for sub in subjects:
            print(f"Downloading {sub}...")
            subprocess.run(
                ["datalad", "get", str(output_dir / sub)],
                check=True,
            )
    else:
        print("Downloading all data (this may take a while)...")
        subprocess.run(
            ["datalad", "get", str(output_dir)],
            check=True,
        )


def download_via_openneuro(output_dir, n_subjects=None):
    """Download LEMON via openneuro-py CLI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "openneuro-py", "download",
        "--dataset", "ds000221",
        "--target-dir", str(output_dir),
    ]

    if n_subjects is not None:
        # openneuro-py supports --include for file patterns
        # Get first N subjects
        includes = [f"sub-{str(i).zfill(6)}" for i in range(1, n_subjects + 1)]
        for inc in includes:
            cmd.extend(["--include", inc])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_via_aws(output_dir, n_subjects=None):
    """Download LEMON EEG via AWS S3 (no credentials needed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3_path = "s3://fcp-indi/data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/"

    cmd = [
        "aws", "s3", "sync",
        s3_path, str(output_dir),
        "--no-sign-request",
    ]

    if n_subjects is not None:
        # Download subject by subject
        for i in range(1, n_subjects + 1):
            sub_id = f"sub-{str(i).zfill(6)}"
            sub_cmd = [
                "aws", "s3", "sync",
                f"{s3_path}{sub_id}/", str(output_dir / sub_id) + "/",
                "--no-sign-request",
            ]
            print(f"Downloading {sub_id}...")
            subprocess.run(sub_cmd, check=True)
    else:
        print(f"Syncing from {s3_path}...")
        subprocess.run(cmd, check=True)


def verify_download(data_dir):
    """Verify downloaded LEMON data structure."""
    data_dir = Path(data_dir)

    # Check for participants file
    participants = data_dir / "participants.tsv"
    if participants.exists():
        import pandas as pd
        df = pd.read_csv(participants, sep="\t")
        print(f"Found participants.tsv: {len(df)} subjects")
        print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
        print(f"  Sex: {df['sex'].value_counts().to_dict()}")
    else:
        print("WARNING: participants.tsv not found")

    # Count EEG files
    vhdr_files = list(data_dir.rglob("*.vhdr"))
    print(f"Found {len(vhdr_files)} EEG recordings (.vhdr files)")

    # Check structure
    subjects = sorted([d.name for d in data_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sub-")])
    print(f"Found {len(subjects)} subject directories")
    if subjects:
        print(f"  First: {subjects[0]}, Last: {subjects[-1]}")

    return len(vhdr_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Download LEMON dataset")
    parser.add_argument("--method", default="datalad",
                        choices=["datalad", "openneuro", "aws"],
                        help="Download method")
    parser.add_argument("--output", default="data/lemon/",
                        help="Output directory")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Download only first N subjects (for testing)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing download")
    args = parser.parse_args()

    if args.verify_only:
        verify_download(args.output)
        return

    methods = {
        "datalad": download_via_datalad,
        "openneuro": download_via_openneuro,
        "aws": download_via_aws,
    }

    try:
        methods[args.method](args.output, args.n_subjects)
        print("\nVerifying download...")
        verify_download(args.output)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Tool not found: {e}. Install required tool first.", file=sys.stderr)
        if args.method == "datalad":
            print("  pip install datalad", file=sys.stderr)
        elif args.method == "openneuro":
            print("  pip install openneuro-py", file=sys.stderr)
        elif args.method == "aws":
            print("  pip install awscli", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
