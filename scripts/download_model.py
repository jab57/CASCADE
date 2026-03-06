#!/usr/bin/env python3
"""
Download the GREmLN model checkpoint required by CASCADE.

The model checkpoint (~120 MB) is not stored in git due to its size.
It is downloaded from the GREmLN Quickstart Tutorial Google Drive folder
provided by the Chan Zuckerberg Initiative / CZ Biohub NY.

Source: https://virtualcellmodels.cziscience.com/quickstart/gremln-quickstart
Publication: Zhang et al. (2025), bioRxiv 2025.07.03.663009

Usage:
    python scripts/download_model.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

# Google Drive folder ID for the GREmLN Quickstart Tutorial package.
# Contains: model.ckpt, network TSVs, and H5AD expression files.
GREMLN_FOLDER_ID = "1cMR9HoAC22i6sKSWgfQUEQRf0UP_w3_m"
GREMLN_FOLDER_URL = f"https://drive.google.com/drive/folders/{GREMLN_FOLDER_ID}?usp=sharing"

DEST = Path(__file__).parent.parent / "models" / "model.ckpt"


def main():
    if DEST.exists():
        print(f"model.ckpt already exists at {DEST}. Nothing to do.")
        print("Delete it and re-run this script to force a fresh download.")
        return

    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Install it with:")
        print("    pip install gdown")
        sys.exit(1)

    DEST.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading GREmLN tutorial folder from Google Drive...")
    print(f"Source: {GREMLN_FOLDER_URL}")
    print("(This downloads ~120 MB for model.ckpt plus associated tutorial files.)")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # gdown downloads the folder contents into a subfolder named after the Drive folder
        gdown.download_folder(
            id=GREMLN_FOLDER_ID,
            output=str(tmp_path),
            quiet=False,
        )

        # Locate model.ckpt anywhere in the downloaded tree
        candidates = list(tmp_path.rglob("model.ckpt"))
        if not candidates:
            print()
            print("ERROR: model.ckpt not found in the downloaded folder.")
            print("The Google Drive folder structure may have changed.")
            print(f"Please download manually from: {GREMLN_FOLDER_URL}")
            print(f"and place model.ckpt at: {DEST}")
            sys.exit(1)

        src = candidates[0]
        print(f"\nCopying {src.name} to {DEST} ...")
        shutil.copy2(src, DEST)

    print(f"\nDone. model.ckpt saved to: {DEST}")
    print()
    print("You can verify the installation with:")
    print("    python verify_installation.py --offline")


if __name__ == "__main__":
    main()
