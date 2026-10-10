"""Download DS2OS and NSL-KDD from the Kaggle mirrors used in the paper.

Requires a Kaggle API token, either as ~/.kaggle/kaggle.json or through the
KAGGLE_USERNAME and KAGGLE_KEY environment variables.

Usage:
    python src/1_download_data.py

Author: Amitabh Chakravorty
"""

import shutil
import subprocess
import sys

from common import DATA_RAW, DS2OS_FILES, NSL_TEST_FILE, NSL_TRAIN_FILE, find_file

DATASETS = {
    "ds2os": "libamariyam/ds2os-dataset",
    "nsl_kdd": "hassan06/nslkdd",
}


def main():
    if shutil.which("kaggle") is None:
        sys.exit("The kaggle command is missing. Install it with: pip install kaggle")

    for name, slug in DATASETS.items():
        target = DATA_RAW / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {slug} into {target}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
            check=True,
        )

    print("\nFiles found:")
    for names in (DS2OS_FILES, NSL_TRAIN_FILE, NSL_TEST_FILE):
        print("  ", find_file(names))


if __name__ == "__main__":
    main()
