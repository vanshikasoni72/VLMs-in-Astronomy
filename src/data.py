"""Loading Galaxy Zoo 2 images and labels from the Kaggle
"Galaxy Zoo - The Galaxy Challenge" dataset.
"""
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("/kaggle/input/galaxy-zoo-the-galaxy-challenge")
CACHE_DIR = Path("/kaggle/working/galaxy_zoo_cache")


def load_solutions(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "training_solutions_rev1.csv")


def _ensure_images_extracted(data_dir: Path) -> Path:
    """Return a directory containing the training images, extracting the
    zip into a writable cache dir first if Kaggle hasn't unzipped it already.
    """
    extracted = data_dir / "images_training_rev1"
    if extracted.exists():
        return extracted

    cached = CACHE_DIR / "images_training_rev1"
    if cached.exists():
        return cached

    zip_path = data_dir / "images_training_rev1.zip"
    if zip_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CACHE_DIR)
        return cached

    raise FileNotFoundError(
        f"Could not find images_training_rev1/ or .zip under {data_dir}. "
        f"Check the actual mounted dataset path (e.g. list /kaggle/input) "
        f"and pass --data-dir if it differs."
    )


def get_test_subset(n: int = 5, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Deterministic smoke-test subset: the first n galaxies by GalaxyID.

    This is only for checking the pipeline runs end-to-end on real images —
    it is NOT the odd/ordinary pilot subset from the project's research
    decisions. That selection (250 flagged-odd + 250 ordinary, with label
    columns/thresholds still undecided) is a separate, later step.
    """
    solutions = load_solutions(data_dir)
    images_dir = _ensure_images_extracted(data_dir)
    subset = solutions.sort_values("GalaxyID").head(n).copy()
    subset["image_path"] = subset["GalaxyID"].apply(lambda gid: images_dir / f"{gid}.jpg")
    return subset
