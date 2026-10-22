"""Select the pretraining images from the brainmarks-smri FOMO300K tables.

uv run python scripts/select_images.py data/fomo300k src/nanobrain/config/fomo300k_images.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANAT_SUFFIXES = ("T1w", "T2w", "FLAIR")
MIN_BRAIN_EXTENT = 100.0  # mm, on every axis
DWI_SHELL_RANGE = (900, 1100)


def main(args: argparse.Namespace):
    images = pd.read_parquet(args.root / "tables" / "images.parquet")
    print(f"all images: {len(images)}")

    # one sequence label per image, None for sequences we don't use
    bval = pd.to_numeric(images["suffix"].str.extract(r"^dwi_bval(\d+)$")[0])
    is_shell = (images["dwi_type"] == "mean") & bval.between(*DWI_SHELL_RANGE)
    sequence = pd.Series(None, index=images.index, dtype=object)
    sequence[images["suffix"].isin(ANAT_SUFFIXES)] = images["suffix"]
    sequence[images["dwi_type"] == "b0"] = "dwi_b0"
    sequence[is_shell] = "dwi_b1000"
    images["sequence"] = sequence
    images = images[images["sequence"].notna()]
    print(f"sequences {sorted(images['sequence'].unique())}: {len(images)}")

    images = images[images["warnings"].map(len) == 0]
    print(f"no qc warnings: {len(images)}")

    brain_extent = np.array(images["mask_extent"].tolist())
    images = images[brain_extent.min(axis=1) >= MIN_BRAIN_EXTENT]
    print(f"brain extent >= {MIN_BRAIN_EXTENT:.0f} mm: {len(images)}")

    # first image by path per (subject, sequence); subject ids repeat across collections
    images = images.sort_values("path")
    images = images.drop_duplicates(["collection", "subject", "sequence"])
    print(f"one per (subject, sequence): {len(images)}")
    print(images["sequence"].value_counts().to_string())

    paths = "derivatives/" + images["path"].str.removeprefix("source/").str.removesuffix(".nii.gz")
    paths = paths + ".npz"
    args.out.write_text("\n".join(paths) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="brainmarks-smri fomo300k folder")
    parser.add_argument("out", type=Path, help="output filelist of npz paths")
    args = parser.parse_args()
    main(args)
