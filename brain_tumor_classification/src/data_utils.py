"""
Dataset inspection and loading utilities.

Kept separate from the notebook so the same functions can be reused for
training scripts, unit tests, or a future re-run without copy/pasting cells.
"""
import os
from collections import Counter
from dataclasses import dataclass

from PIL import Image

CLASS_NAMES = ["no", "yes"]  # index 0 = no tumor, index 1 = tumor present
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    path: str
    label: str          # "yes" or "no"
    width: int
    height: int
    mode: str            # PIL mode, e.g. "RGB", "L", "RGBA"
    format: str           # e.g. "JPEG", "PNG"


def find_dataset_root(data_dir: str) -> str:
    """
    Locate the folder that directly contains 'yes' and 'no' subfolders.

    Some dataset zips extract with an extra nesting level
    (e.g. data/brain_tumor_dataset/yes). This walks a couple of levels down
    so the rest of the pipeline doesn't care about that quirk.
    """
    candidates = [data_dir]
    for root, dirs, _ in os.walk(data_dir):
        lower_dirs = {d.lower() for d in dirs}
        if {"yes", "no"}.issubset(lower_dirs):
            candidates.append(root)
    for c in candidates:
        entries = {d.lower() for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))}
        if {"yes", "no"}.issubset(entries):
            return c
    raise FileNotFoundError(
        f"Could not find 'yes' and 'no' subfolders anywhere under {data_dir}. "
        "See data/README.md for the expected layout."
    )


def load_image_paths_labels(data_dir: str):
    """
    Walk data_dir/yes and data_dir/no and return parallel lists of
    (file paths, string labels). Non-image files are skipped.
    """
    root = find_dataset_root(data_dir)
    paths, labels = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}/yes or {root}/no.")
    return paths, labels


def inspect_dataset(data_dir: str):
    """
    Read every image's basic properties (size, color mode, format) without
    fully decoding pixel data (PIL is lazy until .load()/.convert()).

    Returns a list[ImageRecord] the notebook uses for EDA tables/plots.
    """
    paths, labels = load_image_paths_labels(data_dir)
    records = []
    for path, label in zip(paths, labels):
        with Image.open(path) as img:
            records.append(
                ImageRecord(
                    path=path,
                    label=label,
                    width=img.width,
                    height=img.height,
                    mode=img.mode,
                    format=img.format,
                )
            )
    return records


def summarize_class_counts(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: {"count": n, "pct": round(100 * n / total, 1)} for label, n in counts.items()}
