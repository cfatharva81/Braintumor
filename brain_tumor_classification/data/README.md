# Where to put your data

Place your images like this:

```
data/
├── yes/    <- MRI images WITH a tumor (~259 images)
└── no/     <- MRI images WITHOUT a tumor (~22 images)
```

Any common image format works (`.jpg`, `.jpeg`, `.png`). File names don't matter —
the loader (`src/data_utils.py`) walks each folder and uses the folder name as the label.

If your images currently live in a single flat folder with a separate labels CSV
instead of two subfolders, tell Claude and it will adapt `src/data_utils.py`
(`load_image_paths_labels`) to read that format instead of re-sorting your files.

Once the images are in place, open `notebooks/brain_tumor_classification.ipynb`
and run it top to bottom — Section 1 will re-scan this folder and print what it found
before anything else happens.
