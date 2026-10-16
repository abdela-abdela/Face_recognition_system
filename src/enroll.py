"""Enrol people for the live demo: build a gallery from data/my_faces/<name>/*.jpg.

Only use photos of people who have agreed to it: faces are biometric data under GDPR.
Run from the project root:
    python -m src.enroll
"""
import argparse
from pathlib import Path

import numpy as np

from .config import MY_FACES_DIR
from .matching import GALLERY_FILE, build_gallery, load_threshold, save_gallery

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Build the face gallery for the webcam demo.")
    parser.add_argument("--faces-dir", type=Path, default=MY_FACES_DIR,
                        help="folder with one sub-folder of photos per person")
    parser.add_argument("--model", default=None, help="embedding model (default: the tuned one)")
    args = parser.parse_args()

    if not args.faces_dir.is_dir():
        raise SystemExit(f"Create {args.faces_dir} with one sub-folder per person, e.g. "
                         f"{args.faces_dir / 'Alice'}/photo1.jpg, then run this again.")
    model, threshold = load_threshold(args.model)

    from .embed import detect_and_embed  # imports TensorFlow, so only after the checks above

    names, embeddings = [], []
    for person_dir in sorted(d for d in args.faces_dir.iterdir() if d.is_dir()):
        n = 0
        for f in sorted(person_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            E, _ = detect_and_embed(str(f), model)
            if len(E) != 1:
                print(f"  skipped {person_dir.name}/{f.name}: found {len(E)} faces, need exactly 1")
                continue
            names.append(person_dir.name)
            embeddings.append(E[0])
            n += 1
        print(f"{person_dir.name}: {n} photo(s) enrolled")

    if not embeddings:
        raise SystemExit("No faces enrolled. Check that the photos show one clear face each.")
    ids, centroids = build_gallery(np.stack(embeddings), np.array(names))
    save_gallery(ids, centroids, model, threshold)
    print(f"Saved {len(ids)} people to {GALLERY_FILE.name} (model {model}, threshold {threshold:.3f})")


if __name__ == "__main__":
    main()
