"""Turning faces into embedding vectors with DeepFace.

Compute and cache LFW embeddings from the project root:
    python -m src.embed                    # all models in config.EMBEDDING_MODELS
    python -m src.embed --models ArcFace
"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info messages

import numpy as np
from deepface import DeepFace

from .config import DATA_DIR, EMBEDDING_MODELS, LFW_DETECTOR, LIVE_DETECTOR
from .data import load_lfw, to_bgr_uint8
from .matching import l2_normalize


def embed_faces(faces_bgr, model_name, detector_backend=LFW_DETECTOR, batch_size=64, verbose=False):
    """Embed face crops (uint8 BGR array, one face per image). Returns L2-normalised (N, d)."""
    out, t0 = [], time.time()
    for start in range(0, len(faces_bgr), batch_size):
        batch = faces_bgr[start:start + batch_size]
        reps = DeepFace.represent(batch, model_name=model_name, detector_backend=detector_backend,
                                  enforce_detection=False)
        if reps and isinstance(reps[0], dict):  # a batch of one comes back un-nested
            reps = [reps]
        out.extend(r[0]["embedding"] for r in reps)
        if verbose:
            done = start + len(batch)
            print(f"  {model_name}: {done}/{len(faces_bgr)} ({time.time() - t0:.0f} s)", flush=True)
    return l2_normalize(out)


def detect_and_embed(img, model_name, detector_backend=LIVE_DETECTOR):
    """Find every face in a full photo or webcam frame (path or BGR array).

    Returns (embeddings, boxes) with boxes as (x, y, w, h); both empty if no face is found.
    """
    reps = DeepFace.represent(img, model_name=model_name, detector_backend=detector_backend,
                              enforce_detection=False)
    reps = [r for r in reps if r.get("face_confidence", 0) > 0]  # 0 = no face, whole image returned
    if not reps:
        return np.empty((0, 0), dtype=np.float32), []
    boxes = [tuple(r["facial_area"][k] for k in ("x", "y", "w", "h")) for r in reps]
    return l2_normalize([r["embedding"] for r in reps]), boxes


def embedding_file(model_name):
    return DATA_DIR / f"embeddings_{model_name}.npy"


def get_embeddings(model_name, recompute=False):
    """Cached LFW embeddings for one model; computed first if missing (~5-10 min on CPU)."""
    f = embedding_file(model_name)
    if f.exists() and not recompute:
        return np.load(f)
    images, _, _ = load_lfw()
    E = embed_faces(to_bgr_uint8(images), model_name, verbose=True)
    np.save(f, E)
    return E


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and cache LFW embeddings.")
    parser.add_argument("--models", nargs="+", default=EMBEDDING_MODELS)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    for m in args.models:
        E = get_embeddings(m, recompute=args.recompute)
        print(f"{m}: {E.shape} -> {embedding_file(m).name}")
