"""Loading LFW and splitting it the same way everywhere."""
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, MIN_FACES, SEED, TEST_SIZE

LFW_FILE = DATA_DIR / "lfw.npz"


def load_lfw():
    """Return (images, y, names). Images are float RGB in [0, 1], shape (N, 125, 94, 3).

    The first call downloads LFW (~200 MB) and caches it in data/lfw.npz.
    """
    if LFW_FILE.exists():
        d = np.load(LFW_FILE)
        return d["images"], d["y"], d["names"]
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES, resize=1.0, color=True)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(LFW_FILE, images=lfw.images, y=lfw.target, names=lfw.target_names)
    return lfw.images, lfw.target, lfw.target_names


def to_bgr_uint8(images):
    """Float RGB in [0, 1] -> uint8 BGR, the format OpenCV and DeepFace expect."""
    return np.ascontiguousarray((np.clip(images, 0, 1) * 255).astype(np.uint8)[..., ::-1])


def split(y, test_size=TEST_SIZE, seed=SEED):
    """Stratified (train_idx, test_idx). The fixed seed gives the same split in every notebook."""
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)


def open_set_split(y, n_unknown=12, seed=SEED):
    """Pick n_unknown identities to hide from the gallery. Returns (known_ids, unknown_ids)."""
    ids = np.unique(y)
    unknown = np.random.default_rng(seed).choice(ids, size=n_unknown, replace=False)
    return np.setdiff1d(ids, unknown), np.sort(unknown)
