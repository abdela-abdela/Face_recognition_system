"""Paths and settings shared by every script and notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MY_FACES_DIR = DATA_DIR / "my_faces"   # one sub-folder per (consenting) person, for the live demo

SEED = 42
MIN_FACES = 20      # keep LFW people with at least this many images
TEST_SIZE = 0.25

EMBEDDING_MODELS = ["ArcFace", "Facenet512"]
DEFAULT_MODEL = "ArcFace"

LFW_DETECTOR = "skip"     # LFW images are already cropped around the face
LIVE_DETECTOR = "opencv"  # fast enough for a webcam on CPU (RetinaFace is ~18 s/image on CPU)

# Cosine-similarity thresholds DeepFace recommends, used only if `python -m src.train`
# hasn't been run yet to pick a threshold from the data.
FALLBACK_THRESHOLDS = {"ArcFace": 0.32, "Facenet512": 0.70}
