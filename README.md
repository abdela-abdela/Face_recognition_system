# Face recognition with deep embeddings

A data science project comparing classic **Eigenfaces** with pretrained deep face embeddings
(**ArcFace**, **Facenet512**) on *Labeled Faces in the Wild* (LFW), plus a live webcam demo.

**Research question:** how much do pretrained deep embeddings improve over Eigenfaces, how many
photos per person do they need, and how should the "unknown person" threshold be chosen?

## Pipeline

```
image → face crop → embedding (512 numbers) → compare with known people → name or "Unknown"
```

## Setup

```bash
conda create -n faces python=3.11 -y
conda activate faces
pip install -r requirements.txt
```

In VS Code, select the `faces` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*) and
pick **faces** as the kernel in each notebook.

## How to run

| Step | Command / notebook | Time (CPU) |
|---|---|---|
| 1. Explore the data (downloads LFW the first time) | `notebooks/01_eda.ipynb` | ~1 min |
| 2. Compute embeddings | `python -m src.embed` or `notebooks/02_embeddings.ipynb` | ~8 min the first time |
| 3. Benchmark + pick the demo threshold | `python -m src.train` | ~30 s |
| 4. Compare models | `notebooks/03_models.ipynb` | ~15 s |
| 5. Verification, unknowns, errors | `notebooks/04_evaluation.ipynb` | ~30 s |
| 6. Live demo | see below | |

Run the `python -m ...` commands from this folder (the project root).

### Live webcam demo

1. Put photos in `data/my_faces/<name>/`, for example `data/my_faces/Abdel/1.jpg`, ideally 5–10
   photos per person with one clear face each. **Only use photos of people who have agreed.**
2. `python -m src.enroll` builds `models/gallery.npz` (one average embedding per person).
3. `python -m src.webcam` opens the camera. Press **q** to quit.
   - Add `--threshold 0.6` to be stricter if strangers get named, or use a lower value if you are
     often "Unknown".

## Results (test set: 756 images of 62 people)

**Identification (who is it?)**

| Features | SVM | 1-NN | Nearest centroid |
|---|---|---|---|
| Eigenfaces | 0.442 | 0.460 | 0.473 |
| ArcFace | 0.990 | 0.988 | 0.992 |
| Facenet512 | **0.997** | 0.985 | 0.993 |

*Macro F1. Eigenfaces stays at 0.39–0.49 across PCA sizes (50/100/150) with and without
whitening, so the gap is not a tuning artefact.*

**Verification (same person?) on all 285,390 test pairs**

| Features | AUC | EER | TAR @ FAR = 1% |
|---|---|---|---|
| Eigenfaces | 0.576 | 44.9% | 5.3% |
| ArcFace | 0.995 | 2.5% | 95.8% |
| Facenet512 | 0.998 | 2.0% | 96.4% |

**Few photos per person** (nearest centroid, mean of 5 draws): with **1 photo** per person, ArcFace
reaches 92.8% accuracy and Facenet512 93.3%. With 3 photos both reach ~99%. Eigenfaces reaches 32%
with 10 photos.

**Unknown people (open set, ArcFace):** 12 people are hidden from the gallery as strangers.

| Threshold chosen on training data | Known named correctly | Known called "Unknown" | Strangers given a name |
|---|---|---|---|
| From photo pairs (0.35) | 99.2% | 0.8% | **54.3%** |
| From gallery matches (0.52) | 95.5% | 4.5% | **1.8%** |

Similarity to a person's *average* face is higher than to a single photo, for strangers too. A
threshold tuned on photo pairs therefore names over half the strangers. Tuning it on the
comparison the system actually makes fixes this. The demo uses the gallery threshold.

## Project layout

```
data/            LFW cache, embeddings, my_faces/ (git-ignored)
models/          threshold.json, gallery.npz (git-ignored)
notebooks/       01_eda → 02_embeddings → 03_models → 04_evaluation
reports/         identification.csv, verification.csv (written by src.train)
src/
  config.py      paths and settings
  data.py        loading LFW, train/test and open-set splits
  embed.py       DeepFace embeddings (cached)
  matching.py    galleries, cosine matching, "Unknown"
  metrics.py     verification and open-set metrics
  train.py       benchmark + threshold selection
  enroll.py      build the demo gallery from data/my_faces
  webcam.py      live recognition
```

## Design choices

- **No face detector for LFW:** the images are already cropped around the face, and RetinaFace
  takes ~18 s per image on CPU. The webcam uses OpenCV's fast detector.
- **Pretrained models, not fine-tuned:** with ~3,000 images, training a face network from
  scratch would be far worse. The project evaluates *how to use* the embeddings.
- **Fixed seed (42)** for every split, so all notebooks and scripts see the same train/test data.

## Limitations and ethics

- LFW is dominated by male, light-skinned public figures from the 2000s and has no demographic
  labels, so these numbers say nothing about fairness across groups (see e.g. the RFW dataset).
- LFW photos are mostly frontal press photos, so a webcam in poor light is harder.
- Face embeddings are **biometric data** under GDPR (a special category): only enrol people with
  explicit consent, and delete their data on request.
