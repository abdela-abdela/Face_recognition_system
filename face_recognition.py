import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path

import numpy as np
from deepface import DeepFace

HERE = Path(__file__).parent
from sklearn.datasets import fetch_lfw_people

#Extracting the data
lfw =fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
X_imgs, y, names = lfw.images, lfw.target, lfw.target_names
print(X_imgs.shape, len(names))  # Around 3000 images of 60 people

#Detection, alignment and embeddings
def embed(img, model="ArcFace"):
    # LFW images are float 0–1 RGB; DeepFace expects 0–255 BGR
    img = (img * 255).astype("uint8")[:, :, ::-1] if img.max() <= 1 else img
    rep = DeepFace.represent(img, model_name=model,
                             detector_backend="retinaface",
                             enforce_detection=False, align=True)
    return np.array(rep[0]["embedding"])

emb_path = HERE / "embeddings_arcface.npy"
if emb_path.exists():
    E = np.load(emb_path)
else:
    # slow: only runs the first time
    E = []
    for i, im in enumerate(X_imgs, 1):
        E.append(embed(im))
        if i % 100 == 0 or i == len(X_imgs):
            print(f"Embedded {i}/{len(X_imgs)} images", flush=True)
    E = np.vstack(E)
    np.save(emb_path, E)
np.save(HERE / "labels_arcface.npy", y)   # used by webcam_demo.py

#Building the models
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X_flat = X_imgs.reshape(len(X_imgs), -1)
Xtr, Xte, ytr, yte = train_test_split(X_flat, y, test_size=0.25, stratify=y, random_state=42)
baseline = make_pipeline(PCA(n_components=150, whiten=True), SVC(class_weight="balanced"))
baseline.fit(Xtr, ytr)

# Embeddings + a simple classifier
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

Etr, Ete, ytr, yte = train_test_split(E, y, test_size=0.25, stratify=y, random_state=42)

svm = make_pipeline(Normalizer(), SVC(kernel="linear", probability=True, class_weight="balanced"))
knn = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=3, metric="cosine"))
svm.fit(Etr, ytr); knn.fit(Etr, ytr)

# Evaluation of models
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc

print(classification_report(yte, svm.predict(Ete), target_names=names))
ConfusionMatrixDisplay.from_estimator(svm, Ete, yte)

from sklearn.metrics.pairwise import cosine_similarity
import itertools, random

S = cosine_similarity(Ete)
pairs = list(itertools.combinations(range(len(yte)), 2))
random.seed(0); pairs = random.sample(pairs, 20000)
scores = [S[i, j] for i, j in pairs]
labels = [int(yte[i] == yte[j]) for i, j in pairs]

fpr, tpr, thr = roc_curve(labels, scores)
eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
print(f"AUC={auc(fpr,tpr):.3f}  EER={fpr[eer_idx]:.3f}  threshold={thr[eer_idx]:.3f}")

#Save the model for the live webcam demo (run webcam_demo.py)
import joblib
joblib.dump({"model": svm, "names": names}, HERE / "face_svm.joblib")
print("Saved face_svm.joblib - run webcam_demo.py for the live demo")