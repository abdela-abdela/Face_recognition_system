import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path

import cv2, joblib
import numpy as np
from deepface import DeepFace

HERE = Path(__file__).parent

#Load the known faces saved by face_recognition.py
names = joblib.load(HERE / "face_svm.joblib")["names"]
E = np.load(HERE / "embeddings_arcface.npy")
y = np.load(HERE / "labels_arcface.npy")
E = E / np.linalg.norm(E, axis=1, keepdims=True)

# Cosine similarity to the closest known face. On the LFW test set correct
# matches score ~0.75 and wrong ones ~0.40 - tune this if needed.
SIM_THRESHOLD = 0.45

#Live webcam demo - press q to quit
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    for f in DeepFace.represent(frame, model_name="ArcFace", detector_backend="opencv",
                                enforce_detection=False):
        a = f["facial_area"]
        if a["w"] >= w and a["h"] >= h:   # no face found - DeepFace returned the whole frame
            continue
        emb = np.array(f["embedding"])
        sims = E @ (emb / np.linalg.norm(emb))
        best = sims.argmax()
        person = names[y[best]]
        if sims[best] >= SIM_THRESHOLD:
            label, colour = f"{person} {sims[best]:.2f}", (0, 255, 0)
        else:
            label, colour = f"Unknown ({person} {sims[best]:.2f})", (0, 0, 255)
        cv2.rectangle(frame, (a["x"], a["y"]), (a["x"]+a["w"], a["y"]+a["h"]), colour, 2)
        cv2.putText(frame, label, (a["x"], a["y"]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.imshow("Face recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release(); cv2.destroyAllWindows()
