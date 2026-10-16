"""Live face recognition from the webcam. Press q to quit.

Run from the project root, after `python -m src.enroll`:
    python -m src.webcam
    python -m src.webcam --threshold 0.4    # stricter: fewer false names, more 'Unknown'
"""
import argparse
import os

import cv2

from .matching import GALLERY_FILE, identify, load_gallery

GREEN, RED = (0, 200, 0), (0, 0, 255)


def recognise(frame, gallery, threshold):
    """[(name, similarity, (x, y, w, h)), ...] for every face in a BGR frame."""
    from .embed import detect_and_embed

    names, centroids, model, _ = gallery
    E, boxes = detect_and_embed(frame, model)
    if not boxes:
        return []
    labels, scores = identify(E, names, centroids, threshold, unknown="Unknown")
    return list(zip(labels, scores, boxes))


def draw(frame, results):
    for name, score, (x, y, w, h) in results:
        colour = RED if name == "Unknown" else GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, f"{name} ({score:.2f})", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Live face recognition from the webcam.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="minimum cosine similarity to name someone (default: from gallery)")
    parser.add_argument("--every", type=int, default=3,
                        help="recognise every Nth frame; higher = smoother video on slow CPUs")
    args = parser.parse_args()

    if not GALLERY_FILE.exists():
        raise SystemExit("No gallery yet. Add photos to data/my_faces/<name>/ and run "
                         "`python -m src.enroll` first.")
    gallery = load_gallery()
    threshold = args.threshold if args.threshold is not None else gallery[3]
    print(f"{len(gallery[0])} people in gallery, model {gallery[2]}, threshold {threshold:.3f}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")
    results, frame_no = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % args.every == 0:
            results = recognise(frame, gallery, threshold)
        frame_no += 1
        cv2.imshow("Face recognition (q to quit)", draw(frame, results))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
