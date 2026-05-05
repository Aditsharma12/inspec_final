import os
import torch
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
import torchvision.transforms.functional as F


# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_env(Path(__file__).resolve().parent / ".env")
# ─────────────────────────────────────────────────────────────────────────────

from model.cvit import CViT
from model.pred_func import load_cvit

# Load model once at startup
MODEL_NAME = "cvit2"
WEIGHT     = "cvit2_deepfake_detection_ep_50.pth"

device = torch.device("cpu")

model = load_cvit(WEIGHT, MODEL_NAME, fp16=False)
model.eval()

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)


def extract_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def preprocess_faces(frames):
    faces = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
                face = rgb[y1:y2, x1:x2]
                if face.size > 0:
                    face = cv2.resize(face, (224, 224))
                    faces.append(face)

    return faces


def predict_single(video_path):
    frames = extract_frames(video_path)
    faces = preprocess_faces(frames)

    if len(faces) == 0:
        return {
            "prediction": "NO_FACE",
            "confidence": 0.0
        }

    # Convert to tensor
    faces = np.array(faces) / 255.0
    faces = np.transpose(faces, (0, 3, 1, 2))  # NHWC → NCHW
    faces = torch.tensor(faces, dtype=torch.float32)
    faces = F.normalize(faces, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        outputs = model(faces)
        mean_logits = outputs.mean(dim=0)
        probs = torch.softmax(mean_logits, dim=0)

        class_idx = torch.argmax(probs).item()
        prediction = "FAKE" if class_idx == 0 else "REAL"
        confidence = probs[class_idx].item()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }