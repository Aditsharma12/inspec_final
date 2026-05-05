import os
import io as _io
import cv2
import torch

# ── CPU-only patch ────────────────────────────────────────────────────────────
# The .pkl checkpoint was saved on a GPU machine. PyTorch's internal
# _load_from_bytes re-calls torch.load() without map_location, which crashes
# on CPU-only systems. Patch it to always redirect to CPU.
def _cpu_load_from_bytes(b):
    return torch.load(_io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)
torch.storage._load_from_bytes = _cpu_load_from_bytes
# ─────────────────────────────────────────────────────────────────────────────
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)


# ── HuggingFace Hub weight download ──────────────────────────────────────────
#
#  Upload your weight once with:
#
#    pip install huggingface_hub
#    huggingface-cli login
#    huggingface-cli upload <YOUR_HF_USERNAME>/cvit-deepfake-weights \
#        weight/cvit2_deepfake_detection_ep_50.pth \
#        cvit2_deepfake_detection_ep_50.pth
#
#  Then set HF_WEIGHT_REPO in your .env (or export it as an env var):
#    HF_WEIGHT_REPO=<YOUR_HF_USERNAME>/cvit-deepfake-weights
#
# If HF_WEIGHT_REPO is not set we fall back to checking the local weight/ dir.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_weight_path(filename: str) -> str:
    """
    Returns the local path to the weight file.

    Priority:
      1. HF_WEIGHT_REPO env var  → download (or use cache) from HF Hub
      2. Local weight/ directory → use as-is (legacy / dev mode)
    """
    hf_repo = os.environ.get("HF_WEIGHT_REPO", "").strip()

    if hf_repo:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN") or None
        print(f"[cvit] Fetching '{filename}' from HF Hub repo '{hf_repo}'…")
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            token=token,
        )
        print(f"[cvit] Weight ready at: {local_path}")
        return local_path

    # Fallback – local weight directory (not committed to git)
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weight",
        filename,
    )
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Weight file not found locally at '{local_path}' and "
            "HF_WEIGHT_REPO is not set. "
            "Either upload the weight to Hugging Face and set HF_WEIGHT_REPO, "
            "or place the .pth file in the backend_vid/weight/ directory."
        )
    print(f"[cvit] Using local weight: {local_path}")
    return local_path


def load_cvit(cvit_weight, net, fp16):

    from model.cvit import CViT as cvit  # load cvit2
    if net == 'cvit':
        from model.cvit_old import CViT as cvit

    model = cvit(image_size=224, patch_size=7, num_classes=2, channels=512,
             dim=1024, depth=6, heads=8, mlp_dim=2048)

    model.to(device)

    weight_path = _resolve_weight_path(cvit_weight)
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    _ = model.eval()

    if fp16:
        model.half()

    return model

def face_mtcnn_(frame):
    boxes, con = mtcnn.detect(frame)
    return boxes is not None and con[0]>0.95


def face_mtcnn(frames):
    padding = 0
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0

    for _, frame in tqdm(enumerate(frames), total=len(frames)):

        try:
            boxes, conf = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    if count < 5:
                        # Extract coordinates
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)

                        # Crop the face from the frame
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            resized_face = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
                            resized_face_bgr = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                            temp_face[count] = resized_face_bgr
                            count += 1
        except:
            print('error encountered when extracting video frames')

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    mean_val = torch.mean(y_pred, dim=0)

    if mean_val.numel() == 1:
        mean_val = y_pred

    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
