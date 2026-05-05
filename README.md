# InSpec AI

> **Forensic Deepfake Analysis** — A panel of ten neural networks examines your image or video independently and renders a forensic verdict.

InSpec AI is a full-stack deepfake detection platform with:
- 🖼️ **Image mode** — 10 HuggingFace ViT/SigLIP models run in ensemble (`backend/` on port 8000)
- 🎬 **Video mode** — CViT-based face-frame deepfake detector (`backend_vid/` on port 8001)
- ⚡ **Next.js frontend** — toggle between Image and Video analysis in one UI (`inspecai/` on port 3000)

---

## Project Structure

```
InSpecAI/
├── backend/          FastAPI + HuggingFace Transformers (image, port 8000)
├── backend_vid/      FastAPI + CViT model (video, port 8001)
├── inspecai/         Next.js 14 frontend (port 3000)
├── start_all.bat     One-click launcher (Windows)
└── README.md
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 + |
| Node.js | 20 + |
| npm | 9 + |
| Git | any |

---

## ⚡ Quick Start — `start_all.bat` (Windows)

**The fastest way to run the whole stack — one double-click launches everything.**

```
start_all.bat    ← double-click this from the project root
```

The script opens **three separate terminal windows** simultaneously:

| Window | Service | URL |
|---|---|---|
| `InSpecAI - Image Backend :8000` | FastAPI image inference | http://localhost:8000 |
| `InSpecAI - Video Backend :8001` | FastAPI video inference | http://localhost:8001 |
| `InSpecAI - Frontend :3000` | Next.js dev server | http://localhost:3000 |

### What the batch file does

```batch
:: 1. Activates the venv in backend/ (if it exists) then starts uvicorn on :8000
:: 2. Waits 2 seconds
:: 3. Activates the venv in backend_vid/ (if it exists) then starts uvicorn on :8001
:: 4. Waits 2 seconds
:: 5. Runs `npm run dev` inside inspecai/ on :3000
```

### Before running `start_all.bat`

1. **Python deps installed** for both backends (see Manual Setup below)
2. **Node deps installed** — run `npm install` inside `inspecai/` once
3. **`backend_vid/.env` exists** with `HF_WEIGHT_REPO=Adit1Sharma/cvit-deepfake-weights`

> **First run:** the image models (~86–93 MB each × 10) and the CViT video weight (~1 GB) download automatically from HuggingFace and are cached locally. Every subsequent start is instant.

### You can also run it from PowerShell / CMD

```powershell
# From the project root
.\start_all.bat
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `uvicorn` not found | Install deps: `pip install -r backend/requirements.txt` |
| `npm` not found | Install Node.js 20+ from https://nodejs.org |
| Video backend crashes on start | Check that `backend_vid/.env` has `HF_WEIGHT_REPO` set |
| Port already in use | Kill the old process or change the port in the bat file |

---

## Manual Setup (step by step)

### 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/InSpecAI.git
cd InSpecAI
```

---

### 2 — Image Backend (`backend/`)

```bash
cd backend

# (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

**Create `backend/.env`** (copy from example):

```bash
copy .env.example .env
```

Open `.env` and fill in optional values:

```ini
# Optional — speeds up first HuggingFace model download
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Optional — saves benchmark results to Neon Postgres
DATABASE_URL=postgresql://user:password@host/db?sslmode=require
```

**Run the image backend:**

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

---

### 3 — Video Backend (`backend_vid/`)

```bash
cd backend_vid

# (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Create `backend_vid/.env`:**

```bash
copy .env.example .env
```

Open `.env` and set the HuggingFace repo that holds the CViT weight:

```ini
# Required — HF repo where the .pth weight is hosted
HF_WEIGHT_REPO=Adit1Sharma/cvit-deepfake-weights

# Optional — only needed if the repo is private
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> The weight (~1 GB) is downloaded automatically on first startup and cached at  
> `~/.cache/huggingface/hub/` — you never need to manage the file manually.

**Run the video backend:**

```bash
uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

---

### 4 — Frontend (`inspecai/`)

```bash
cd inspecai
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Image / Video Toggle

The UI has an **IMAGE | VIDEO** toggle in the top header.

| Mode | What it does |
|---|---|
| **IMAGE** | Upload JPG / PNG / WEBP → runs 10 HF models in parallel → shows verdict + confidence per model |
| **VIDEO** | Upload MP4 / AVI / MOV / MKV / WEBM → extracts frames → detects faces → runs CViT → shows REAL / FAKE / NO_FACE |

---

## Dataset Format (Batch Benchmark)

Switch to **BATCH BENCHMARK** tab (image mode) and select a folder with this structure:

```
dataset/
  real/
    image_001.jpg
    image_002.png
  fake/
    image_001.jpg
    image_002.jpg
```

Supported formats: `JPG`, `JPEG`, `PNG`, `WEBP`.

---

## Environment Variables Reference

### `backend/.env`

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | No | HuggingFace token (for gated models) |
| `DATABASE_URL` | No | Neon/Postgres URL to persist benchmark results |
| `INSPEC_DEVICE` | No | CUDA device index (default: `-1` = CPU) |

### `backend_vid/.env`

| Variable | Required | Description |
|---|---|---|
| `HF_WEIGHT_REPO` | **Yes** | HF repo ID containing the CViT `.pth` weight |
| `HF_TOKEN` | No | HF token (only if the repo is private) |

---

## API Endpoints

### Image Backend — `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Server status + loaded model list |
| GET | `/api/models` | Available model metadata |
| POST | `/api/predict` | Run one model on an uploaded image |
| POST | `/api/benchmarks` | Save a batch benchmark result |
| GET | `/api/benchmarks` | List saved benchmarks |
| GET | `/api/benchmarks/{id}` | Get one benchmark detail |
| POST | `/api/single-images` | Save a single-image analysis |
| GET | `/api/single-images` | List saved single-image analyses |

### Video Backend — `http://localhost:8001`

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Upload a video → returns `{ prediction, confidence }` |

---

## GPU Acceleration

**Image backend** — set `INSPEC_DEVICE=0` in `backend/.env` or before starting:
```bash
set INSPEC_DEVICE=0
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**Video backend** — automatically uses CUDA if available (detected via `torch.cuda.is_available()`).

---

## What's git-safe

```
✅  backend/           pure Python — no binaries
✅  backend_vid/       Python — weight downloaded from HF on first run
✅  inspecai/          Next.js source
❌  backend_vid/weight/*.pth   gitignored (1 GB — hosted on HuggingFace)
❌  backend_vid/venv/          gitignored
❌  backend_vid/uploads/       gitignored
❌  backend/venv/              gitignored (if present)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Vanilla CSS |
| Image API | FastAPI, HuggingFace `transformers`, Pillow |
| Video API | FastAPI, PyTorch, CViT, OpenCV, facenet-pytorch |
| Weight storage | HuggingFace Hub |
| DB (optional) | Neon Postgres via `psycopg` |
