from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from predictor import predict_single

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Deepfake Detection API 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_single(filepath)

    return result