from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import numpy as np
import cv2
import insightface
from typing import List
from PIL import Image
import pillow_heif
import io
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os


pillow_heif.register_heif_opener()

DB_FILE = "faces.db"

app = FastAPI(title="Face Search API")


# Allow frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LOAD FACE MODEL
# ----------------------------
os.environ["ORT_LOGGING_LEVEL"] = "ERROR"
os.environ["OMP_NUM_THREADS"] = "1"

model = insightface.app.FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"]
)

model.prepare(
    ctx_id=-1,
    det_size=(320, 320)   # smaller = MUCH less memory
)

# ----------------------------
# LOAD DATABASE INTO MEMORY
# ----------------------------

def load_faces():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT file_id, file_name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()

    embeddings = []
    meta = []

    for file_id, file_name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings.append(emb)
        meta.append((file_id, file_name))

    embeddings = np.vstack(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings, meta


FACE_EMBEDDINGS, FILE_META = load_faces()
print(f"Loaded {len(FILE_META)} face embeddings into memory")

# ----------------------------
# SEARCH ENDPOINT (WITH PAGINATION)
# ----------------------------
def decode_image(img_bytes: bytes):
    """
    Robust image decoder:
    - JPG / PNG
    - HEIC / HEIF (iPhone)
    - Any PIL-supported format
    Returns OpenCV BGR image or None
    """
    # Fast path: OpenCV
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        # Fallback: PIL (universal)
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    # 🔹 MEMORY FIX: downscale large images
    MAX_SIDE = 1024  # safe for face detection
    h, w = img.shape[:2]
    max_dim = max(h, w)

    if max_dim > MAX_SIDE:
        scale = MAX_SIDE / max_dim
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img


@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    img = decode_image(img_bytes)

    if img is None:
        return {"error": "Unsupported or corrupted image"}

    faces = model.get(img)
    if not faces:
        return {"count": 0, "results": []}

    query_emb = faces[0].embedding
    query_emb /= np.linalg.norm(query_emb)

    # Cosine similarity
    sims = FACE_EMBEDDINGS @ query_emb

    MIN_SCORE = 0.65  # fixed, good default

    results = [
        {
            "file_id": FILE_META[i][0],
            "file_name": FILE_META[i][1],
            "preview_url": f"https://drive.google.com/uc?id={FILE_META[i][0]}",
            "download_url": f"https://drive.google.com/uc?id={FILE_META[i][0]}&export=download",
            "score": float(score),
        }
        for i, score in enumerate(sims)
        if score >= MIN_SCORE
    ]

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "count": len(results),
        "results": results
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists(INDEX_HTML):
        return "<h3>index.html not found</h3>"
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        return f.read()
