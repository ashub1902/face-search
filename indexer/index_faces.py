# optimized_index_faces.py
import os
import sqlite3
import rawpy
import io
import cv2
import numpy as np
import requests
from tqdm import tqdm
import insightface
from googleapiclient.discovery import build
from google.oauth2 import service_account
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# CONFIGURATION
# ----------------------------
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_IDS = [
    # "1d1BI5cvA9j9XPihFGUyjk4GApHasa8zS",
    # "1prda739i07JQc4Or3r0c1Lw7qgqBsULI",
    # "1TtL58Ps06GHsM5m55-54pl1XTCBVlOpO",
    # "1nUxp5Dsbg8UgOBqHxjZSdVLvpkbV2fYM",
    # "17SuEPaTXNv4UoNc1VcAYRfuE_V-lH4zN",
    # "1eKv_jmYQfsYedXqX_b3ugLOxXDxnHBHN",
    # "1Zo8RrdkOtOt_i_ASLCALdlrVcCrG_M0P",
    # "1lM-k6AFwcLQs1-MRdJ3So2RP0ooqbLe2",
    # "1Tu6pTPsmdIgPcTyFMgqbiP7iMNqbOgu4",
    # "1dwbYQA_8hUhm_uABng3cW8kid6VBNGrI",
    "1JBKLHefjtlgHw2YfmuqCiAW-kEEdH-Lq"
]

DB_FILE = "faces.db"
BATCH_SIZE = 100   # commit every 100 faces
MAX_WORKERS = 4    # adjust based on your M1 cores

# ----------------------------
# GOOGLE DRIVE
# ----------------------------
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=creds)
    return service

def list_images(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents"

    images = []
    page_token = None
    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
            pageSize=1000
        ).execute()

        for f in response.get('files', []):
            name = f['name'].lower()
            if f['mimeType'].startswith('image/') or name.endswith(('.arw', '.cr2', '.nef')):
                images.append(f)

        page_token = response.get('nextPageToken')
        if not page_token:
            break

    print(f"Folder {folder_id}: {len(images)} images found")
    return images

# ----------------------------
# IMAGE DOWNLOAD / READ
# ----------------------------
def download_image(file):
    file_id, file_name = file['id'], file['name']
    url = f"https://drive.google.com/uc?id={file_id}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None, file_id, file_name
    content = resp.content
    ext = file_name.split('.')[-1].lower()

    try:
        if ext in ['jpg', 'jpeg', 'png']:
            img_array = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif ext == 'arw':
            with rawpy.imread(io.BytesIO(content)) as raw:
                rgb = raw.postprocess()
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            return None, file_id, file_name

        # Optional: resize large images for speed
        h, w = img.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        return img, file_id, file_name
    except:
        return None, file_id, file_name

# ----------------------------
# DATABASE
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT,
            file_name TEXT,
            folder_id TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

def insert_faces(conn, rows):
    c = conn.cursor()
    c.executemany('''
        INSERT INTO faces (file_id, file_name, folder_id, embedding)
        VALUES (?, ?, ?, ?)
    ''', rows)
    conn.commit()

# ----------------------------
# FACE DETECTION
# ----------------------------
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # CPU

def process_file(file):
    image, file_id, file_name = download_image(file)
    if image is None:
        return []

    faces = model.get(image)
    results = []
    folder_id = file.get('parents', [''])[0]
    for face in faces:
        results.append((file_id, file_name, folder_id, face.embedding.tobytes()))
    return results

# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    images = []
    for folder_id in FOLDER_IDS:
        images.extend(list_images(folder_id))
    print(f"Total images: {len(images)}")

    conn = init_db()
    rows_to_commit = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, img): img for img in images}
        for future in tqdm(as_completed(futures), total=len(images)):
            try:
                rows = future.result()
                rows_to_commit.extend(rows)
                if len(rows_to_commit) >= BATCH_SIZE:
                    insert_faces(conn, rows_to_commit)
                    rows_to_commit = []
            except Exception as e:
                print(f"Error processing image: {e}")

    if rows_to_commit:
        insert_faces(conn, rows_to_commit)

    conn.close()
    print("Indexing completed!")

if __name__ == "__main__":
    main()
