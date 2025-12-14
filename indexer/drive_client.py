from googleapiclient.discovery import build
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_images(folder_id):
    """
    List all image files in a Google Drive folder, including RAW formats (ARW, CR2, NEF),
    and skip non-image files like MP4, XML, etc.
    """
    service = get_drive_service()
    query = f"'{folder_id}' in parents"
    
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, parents)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    files = results.get('files', [])
    images = []

    for f in files:
        name_lower = f['name'].lower()
        # Accept standard images or common RAW extensions
        if f['mimeType'].startswith('image/') or name_lower.endswith(('.arw', '.cr2', '.nef')):
            images.append(f)
        else:
            print(f"Skipping non-image file: {f['name']} ({f['mimeType']})")
    
    print(f"Folder {folder_id}: {len(images)} images found")
    return images


