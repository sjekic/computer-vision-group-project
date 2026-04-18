"""
Google Drive Dataset Downloader
================================
Downloads the entire IE Tower image dataset from a shared Google Drive folder
into data/raw/, preserving the location sub-folder structure.

Each teammate only needs to:
  1. pip install -r requirements.txt
  2. Place credentials.json in the project root (one-time setup, see README)
  3. python src/download_dataset.py

On first run a browser window opens for Google OAuth consent.
A token.json is saved locally so subsequent runs are silent/automatic.

Usage:
    python src/download_dataset.py                        # download all folders
    python src/download_dataset.py --folder-id <ID>       # custom Drive folder
    python src/download_dataset.py --out-dir data/raw     # custom output dir
    python src/download_dataset.py --dry-run              # list files, no download
    python src/download_dataset.py --resume               # skip already-downloaded files
"""

import os
import io
import argparse
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

# The shared Google Drive folder ID from the project link:
# https://drive.google.com/drive/folders/0AK4xDJSGk2TLUk9PVA
DEFAULT_FOLDER_ID = "0AK4xDJSGk2TLUk9PVA"

DEFAULT_OUT_DIR = Path("data/raw")
CREDENTIALS_FILE = Path("credentials.json")   # OAuth2 client secret file
TOKEN_FILE = Path("token.json")               # saved after first auth

# Read-only access to Drive files is sufficient
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Image MIME types to download (skip Docs, Sheets, etc.)
IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
    "image/heic",
    "image/heif",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Auth ─────────────────────────────────────────────────────────────────────

def get_credentials() -> Credentials:
    """
    Load saved credentials or run the OAuth2 browser flow.
    Saves token.json after first successful auth so future runs are automatic.
    """
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Refresh or re-authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired access token...")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"\n\n[ERROR] '{CREDENTIALS_FILE}' not found.\n"
                    "Please follow the setup instructions in README.md to obtain\n"
                    "a credentials.json from Google Cloud Console.\n"
                )
            logger.info("Opening browser for Google OAuth consent...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future runs
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        logger.info(f"Token saved to {TOKEN_FILE}")

    return creds


# ─── Drive helpers ────────────────────────────────────────────────────────────

def list_folder_contents(service, folder_id: str) -> list[dict]:
    """Return all files and sub-folders inside a Drive folder (handles pagination)."""
    items = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def download_file(service, file_id: str, dest_path: Path) -> bool:
    """Download a single file from Drive. Returns True on success."""
    try:
        request = service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        with open(dest_path, "wb") as f:
            f.write(fh.getvalue())
        return True

    except Exception as e:
        logger.warning(f"Failed to download {dest_path.name}: {e}")
        return False


def walk_drive_folder(
    service,
    folder_id: str,
    local_dir: Path,
    dry_run: bool,
    resume: bool,
    stats: dict,
    depth: int = 0,
):
    """
    Recursively walk a Drive folder tree and download image files.
    Mirrors the folder structure locally under local_dir.
    """
    items = list_folder_contents(service, folder_id)

    folders = [i for i in items if i["mimeType"] == "application/vnd.google-apps.folder"]
    files   = [i for i in items if i["mimeType"] in IMAGE_MIME_TYPES]
    other   = [i for i in items if i not in folders and i not in files]

    if other and depth == 0:
        logger.info(f"Skipping {len(other)} non-image file(s) in root")

    # Download images at this level
    for file in tqdm(files, desc=f"{'  ' * depth}{local_dir.name}", leave=(depth == 0)):
        dest = local_dir / file["name"]

        if resume and dest.exists():
            stats["skipped"] += 1
            continue

        if dry_run:
            size_kb = int(file.get("size", 0)) // 1024
            logger.info(f"[DRY RUN] Would download: {local_dir.name}/{file['name']} ({size_kb} KB)")
            stats["total"] += 1
            continue

        success = download_file(service, file["id"], dest)
        if success:
            stats["downloaded"] += 1
            stats["total"] += 1
        else:
            stats["failed"] += 1

    # Recurse into sub-folders (each becomes a location class)
    for folder in sorted(folders, key=lambda x: x["name"]):
        sub_dir = local_dir / folder["name"]
        walk_drive_folder(
            service, folder["id"], sub_dir,
            dry_run, resume, stats, depth + 1
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download IE Tower dataset from Google Drive")
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID,
                        help="Google Drive folder ID (default: project dataset folder)")
    parser.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR,
                        help="Local output directory (default: data/raw)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="List files without downloading")
    parser.add_argument("--resume",    action="store_true", default=True,
                        help="Skip files that already exist locally (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-download all files even if they exist")
    args = parser.parse_args()

    logger.info("Authenticating with Google Drive...")
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    logger.info(f"Starting {'DRY RUN ' if args.dry_run else ''}download")
    logger.info(f"  Source folder : https://drive.google.com/drive/folders/{args.folder_id}")
    logger.info(f"  Destination   : {args.out_dir.resolve()}")
    logger.info(f"  Resume mode   : {args.resume}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}

    walk_drive_folder(
        service=service,
        folder_id=args.folder_id,
        local_dir=args.out_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        stats=stats,
    )

    print("\n" + "─" * 50)
    print("Download complete" if not args.dry_run else "Dry run complete")
    print(f"  Downloaded : {stats['downloaded']}")
    print(f"  Skipped    : {stats['skipped']}  (already existed)")
    print(f"  Failed     : {stats['failed']}")
    print(f"  Total      : {stats['total']}")
    print("─" * 50)

    if not args.dry_run and stats["downloaded"] > 0:
        print(f"\nDataset ready at: {args.out_dir.resolve()}")
        print("Next step: python src/preprocess.py --augment")


if __name__ == "__main__":
    main()
