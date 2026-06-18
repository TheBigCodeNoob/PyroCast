"""
Monitors the v2 Earth Engine export tasks and pulls completed TFRecord shards
from Google Drive directly into the local repo. Runs hands-free.

Reuses the OAuth credentials that `earthengine authenticate` already saved at
~/.config/earthengine/credentials (which include drive scope). No extra auth
flow needed.

Usage:
    python monitor_gee_export.py            # Watch & download until all done
    python monitor_gee_export.py --once     # One scan & exit (for cron-style)
"""
import argparse
import json
import os
import pathlib
import sys
import time
from typing import Iterable

import ee

EXPORT_FOLDER = "Fire_Prediction_Dataset_Florida_v3"
TASK_DESCRIPTION_PREFIX = "Export_Florida_Fire_Dataset_v3_Part_"
PROJECT_ID = "gleaming-glass-426122-k0"
LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "Training Data Florida"
POLL_INTERVAL_SEC = 120  # 2 minutes between Drive scans

CREDS_PATH = pathlib.Path.home() / ".config" / "earthengine" / "credentials"


def build_drive_service():
    """Build a Drive v3 client from the existing earthengine OAuth credentials.

    Reuses the public installed-app client_id/secret bundled with the
    earthengine-api package (ee.oauth) so we don't ship any secrets in this repo.
    """
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from ee import oauth as ee_oauth  # public client_id/secret used by `earthengine` CLI

    if not CREDS_PATH.exists():
        sys.exit("No earthengine credentials found. Run `earthengine authenticate` first.")

    with open(CREDS_PATH) as f:
        data = json.load(f)

    creds = Credentials(
        token=None,
        refresh_token=data["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee_oauth.CLIENT_ID,
        client_secret=ee_oauth.CLIENT_SECRET,
        scopes=data.get("scopes", []),
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_export_tasks() -> list:
    """Return the v2 export tasks, newest first."""
    tasks = ee.data.listOperations()
    out = []
    for t in tasks:
        meta = t.get("metadata", {})
        desc = meta.get("description", "")
        if desc.startswith(TASK_DESCRIPTION_PREFIX):
            out.append({
                "name": t.get("name", ""),
                "description": desc,
                "state": meta.get("state", "UNKNOWN"),
                "type": meta.get("type", ""),
            })
    return out


def find_drive_folder(drive, folder_name: str):
    """Find the export folder by name (most-recently-modified if multiple)."""
    q = (
        f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    res = drive.files().list(
        q=q,
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    return files[0] if files else None


def list_files_in_folder(drive, folder_id: str) -> Iterable[dict]:
    """Yield all files inside the given Drive folder."""
    page_token = None
    while True:
        res = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, size, modifiedTime)",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break


def download_file(drive, file_id: str, dest_path: pathlib.Path):
    """Stream a Drive file to disk."""
    from googleapiclient.http import MediaIoBaseDownload

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(tmp, "wb") as fh:
        request = drive.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(fh, request, chunksize=4 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"    {pct}% ", end="\r", flush=True)
    tmp.rename(dest_path)
    print()


def scan_and_download(drive) -> tuple[int, int]:
    """Returns (downloaded_this_round, total_local_files_present)."""
    folder = find_drive_folder(drive, EXPORT_FOLDER)
    if not folder:
        print(f"Drive folder '{EXPORT_FOLDER}' not found yet.")
        return 0, 0

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    present = 0
    for f in list_files_in_folder(drive, folder["id"]):
        # GEE TFRecord exports come as <description>.tfrecord.gz or .tfrecord with possible
        # multi-shard suffixes (e.g. "..._Part_3-00000-of-00001.tfrecord.gz").
        if not (f["name"].endswith(".tfrecord") or f["name"].endswith(".tfrecord.gz")):
            continue
        dest = LOCAL_DIR / f["name"]
        if dest.exists() and int(f.get("size", 0)) == dest.stat().st_size:
            present += 1
            continue
        size_mb = int(f.get("size", 0)) / 1024 / 1024
        print(f"  Downloading {f['name']}  ({size_mb:.1f} MB) ...")
        try:
            download_file(drive, f["id"], dest)
            downloaded += 1
            present += 1
        except Exception as e:
            print(f"    Download failed: {e}")
    return downloaded, present


def task_summary(tasks: list) -> str:
    counts = {}
    for t in tasks:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit.")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT_ID)
    drive = build_drive_service()
    print(f"Output dir: {LOCAL_DIR}")

    while True:
        tasks = list_export_tasks()
        if tasks:
            print(f"[{time.strftime('%H:%M:%S')}] Tasks: {task_summary(tasks)}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No matching tasks visible.")

        downloaded, present = scan_and_download(drive)
        print(f"  Local files present: {present} | Newly downloaded: {downloaded}")

        if tasks and all(t["state"] in ("SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED")
                         for t in tasks) and downloaded == 0:
            print("All tasks finished and Drive is in sync. Exiting.")
            return

        if args.once:
            return

        print(f"  Sleeping {POLL_INTERVAL_SEC}s ...")
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
