"""
Monitors the v4 Earth Engine export tasks and pulls completed TFRecord shards
from Google Drive into 'Training Data Florida/v4/'. Runs hands-free until all 12
tasks finish and Drive is synced, then exits.

Reuses the OAuth credentials saved by `earthengine authenticate`
(~/.config/earthengine/credentials, drive scope included). Scans ALL Drive
folders named like the export folder (GEE created two), so no shard is missed.

Usage:
    python monitor_gee_export_v4.py            # watch & download until all done
    python monitor_gee_export_v4.py --once     # one scan & exit
"""
import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Iterable

import ee

EXPORT_FOLDER = "Fire_Prediction_Dataset_Florida_v4"
TASK_DESCRIPTION_PREFIX = "Export_Florida_Fire_Dataset_v4_Part_"
PROJECT_ID = "gleaming-glass-426122-k0"
LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "Training Data Florida" / "v4"
POLL_INTERVAL_SEC = 180
CREDS_PATH = pathlib.Path.home() / ".config" / "earthengine" / "credentials"
TERMINAL = ("SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED")


def build_drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from ee import oauth as ee_oauth
    if not CREDS_PATH.exists():
        sys.exit("No earthengine credentials. Run `earthengine authenticate` first.")
    data = json.load(open(CREDS_PATH))
    creds = Credentials(
        token=None, refresh_token=data["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
        scopes=data.get("scopes", []))
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_export_tasks() -> list:
    out = []
    for t in ee.data.listOperations():
        meta = t.get("metadata", {})
        desc = meta.get("description", "")
        if desc.startswith(TASK_DESCRIPTION_PREFIX):
            out.append({"description": desc, "state": meta.get("state", "UNKNOWN")})
    return out


def find_all_folders(drive, folder_name: str) -> list:
    res = drive.files().list(
        q=(f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
           "and trashed = false"),
        fields="files(id, name)", pageSize=50).execute()
    return res.get("files", [])


def list_files_in_folder(drive, folder_id: str) -> Iterable[dict]:
    page_token = None
    while True:
        res = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, size)",
            pageSize=200, pageToken=page_token).execute()
        for f in res.get("files", []):
            yield f
        page_token = res.get("nextPageToken")
        if not page_token:
            break


def download_file(drive, file_id: str, dest_path: pathlib.Path):
    from googleapiclient.http import MediaIoBaseDownload
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(tmp, "wb") as fh:
        request = drive.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    tmp.rename(dest_path)


def scan_and_download(drive) -> tuple:
    folders = find_all_folders(drive, EXPORT_FOLDER)
    if not folders:
        print(f"  Drive folder '{EXPORT_FOLDER}' not found yet.", flush=True)
        return 0, 0
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = present = 0
    seen = set()
    for folder in folders:
        for f in list_files_in_folder(drive, folder["id"]):
            name = f["name"]
            if not (name.endswith(".tfrecord") or name.endswith(".tfrecord.gz")):
                continue
            if name in seen:
                continue
            seen.add(name)
            dest = LOCAL_DIR / name
            if dest.exists() and int(f.get("size", 0)) == dest.stat().st_size:
                present += 1
                continue
            size_mb = int(f.get("size", 0)) / 1024 / 1024
            print(f"  Downloading {name} ({size_mb:.1f} MB) ...", flush=True)
            try:
                download_file(drive, f["id"], dest)
                downloaded += 1
                present += 1
            except Exception as e:
                print(f"    Download failed: {e}", flush=True)
    return downloaded, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    ee.Initialize(project=PROJECT_ID)
    drive = build_drive_service()
    print(f"Output dir: {LOCAL_DIR}", flush=True)

    while True:
        tasks = list_export_tasks()
        states = dict(Counter(t["state"] for t in tasks))
        print(f"[{time.strftime('%H:%M:%S')}] v4 tasks ({len(tasks)}): {states}", flush=True)
        downloaded, present = scan_and_download(drive)
        print(f"  shards local: {present} | newly downloaded: {downloaded}", flush=True)

        all_terminal = tasks and all(t["state"] in TERMINAL for t in tasks)
        if all_terminal and downloaded == 0:
            failed = [t["description"] for t in tasks if t["state"] in ("FAILED", "CANCELLED")]
            print(f"DONE. All tasks terminal. shards={present}. failed={failed}", flush=True)
            return
        if args.once:
            return
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
