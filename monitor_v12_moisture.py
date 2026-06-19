"""Monitor + download v11 case-crossover export -> 'Training Data Florida/v12_moisture/'."""
import json, pathlib, time
from collections import Counter
import ee
EXPORT_FOLDER = "Fire_v12_moisture"; PREFIX = "Fire_v12_"
PROJECT_ID = "gleaming-glass-426122-k0"
LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "Training Data Florida" / "v12_moisture"
CREDS = pathlib.Path.home() / ".config" / "earthengine" / "credentials"
TERMINAL = ("SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED")


def drive():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from ee import oauth
    data = json.load(open(CREDS))
    c = Credentials(token=None, refresh_token=data["refresh_token"], token_uri="https://oauth2.googleapis.com/token",
                    client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET, scopes=data.get("scopes", []))
    return build("drive", "v3", credentials=c, cache_discovery=False)


def tasks():
    return [t["metadata"] for t in ee.data.listOperations() if t.get("metadata", {}).get("description", "").startswith(PREFIX)]


def folders(d):
    return d.files().list(q=(f"name='{EXPORT_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false"), fields="files(id)").execute().get("files", [])


def files_in(d, fid):
    tok = None
    while True:
        r = d.files().list(q=f"'{fid}' in parents and trashed=false", fields="nextPageToken, files(id,name,size)", pageSize=200, pageToken=tok).execute()
        for f in r.get("files", []):
            yield f
        tok = r.get("nextPageToken")
        if not tok:
            break


def dl(d, fid, dest):
    from googleapiclient.http import MediaIoBaseDownload
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, d.files().get_media(fileId=fid), chunksize=4 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def main():
    ee.Initialize(project=PROJECT_ID)
    d = drive(); LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        st = dict(Counter(t["state"] for t in tasks()))
        downloaded = present = 0; seen = set()
        for fol in folders(d):
            for f in files_in(d, fol["id"]):
                if not f["name"].endswith(".csv") or f["name"] in seen:
                    continue
                seen.add(f["name"]); dest = LOCAL_DIR / f["name"]
                if dest.exists() and int(f.get("size", 0)) == dest.stat().st_size:
                    present += 1; continue
                print(f"  downloading {f['name']}", flush=True)
                dl(d, f["id"], dest); downloaded += 1; present += 1
        print(f"[{time.strftime('%H:%M:%S')}] v12_moisture {st} | csv local: {present} new: {downloaded}", flush=True)
        ts = tasks()
        if ts and all(t["state"] in TERMINAL for t in ts) and downloaded == 0:
            print(f"DONE. csv={present}", flush=True); return
        time.sleep(120)


if __name__ == "__main__":
    main()
