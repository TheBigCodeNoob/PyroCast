import sys, json, pathlib
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from ee import oauth
prefix, dest = sys.argv[1], pathlib.Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
data = json.load(open(pathlib.Path.home()/".config"/"earthengine"/"credentials"))
c = Credentials(token=None, refresh_token=data["refresh_token"], token_uri="https://oauth2.googleapis.com/token", client_id=oauth.CLIENT_ID, client_secret=oauth.CLIENT_SECRET, scopes=data.get("scopes", []))
d = build("drive","v3",credentials=c, cache_discovery=False)
tok=None; got=0
while True:
    r = d.files().list(q=f"name contains '{prefix}' and trashed=false", fields="nextPageToken, files(id,name,size)", pageSize=200, pageToken=tok).execute()
    for f in r.get("files",[]):
        if not f["name"].startswith(prefix) or not f["name"].endswith(".csv"): continue
        p = dest/f["name"]
        if p.exists() and int(f.get("size",0))==p.stat().st_size: got+=1; continue
        with open(p,"wb") as fh:
            dl=MediaIoBaseDownload(fh, d.files().get_media(fileId=f["id"]), chunksize=4*1024*1024); done=False
            while not done: _,done=dl.next_chunk()
        got+=1
    tok=r.get("nextPageToken")
    if not tok: break
print(f"downloaded/present {got} files -> {dest}")
