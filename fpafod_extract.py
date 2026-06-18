"""Pull all SE-US FPA-FOD wildfire ignitions 2017-2020 (lat/lon/year/doy/cause/size)
from the FS ArcGIS FeatureServer (paginated). Saves fpafod_se.csv. No bulk download."""
import csv, time, urllib.parse, urllib.request, json

BASE = "https://apps.fs.usda.gov/ArcX/rest/services/EDW/EDW_FireOccurrence6thEdition_01/MapServer/29/query"
WHERE = "fire_year>=2017 AND state IN ('FL','GA','AL','SC','NC','MS','LA','TN')"
FIELDS = "fod_id,fire_year,discovery_doy,latitude,longitude,nwcg_cause_classification,nwcg_general_cause,fire_size"
OUT = "fpafod_se.csv"


def page(offset):
    params = {'where': WHERE, 'outFields': FIELDS, 'returnGeometry': 'false', 'f': 'json',
              'resultOffset': offset, 'resultRecordCount': 2000, 'orderByFields': 'fod_id'}
    url = BASE + '?' + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r).get('features', [])
        except Exception as e:
            print(f'  retry {attempt} ({e})', flush=True); time.sleep(3)
    return []


rows, offset = [], 0
while True:
    feats = page(offset)
    if not feats:
        break
    rows += [f['attributes'] for f in feats]
    offset += len(feats)
    if offset % 10000 == 0 or len(feats) < 2000:
        print(f'  pulled {offset}', flush=True)
    if len(feats) < 2000:
        break

cols = ['fod_id', 'fire_year', 'discovery_doy', 'latitude', 'longitude', 'nwcg_cause_classification', 'nwcg_general_cause', 'fire_size']
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'Saved {OUT}: {len(rows)} ignitions')
# quick summaries
import collections
yr = collections.Counter(r.get('fire_year') for r in rows)
cz = collections.Counter(r.get('nwcg_cause_classification') for r in rows)
print('by year:', dict(sorted(yr.items())))
print('by cause:', dict(cz))
