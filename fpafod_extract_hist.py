"""Pull ALL Florida FPA-FOD wildfire ignitions 2001-2016 (the MODIS-era historical extension to
the 2017-2020 training window) from the FS ArcGIS FeatureServer. Saves fpafod_fl_hist.csv."""
import csv, time, urllib.parse, urllib.request, json

BASE = "https://apps.fs.usda.gov/ArcX/rest/services/EDW/EDW_FireOccurrenceCurrentEdition_01/MapServer/0/query"
WHERE = "fire_year>=2001 AND fire_year<=2016 AND state='FL'"
FIELDS = "fod_id,fire_year,discovery_doy,latitude,longitude,nwcg_cause_classification,nwcg_general_cause,fire_size"
OUT = "fpafod_fl_hist.csv"


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
import collections
yr = collections.Counter(r.get('fire_year') for r in rows)
cz = collections.Counter(r.get('nwcg_cause_classification') for r in rows)
print('by year:', dict(sorted(yr.items())))
print('by cause:', dict(cz))
