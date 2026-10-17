"""
HISTORICAL extension of the Florida training set: ALL 53k FL FPA-FOD ignitions 2001-2016
(MODIS era) + season-matched FL negatives, with the full best-model feature stack. Design is
identical to Dataget_fl.py (positives at discovery-2d; negatives = stratified land points with
months drawn from the positives' month distribution, years uniform 2001-2016).

Known-and-accepted anachronisms (document for training): NLCD/WorldPop/GHSL/canopy are ~2020
vintage; NightLights falls back to 0 before VIIRS (2012); SMAP null before 2015 (HGB-native).

  python Dataget_fl_hist.py --test | python Dataget_fl_hist.py
"""
import ee, sys, csv, random, datetime
ee.Initialize(project='gleaming-glass-426122-k0')
from Dataget_v11e import features
from Dataget_v11h_canopy import extra as canopy_extra
from Dataget_v12_moisture import extra as moist_extra
from Dataget_v11e import COLUMNS as C_BASE

FL_BBOX = ee.Geometry.Rectangle([-87.6, 24.5, -79.8, 31.0])
LEAD_DAYS = 2
N_NEG = 40000
POS_BATCHES = 30
NEG_BATCHES = 22
YEAR0, NYEARS = 2001, 16
EXPORT_FOLDER = 'Fire_FL_hist'
NLCD = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").filter(ee.Filter.eq('system:index', '2021')).first().select('landcover')
LAND = NLCD.neq(11).And(NLCD.neq(12))
BUILT = ee.Image(ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filterDate('2019-01-01', '2021-06-01').first()).select('built_surface').rename('built').unmask(0)
COLUMNS = C_BASE + ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km',
                    'ndmi', 'smap_surface', 'smap_root', 'lst_day', 'lst_night', 'et', 'pet', 'built']


def _load_positives():
    rows = list(csv.DictReader(open('fpafod_fl_hist.csv')))
    random.seed(42); random.shuffle(rows)
    pts, months = [], []
    for r in rows:
        try:
            lon, lat = float(r['longitude']), float(r['latitude'])
            yr, doy = int(float(r['fire_year'])), int(float(r['discovery_doy']))
        except Exception:
            continue
        if not (24.5 < lat < 31.0 and -87.6 < lon < -79.8) or doy < 1 or doy > 366:
            continue
        d = datetime.datetime(yr, 1, 1) + datetime.timedelta(days=doy - 1 - LEAD_DAYS)
        tm = int(d.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        cause = 1 if r['nwcg_cause_classification'] == 'Human' else (0 if r['nwcg_cause_classification'] == 'Natural' else -1)
        pts.append((lon, lat, tm, cause))
        months.append((datetime.datetime(yr, 1, 1) + datetime.timedelta(days=doy - 1)).month)
    return pts, months


POS, MONTHS = _load_positives()
_POOL = []
for _m in range(1, 13):
    _POOL += [_m] * max(1, round(MONTHS.count(_m) / max(1, len(MONTHS)) * 1000))
MONTH_POOL = ee.List(_POOL)


def combined(f):
    d = features(f).toDictionary().combine(canopy_extra(f).toDictionary()).combine(moist_extra(f).toDictionary())
    d = d.set('built', BUILT.reduceRegion(ee.Reducer.first(), f.geometry(), 100).get('built'))
    return ee.Feature(f.geometry(), d)


def pos_fc(lo, hi):
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p[0], p[1]]), {'label': 1, 'cause': p[3], 'target_time': p[2]}) for p in POS[lo:hi]])


def negatives(seed):
    pts = LAND.selfMask().rename('b').stratifiedSample(numPoints=N_NEG, classBand='b', region=FL_BBOX, scale=1000, seed=seed, geometries=True, tileScale=8)
    pts = pts.randomColumn('d', seed + 1).randomColumn('dd', seed + 2).randomColumn('yr', seed + 3).randomColumn('part', seed + 4)
    n = MONTH_POOL.size()

    def setd(f):
        m = ee.Number(MONTH_POOL.get(ee.Number(f.get('d')).multiply(n).floor().min(n.subtract(1))))
        day = ee.Number(f.get('dd')).multiply(27).floor().add(1)
        year = ee.Number(YEAR0).add(ee.Number(f.get('yr')).multiply(NYEARS).floor()).min(YEAR0 + NYEARS - 1)
        return f.set({'label': 0, 'cause': -1, 'target_time': ee.Date.fromYMD(year, m, day).millis(), 'part': f.get('part')})
    return pts.map(setd)


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(combined).first()).toDictionary().getInfo()
    import datetime as dt
    yrs = sorted({dt.datetime.utcfromtimestamp(p[2] / 1000).year for p in POS[:5]})
    print(f'{len(POS)} hist positives (sample years {yrs}); cols present {len([c for c in COLUMNS if c in one])}/{len(COLUMNS)}')
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    print(f'FL-hist export: pos={len(POS)} neg={N_NEG} cols={len(COLUMNS)} tasks={POS_BATCHES}+{NEG_BATCHES}')
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(combined), description=f'Fire_FLhist_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_FLhist_pos_{i+1}', flush=True)
    neg = negatives(909)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(combined), description=f'Fire_FLhist_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_FLhist_neg_{i+1}', flush=True)
    print(f"-> Drive folder '{EXPORT_FOLDER}' (download tomorrow with: python dl_by_name.py Fire_FLhist_ \"Training Data Florida/FL_hist\")")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
