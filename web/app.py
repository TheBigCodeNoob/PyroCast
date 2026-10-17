"""PyroCast Florida — web backend. Serves the precomputed risk map and answers on-demand
"why is this high/low risk?" explanations. Pure sklearn model, no GEE at request time."""
import pathlib, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore', message='X does not have valid feature names')  # LGBM log spam
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pyrocast_core as core

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'
app = FastAPI(title='PyroCast Florida')


@app.exception_handler(Exception)
async def _unhandled(request, exc):
    """Surface the real error (e.g. a failed model load) to the client + logs instead of a bare 500
    that the UI shows as 'Could not analyze this location.'"""
    import traceback
    traceback.print_exc()
    return JSONResponse({'error': f'{type(exc).__name__}: {exc}'}, status_code=500)


_grid = None
_coords = None
_grid_mtime = 0.0


def _ensure():
    """Load the grid + model once; reload if the live refresher rewrote fl_grid_full.csv. Atomic:
    _grid/_coords are set ONLY after the model loads, so a failed load doesn't half-initialize
    (leaving _coords None and poisoning every later request)."""
    global _grid, _coords, _grid_mtime
    f = DATA / 'fl_grid_full.csv'
    if not f.exists():
        return _coords is not None
    mt = f.stat().st_mtime
    if _coords is None or mt != _grid_mtime:
        g = pd.read_csv(f).reset_index(drop=True)
        core.load(); core.set_medians(g)
        _grid = g
        _coords = g[['lon', 'lat']].values
        _grid_mtime = mt
    return True


def _nearest(lon, lat):
    d = (_coords[:, 0] - lon) ** 2 + (_coords[:, 1] - lat) ** 2
    return int(np.argmin(d))


@app.get('/api/risk')
def risk():
    p = DATA / 'fl_risk.json'
    if not p.exists():
        return JSONResponse({'error': 'risk grid not computed yet'}, status_code=503)
    return FileResponse(p, media_type='application/json')


@app.get('/api/meta')
def meta():
    p = DATA / 'fl_risk.json'
    if p.exists():
        return json.load(open(p)).get('meta', {})
    return {}


_ignitions = None


@app.get('/api/ignitions')
def ignitions():
    """Actual recorded Florida wildfire ignitions (FPA-FOD 2017-2020) for visual validation."""
    global _ignitions
    if _ignitions is None:
        f = HERE.parent / 'fpafod_se.csv'
        if not f.exists():
            return {'points': [], 'n': 0}
        d = pd.read_csv(f, usecols=['latitude', 'longitude', 'nwcg_cause_classification'])
        d = d[(d.latitude < 31.0) & (d.latitude > 24.4) & (d.longitude > -87.7) & (d.longitude < -79.8)]
        _ignitions = [{'lon': round(float(r.longitude), 4), 'lat': round(float(r.latitude), 4),
                       'c': 1 if r.nwcg_cause_classification == 'Human' else 0} for r in d.itertuples()]
    return {'points': _ignitions, 'n': len(_ignitions)}


class ExplainReq(BaseModel):
    lon: float
    lat: float
    radius_km: float = 0.0   # >0 = explain the group of points within this radius


@app.post('/api/explain')
def explain(req: ExplainReq):
    if not _ensure():
        return JSONResponse({'error': 'grid not ready'}, status_code=503)
    if req.radius_km and req.radius_km > 0:
        deg = req.radius_km / 111.0
        sel = np.where((np.abs(_coords[:, 0] - req.lon) < deg) & (np.abs(_coords[:, 1] - req.lat) < deg))[0]
        if len(sel) == 0:
            sel = np.array([_nearest(req.lon, req.lat)])
        sel = sel[:40]
        aggP, aggC, acts = {}, {}, {}
        rates, rels, expos, tiers, los, his = [], [], [], [], [], []
        weather = False
        for i in sel:
            ex = core.explain(_grid.iloc[int(i)])
            rates.append(ex.get('rate', 0.0)); rels.append(ex.get('rel_risk', 0.0))
            los.append(ex.get('rate_lo', 0.0)); his.append(ex.get('rate_hi', 0.0))
            expos.append(ex.get('exposure', 0.0)); tiers.append(ex.get('tier', 0))
            weather = weather or bool(ex.get('weather_driven'))
            for a in ex.get('actions', []):
                acts.setdefault(a['driver'], a['do'])
            for c in ex.get('place', []):
                aggP[c['factor']] = aggP.get(c['factor'], 0.0) + c['effect']
            for c in ex.get('conditions', []):
                aggC[c['factor']] = aggC.get(c['factor'], 0.0) + c['effect']
        place = sorted([{'factor': k, 'effect': round(v / len(sel), 4)} for k, v in aggP.items()], key=lambda x: -x['effect'])[:4]
        cond = sorted([{'factor': k, 'effect': round(v / len(sel), 4)} for k, v in aggC.items()], key=lambda x: -x['effect'])[:4]
        tmax = int(max(tiers)) if tiers else 0                       # worst tier in the area = what to prioritize
        return {'n_points': int(len(sel)), 'place': place, 'conditions': cond,
                'rate': round(float(np.mean(rates)), 2), 'rel_risk': round(float(np.mean(rels)), 1),
                'rate_lo': round(float(np.mean(los)), 2), 'rate_hi': round(float(np.mean(his)), 2),
                'exposure': round(float(np.mean(expos)), 2),
                'tier': tmax, 'tier_name': core.TIER_NAMES[min(tmax, 5)],
                'actions': [{'driver': k, 'do': v} for k, v in list(acts.items())[:3]],
                'weather_driven': bool(weather)}
    i = _nearest(req.lon, req.lat)
    out = core.explain(_grid.iloc[i])
    out['risk'] = round(float(_grid.iloc[i].get('risk', out['risk'])), 4)   # percentile for display
    out['grid_lon'] = round(float(_coords[i, 0]), 4); out['grid_lat'] = round(float(_coords[i, 1]), 4)
    return out


@app.get('/scorecard')
def scorecard():
    """The self-audit: every significant FL fire since launch, scored against the map PyroCast
    published ~7 days earlier. Honest by construction: events are selected by NASA FIRMS, not us."""
    from fastapi.responses import HTMLResponse
    sc = {'start': '2026-10-16', 'events': []}
    p = DATA / 'scorecard.json'
    if p.exists():
        sc = json.load(open(p))
    ev = sorted(sc.get('events', []), key=lambda e: e['date'], reverse=True)
    rows = ''
    for e in ev:
        img = f"<a href='/data/audit/{e['img']}'><img src='/data/audit/{e['img']}' style='height:72px;border-radius:6px'></a>" if e.get('img') else '—'
        rows += (f"<tr><td>{e['date']}</td><td>{e['lat']:.2f}, {e['lon']:.2f}</td>"
                 f"<td>{e['detections']}</td><td><b>{e['pctile']:.0f}th</b></td>"
                 f"<td>{e['tier']}</td><td>{img}</td></tr>")
    if ev:
        pcts = [e['pctile'] for e in ev]
        med = sorted(pcts)[len(pcts) // 2]
        hi = sum(1 for e in ev if e['tier'] in ('High', 'Very High', 'Extreme'))
        summary = (f"<div class=stats><div><b>{len(ev)}</b><span>fires audited</span></div>"
                   f"<div><b>{med:.0f}th</b><span>median ignition percentile</span></div>"
                   f"<div><b>{hi}/{len(ev)}</b><span>started in High+ zones (top 25% of land)</span></div></div>")
    else:
        summary = ("<p class=empty>No significant fires detected in Florida since launch — "
                   "the watch is running and every qualifying fire will appear here automatically.</p>")
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>PyroCast — Live Scorecard</title>
<meta name=viewport content='width=device-width,initial-scale=1'>
<style>body{{background:#0e1116;color:#dde3ea;font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0}} .sub{{color:#8b97a5;font-size:13px;margin:6px 0 18px}}
.stats{{display:flex;gap:28px;background:#161b22;border:1px solid #2d333b;border-radius:10px;padding:16px 20px;margin:14px 0}}
.stats b{{font-size:26px;display:block;color:#f3c63f}} .stats span{{font-size:12px;color:#8b97a5}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td,th{{padding:8px 10px;border-bottom:1px solid #2d333b;text-align:left}}
th{{color:#8b97a5;font-weight:600;font-size:12px;text-transform:uppercase}}
.empty{{background:#161b22;border:1px solid #2d333b;border-radius:10px;padding:18px}} a{{color:#6cb6ff}}</style></head><body>
<h1>Live self-audit scorecard</h1>
<p class=sub>Every significant Florida fire detected by NASA FIRMS since {sc.get('start', '2026-10-16')},
scored against the risk map PyroCast had published <b>7 days before</b> each fire started.
Fires are chosen by the satellite, not by us. <a href='/'>&larr; back to the map</a></p>
{summary}
{'<table><tr><th>fire start</th><th>location</th><th>detections</th><th>pre-fire rank</th><th>tier</th><th>map</th></tr>' + rows + '</table>' if ev else ''}
</body></html>"""
    return HTMLResponse(html)


@app.get('/')
def index():
    # no-cache: browsers heuristic-cache responses without it, serving stale UI after deploys
    return FileResponse(HERE / 'static' / 'index.html', headers={'Cache-Control': 'no-cache'})


@app.on_event('startup')
def _start_live():
    try:
        import live
        live.start()
    except Exception as e:
        print(f'[live] not started: {type(e).__name__}: {e}')


app.mount('/data', StaticFiles(directory=str(DATA)), name='data')   # serves fl_risk.png + fl_priority.png
app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')
