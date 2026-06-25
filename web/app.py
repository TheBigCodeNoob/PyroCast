"""PyroCast Florida — web backend. Serves the precomputed risk map and answers on-demand
"why is this high/low risk?" explanations. Pure sklearn model, no GEE at request time."""
import pathlib, json, numpy as np, pandas as pd
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

_grid = None
_coords = None


def _ensure():
    global _grid, _coords
    if _grid is None:
        f = DATA / 'fl_grid_full.csv'
        if not f.exists():
            return False
        _grid = pd.read_csv(f).reset_index(drop=True)
        core.load(); core.set_medians(_grid)
        _coords = _grid[['lon', 'lat']].values
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
        agg, risks = {}, []
        for i in sel:
            ex = core.explain(_grid.iloc[int(i)])
            risks.append(float(_grid.iloc[int(i)].get('risk', ex['risk'])))   # percentile
            for c in ex['raises_risk'] + ex['lowers_risk']:
                agg[c['factor']] = agg.get(c['factor'], 0.0) + c['effect']
        items = sorted(agg.items(), key=lambda kv: -kv[1])
        raises = [{'factor': k, 'effect': round(v / len(sel), 4)} for k, v in items if v > 0][:4]
        lowers = [{'factor': k, 'effect': round(v / len(sel), 4)} for k, v in items if v < 0][-2:]
        return {'risk': round(float(np.mean(risks)), 4), 'n_points': int(len(sel)), 'raises_risk': raises, 'lowers_risk': lowers}
    i = _nearest(req.lon, req.lat)
    out = core.explain(_grid.iloc[i])
    out['risk'] = round(float(_grid.iloc[i].get('risk', out['risk'])), 4)   # percentile for display
    out['grid_lon'] = round(float(_coords[i, 0]), 4); out['grid_lat'] = round(float(_coords[i, 1]), 4)
    return out


@app.get('/')
def index():
    return FileResponse(HERE / 'static' / 'index.html')


app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')
