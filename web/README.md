# PyroCast Florida — web demo

A live-style wildfire-ignition risk map for Florida. The model scores a dense grid of points;
the website just displays precomputed results (instant load), and you can click anywhere to ask
**why** that spot is high or low risk.

## Run it

```
pip install fastapi uvicorn   # plus the project's sklearn/pandas/joblib
python web/compute_risk.py     # score the grid -> web/data/fl_risk.json + fl_grid_full.csv
python -m uvicorn app:app --app-dir web --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

## How it works

- **`compute_risk.py`** scores a grid of Florida points with `best_model_fl.joblib` (the
  Florida model) and writes two files: a compact `fl_risk.json` (lon/lat/risk — what the map
  draws) and `fl_grid_full.csv` (all features — used for explanations).
- **`app.py`** (FastAPI) serves the map data and answers `/api/explain` — no Earth Engine at
  request time, so it's fast.
- **`static/index.html`** is the map (Leaflet). Click a point (or switch to Area mode and click a
  region) to see the top factors raising or lowering the risk there.
- **Explanations** use *grouped occlusion*: for each group of conditions (drought, fuel, nearness
  to people, …) the model is re-run with that group neutralized; how much the risk drops is how
  much it was contributing. Plain-English, model-faithful, no extra libraries.

## Keeping it fresh (the 6–12 h refresh)

`refresh.py` re-pulls the latest weather/satellite layers, recomputes the grid, and overwrites
the data files. Schedule it:

- **cron** (Linux/Mac): `0 */8 * * * cd /path/to/PyroCast-1 && python web/refresh.py`
- **Windows Task Scheduler**: run `python web\refresh.py` every 8 hours.

Each refresh exports the grid from Earth Engine (~1 h), so the website itself never waits on it —
it always reads the last good precomputed result. (A production version would cache the *static*
layers — canopy, land cover, human access — and only re-fetch the daily weather/moisture, cutting
the refresh to minutes.)

## Honest note for the demo

Risk values are **relative** (the model is trained balanced, so 0.5 is "average grid cell," not
"50% chance"). The map is a *prioritization* tool — it reliably ranks where ignition is most
likely (top-5% of cells capture the bulk of real fires). It is not a probability of a fire today.
Some satellite moisture layers lag real-time by a couple of weeks; the model handles those gaps
natively, leaning on the always-current weather, drought, fuel, and human-access layers.
