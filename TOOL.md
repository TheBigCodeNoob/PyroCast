# PyroCast — using it as a tool

Three ways to actually *use* the trained model, beyond the research metrics.

## 1. Operational risk map (any date)

Generate an ignition-risk heatmap for the fire-prone Southeast on any day:

```
python Dataget_grid.py 2020-04-15      # export full features over a 0.1-deg grid for that date
python dl_by_name.py Fire_grid_ "Training Data Florida/grid"   # download the grid from Drive
python render_grid.py                  # score every cell + draw the heatmap
```

Outputs:
- `figures/12_operational_map.png` — SE-wide risk heatmap
- `figures/12b_operational_florida.png` — Florida zoom
- `operational_risk_grid.csv` — every grid cell scored (lon, lat, risk), sorted high to low

The grid export takes ~1-2 hr on Earth Engine (it computes the full 57-feature stack per cell).
Change the date argument to map any day; static features (canopy, land cover, human access) are
cached by EE, only the weather/moisture layers re-compute.

## 2. Score arbitrary points

`predict.py` scores any CSV that has the raw feature columns:

```
python predict.py my_points.csv best_model_v13.joblib
# -> my_points_scored.csv with a 'risk' column (0-1), sorted high to low
```

The model handles missing features natively (HistGradientBoosting), so partial inputs still score.

## 3. Where x When forecaster

The model above answers **"is this place fire-prone?"** (the where). `model_wherewhen.py` pairs it
with a second model trained on same-location/different-day negatives to answer **"is this day
dangerous here?"** (the when). Multiplying them gives a place-AND-day risk:

```
risk(place, day) = P(where) x P(when)
```

On the full place-and-day task this scores ~0.80 (vs ~0.72 for either specialist alone) — the two
capture complementary signals.

## What the model is (best_model_v13.joblib)

HistGradientBoosting on 25k real FPA-FOD wildfire ignitions (SE US, 2017-2020), 58 features:
weather, drought, vegetation greenness, canopy/tree-cover, vegetation & soil moisture, water-stress,
land cover, neighborhood landscape context, agriculture, and human access (population, distance-to-
developed, nightlights, built-up surface).

Honest performance (blocked space+time = new place + future year): **~0.86 AUC** operational,
**~0.83 crutch-free** (remoteness removed), **~0.81** even with every human clue deleted. Weakest in
flat/uniform Florida (~0.77). See `final_numbers.md` and `EXPERIMENTS.md` for the full accounting.

**Honest caveat for any real deployment:** the model is excellent at *prioritizing* where to look
(top-5% riskiest cells capture the bulk of fires), but the absolute probabilities are at the training
balance, not the true (low) base rate. It is a where-it's-likely tool, not a fire alarm.
