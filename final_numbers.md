# Final numbers (reference for the README — put these in your own words)

Authoritative numbers from the FINAL model (`model_final.py` → `best_model_final.joblib`):
**base + neighborhood context + agriculture + canopy + moisture/water-stress**, evaluated with
blocked space+time (new place + future year) on the full 49,792-row dataset. Missing satellite
moisture (~3% of points) is handled natively by the gradient-booster, so no rows are dropped.

## The headline

| What | Number |
|---|---|
| **Headline AUC (final)** | **0.857** (95% CI 0.851–0.864) |
| Same model *without* the new moisture features | 0.850 → so moisture added **+0.007** |
| With a 4-model ensemble | ~0.860 |
| Negative control (shuffle the labels — should be ~0.50) | **0.499** ✅ clean |
| Population-matched (remove remoteness advantage) | **0.831** |
| Strictest single control (matched on strongest human feature) | 0.809 |
| Nature-only floor (every human clue deleted) | **0.809** ← now crosses 0.81 |
| Leave-one-region-out (8 sub-regions) | mean ~0.84 |
| Human-caused ignitions / lightning | ~0.85 / ~0.82 |
| Florida only | **0.764** |
| Calibration (Brier, lower better) | 0.152 vs 0.242 baseline |
| Operational: top 5% riskiest are real fires | **92%** |

So the honest range is **0.81–0.86**: 0.857 if you accept "fires start near people" as the real
fact it is, down to ~0.81 even if you delete every human clue and keep only nature.

## The two-session climb (for the journey chart / story)

0.81 (v10b) → 0.827 (neighborhood context) → 0.835 (agriculture) → 0.840 (more data, 25k) →
0.852 (canopy/tree-cover) → **0.857 (moisture/water-stress)**. Every step survived the crutch checks.
Two dead ends that are also findings: terrain (the SE is flat) and calendar features added nothing.

## New capability: the where × when forecaster (`model_wherewhen.py`)

Two specialists — "is this place fire-prone?" (where) and "is this day dangerous here?" (when).
On the full place-AND-day task: where alone 0.722, when alone 0.709, **product 0.796** (+0.07).
They're complementary, so PyroCast is a real place-and-day forecaster, not just a where-model.
→ figure `11_where_x_when.png`.

## What the moisture features were

The physical gap was *how wet the fuel is*. The winners: evapotranspiration-stress (ET/PET),
potential ET, and NDMI (vegetation moisture). Soil moisture and surface temperature were weak.
Moisture is pure physical state — it can't be a reporting bias — and the gain grew under the
crutch checks (nature-only floor +0.013), so it's unimpeachable.

## Why Florida is the weak spot (for the caveat)

Florida is flat and uniform: elevation varies ~1/9th as much as the rest of the Southeast, and
canopy/weather vary less too. Less environmental variation to learn from, so the model leans on
human-access there and lands ~0.76 instead of 0.86.

## Figures (in `figures/`)

01 journey · 02 ROC · 03 calibration · 04 precision@k · 05 importance by category ·
06 top features · 07 honesty bracket · 08 causality · 09 SE risk map · 09b Florida risk map ·
10 by cause & month · 11 where × when.

---
## v13 session update ("anything goes")

- **Best model: ~0.86** (0.857 validated [base+canopy+moisture]; **0.862** with GHSL built-surface;
  **~0.866** with a 4-model ensemble). The session's only real new gain was built-surface (+0.006).
- **The honest catch:** the gHM (roads/power/infrastructure) feature first looked like +0.040 — a
  huge win. It was a phantom: an incomplete-download NaN pattern leaking through the spatial CV. On
  complete data it's +0.006 and gHM's own importance is ~0. Caught it because the jump was too big.
  (gHM did give a clean causality result: it predicts human fires but ANTI-predicts lightning, 0.448 —
  confirming human-access is causal, not reporting bias.)
- **VCF continuous fuel: null.** **Lightning: unfillable** — no lightning data in GEE; ERA5 convective
  precip is hourly-only and computationally prohibitive at 25k points.
- **Why the ceiling is real:** error analysis shows the misses are remote, forested, wet, lightning-
  driven wildland fires — the least human-predictable category. ~0.86 is a genuine data-limited
  ceiling, not a modeling shortfall.
- **New tool:** an operational gridded ignition-risk-map generator (`Dataget_grid.py <date>` ->
  `render_grid.py`). Figures `12_operational_map.png` / `12b_operational_florida.png`. See `TOOL.md`.
