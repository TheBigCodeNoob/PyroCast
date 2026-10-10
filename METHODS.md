# PyroCast Florida — label accuracy & methods

Every number the tool publishes is grounded in the real fire record and reported only to the
precision the data supports. This document is the audit trail; regenerate it any time with:

```
conda run -n base python web/audit_labels.py      # the full validation report
conda run -n base python web/compute_risk.py       # rebuild the map + calibrated labels
```

## Data

- **Model:** `best_model_fl_ensemble.joblib` — HistGradientBoosting + RandomForest + LightGBM ensemble,
  58 features, trained on 7,515 Florida rows. Its blocked space-and-time headline is AUROC **0.785**
  (95% spatial-block bootstrap interval **0.760–0.810**, 1,773 held-out samples across 60 blocks).
- **Ground truth:** FPA-FOD recorded ignitions, Florida, **2017–2020** (4 years). 10,361 of 10,484
  in-bbox ignitions snap onto the grid (98.8%); the rest fall on ocean / no-vegetation cells that are
  not modeled.
- **Grid:** 9,417 land cells at 0.04° (~**17.2 km²**/cell, ~4.4 km).

## How the labels are built

1. **Tiers** (Minimal, Low, Moderate, High, Very High, Extreme) are assigned by model-score
   **percentile**, with cut points at the 25th/50th/75th/90th/98th percentiles.
2. **The published rate for each tier is its OBSERVED ignition rate** over 2017–2020 — not a model
   extrapolation. Every cell in a tier shows that tier's empirical rate, with a Poisson 95% CI. (An
   earlier build showed per-cell model-fit rates, which over-stated the top tier at "up to 5.8"; that
   was removed.)
3. **Relative risk** = tier rate ÷ the **statewide average** rate (1.59 /100 km²/yr).

### Validated tier rates (ignitions / 100 km² / yr)

| Tier | cells | observed rate | 95% CI | × FL avg |
|------|------:|------:|:------:|:---:|
| Minimal | 2354 | 0.81 | 0.77–0.85 | 0.5× |
| Low | 2354 | 1.40 | 1.34–1.46 | 0.9× |
| Moderate | 2354 | 1.75 | 1.69–1.82 | 1.1× |
| High | 1413 | 2.11 | 2.02–2.21 | 1.3× |
| Very High | 753 | 2.80 | 2.66–2.95 | 1.8× |
| Extreme | 189 | 3.38 | 3.07–3.72 | 2.1× |

Monotonic across all tiers (verified). Statewide average: **1.59**.

## Validation

- **Predictive ranking (headline)** — AUROC **0.785** on out-of-fold Florida samples. Five-fold
  evaluation holds out 0.6° spatial blocks; each model trains on years through 2019 and is tested on
  2020 in blocks it never saw. The test contains 1,060 fires and 713 season-matched sampled
  background points. Reproduce with `python validate_fl_headline.py`. This estimates discrimination
  between recorded fires and the sampled background, not precision or calibration at the real-world
  fire rate; use a prospective, population-representative evaluation for operational alert claims.
- **Spatial holdout** — refit the rate curve on 0.6° spatial blocks it never saw, predict the held-out
  blocks: out-of-sample error **0.00–0.17** ignitions/100 km²/yr per tier. The rates are not overfit;
  they generalize to unseen areas.
- **Temporal holdout** — calibrate on 2017–2019, test on 2020: tier order holds; point rates vary by
  roughly ±20–30% year to year. **Read the tier and its CI, not a single decimal.**
- **Retrospective grid diagnostic** — AUROC **0.643** for ranking whether a cell had any recorded
  ignition in 2017–2020; the top 5% of cells contain 9.8% of those ignitions (~2× lift), and the top
  10% contain 18.3%. This is useful for describing the current map's retrospective concentration,
  but it is not an independent performance estimate: the scored fire record overlaps all 4,779
  Florida positive training samples. Its 0.6° spatial-block bootstrap interval is 0.615–0.679.

## Cause mix (what is and isn't predictable)

16.8% of Florida ignitions are **natural (lightning)**, which the model cannot anticipate from
landscape + weather. Lightning's share **falls** with tier (23% Minimal → 12% Extreme), so the high
tiers are more human-caused — i.e. more *preventable*, which is where mitigation pays off.

## "People & property at risk" (the second layer)

Priority = **ignition-risk percentile × exposure percentile**, where exposure is the percentile-rank
blend of population density, built surface, and nearby development. Top-priority cells average ~2× the
ignition rate **and** high exposure. **This is exposure-weighted ignition risk — it is NOT a social
vulnerability index (SVI)**; it does not include demographics, response capacity, or evacuation
factors. Use it to find where an ignition would reach the most people and property, not as a measure of
community vulnerability in the FEMA/CDC sense.

## Limitations (read before operational use)

- **Static snapshot.** The weather/drought/fuel-moisture inputs are a single export (June 2026). The
  map reflects those conditions; the "weather-driven" note describes that snapshot, not live weather.
  A live 72-hour refresh requires an Earth Engine **service account** (see deployment notes).
- **Rates are 4-year averages.** Any single year varies (see temporal holdout).
- **Lightning (~1 in 6 fires) is unpredictable** from these inputs.
- **Resolution ~4.4 km** — neighborhood/landscape scale, not parcel-level.
- **Ranking lift ≈2×** — strong enough to prioritize scarce mitigation and detection resources, not to
  guarantee where the next fire starts.
