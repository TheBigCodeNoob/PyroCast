# Final numbers (reference for the README — put these in your own words)

These supersede the slightly-older numbers in the README. They come from `validate_final.py`
run on the complete dataset (49,792 rows). The README currently says 0.852 / 0.811 / ~0.79;
the authoritative numbers below are very close, and the crutch-free one actually improved.

## The headline (blocked space+time = new place + future year)

| What | Number |
|---|---|
| Headline AUC (single model) | **0.850** (95% CI 0.844–0.858) |
| With a 4-model ensemble (HGB+RF+ExtraTrees+LightGBM) | **0.855** |
| Negative control (shuffle the labels — should be ~0.50) | **0.496** ✅ clean |
| Population-matched (remove remoteness advantage) | **0.826** |
| Strictest single control (matched on strongest human feature) | 0.797 |
| Nature-only floor (every human clue deleted) | **0.796** |
| Leave-one-region-out (8 sub-regions) | mean 0.842, worst 0.796 |
| Human-caused ignitions | 0.854 |
| Lightning ignitions | 0.816 |
| Florida only | **0.772** (pop-matched 0.718) |
| Calibration (Brier, lower better) | 0.155 vs 0.242 baseline |
| Operational: top 5% riskiest are real fires | **90%** (top 1% → 92%) |

So the honest range is **0.80–0.85**, not a single number — and even the most paranoid floor (all human clues gone) is ~0.80.

## Two new results from the final session

**The where × when product (`model_wherewhen.py`).** Two specialist models — one for "is this place
fire-prone?" (where), one for "is this day dangerous here?" (when). On the full place-AND-day task:
- where model alone: 0.722
- when model alone: 0.709
- **where × when product: 0.796** (+0.07 over either alone)

They capture complementary signals, so PyroCast is a full place-and-day forecaster (~0.80 on the
hardest task), not just a where-model. → figure `11_where_x_when.png`.

**Ensemble (`ensemble_test.py`).** Averaging four model families reaches **0.855** (vs 0.850 for the
single model). Small, but free and honest — same features, just better modeling. LightGBM alone (0.854)
slightly beats the original gradient-boosting (0.850).

## Why Florida is the weak spot (for the caveat)

Florida is flat and uniform: its elevation varies only ~1/9th as much as the rest of the Southeast,
and its canopy/weather vary less too. There's simply less environmental variation for the model to
learn from, so in Florida it leans harder on human-access and lands around 0.77 instead of 0.85.

## Figures (in `figures/`, ready to drop into a poster/slides)

| File | Shows |
|---|---|
| `01_journey.png` | The honesty story: reported vs honest AUC across versions, crutches annotated |
| `02_roc.png` | ROC curve of the final model |
| `03_calibration.png` | Predicted probability vs observed fire rate |
| `04_precision_at_k.png` | Precision & recall as you flag the top X% riskiest |
| `05_importance_by_category.png` | What the model relies on, grouped |
| `06_importance_top.png` | Top 14 individual features |
| `07_honesty_bracket.png` | AUC as you strip every crutch (the 0.80–0.85 range) |
| `08_causality.png` | Distance-to-development for human vs lightning vs random fires |
| `09_risk_map.png` | Predicted ignition-risk heatmap, Southeast US |
| `09b_risk_map_florida.png` | Florida risk vs actual fires |
| `10_by_cause_and_month.png` | Predictability by cause and across the year |
| `11_where_x_when.png` | Combining where + when beats either alone |
