# 🔥 PyroCast — Honest Spatiotemporal Wildfire-Ignition Prediction for the Southeastern US

**What it does:** given a place and a day, PyroCast predicts how likely a wildfire is to *ignite* there in the next couple of days, from weather, drought, vegetation, terrain, fuel structure, and human-landscape signals.

**The headline number:** **0.852 AUC** (95% CI **[0.844, 0.860]**) on *genuinely new places and future time* — and, unusually, that number is **honest**: it survives an aggressive battery of "are you cheating?" tests. The crutch-stripped floor is **~0.79**.

**Why this README is so long:** the whole point of this project is *not* the number — it's the **discipline that makes the number trustworthy**. Most of the work was catching ourselves cheating and fixing it. This document records every struggle in detail, then explains it all again in plain English for a non-technical reader (a science-fair judge).

---

## 📑 Contents

- [PART I — The Technical Record](#part-i--the-technical-record)
  - [1. What PyroCast actually predicts](#1-what-pyrocast-actually-predicts)
  - [2. The one idea that matters: honest evaluation](#2-the-one-idea-that-matters-honest-evaluation)
  - [3. The data](#3-the-data)
  - [4. The features](#4-the-features)
  - [5. The model](#5-the-model)
  - [6. The metric: blocked space+time cross-validation](#6-the-metric-blocked-spacetime-cross-validation)
  - [7. The struggle — a version-by-version war diary](#7-the-struggle--a-version-by-version-war-diary)
  - [8. The crutch hunt: is the skill real?](#8-the-crutch-hunt-is-the-skill-real)
  - [9. Current state — exact numbers](#9-current-state--exact-numbers)
  - [10. Honest caveats](#10-honest-caveats)
  - [11. Reproducing this](#11-reproducing-this)
  - [12. File guide](#12-file-guide)
- [PART II — The Plain-English Version (for everyone)](#part-ii--the-plain-english-version-for-everyone)

---
---

# PART I — The Technical Record

## 1. What PyroCast actually predicts

PyroCast is a **spatiotemporal wildfire-ignition likelihood model**. For a given (location, date), it outputs a probability that a wildfire will *start* at that location within a short lead window (we use **2 days before discovery**, so it is a *forecast*, not a *nowcast*).

This is deliberately **not** three other things people often confuse it with:

| What it is NOT | Why we avoided it |
|---|---|
| **Fire *spread*** (how a known fire grows) | Different problem; already well-studied (e.g. Google's Next-Day Wildfire Spread). |
| **Static *susceptibility*** (a fixed risk map) | Has no time dimension; tends to memorize geography/biome (this is exactly how our v2 cheated). |
| **Large-fire occurrence** (MTBS perimeters) | Biased toward big remote fires; the "skill" is mostly remoteness (this is how v7 cheated). |

We frame it as **presence/background**: *positives* are real recorded ignitions; *negatives* are background place-days where no fire was recorded. The model learns to rank a real ignition above a background point.

The region is the **Southeastern US** — 8 states, bounding box `lat 24.5–37.5, lon −94.5–−75.0`. (The project is historically named "Florida"; it grew to the whole Southeast because Florida alone is too uniform to learn from — see [caveats](#10-honest-caveats).)

---

## 2. The one idea that matters: honest evaluation

A wildfire model can score a beautiful AUC and be **completely useless**, because the test set leaks the answer. Our entire methodology is built to prevent that. Three leakage traps dominated this project:

1. **Biome / geography leakage** — the model learns "this *region* burns a lot" instead of "a fire will start *here, now*." Catch it by testing on **spatially held-out** areas.
2. **Calendar-season leakage** — positives are summer, negatives are random months, so the model just learns "summer = fire." Catch it by **season-matching** the negatives to the positives' month distribution.
3. **Reporting / remoteness / sampling bias** — the *way the data was collected* differs between positives and negatives in a way that has nothing to do with fire (e.g. large fires only happen in remote wildland; fires near towns get reported more). Catch it by **distribution-matching** positives and negatives on the suspect variable and re-scoring.

If a gain doesn't survive all three, we don't keep it. That rule is why the project went *down* (0.95 → 0.69) before it went honestly *up* (→ 0.852).

---

## 3. The data

**Positives — real wildfire ignitions (FPA-FOD).**
The Fire Program Analysis Fire-Occurrence Database is the federal record of US wildfire ignitions: location, discovery date, and **cause**. We pulled **52,633 SE-US ignitions, 2017–2020**, directly from the US Forest Service ArcGIS FeatureServer (`fpafod_extract.py`). Crucially, FPA-FOD **excludes prescribed fire** and records *actual ignitions* (not just large fires). Cause breakdown: **~94% human, ~5% lightning, ~1% other.** We sample up to **25,000** of these as positives.

**Negatives — season-matched background.**
Random points drawn from **all non-water land** across the SE (NLCD land cover ≠ water/ice), each assigned a date sampled from the **same month distribution as the positives** (season-matching). Critically, negatives are drawn from *all* land — including developed areas — *not* just burnable vegetation. (Sampling negatives only from wildland would hand the model a "developed = fire" giveaway, since 94% of fires are human; see [v10b](#v10b--the-breakthrough-fpa-fod-real-ignitions).)

**Lead time.** All predictive features are evaluated at **discovery − 2 days**, so the model never sees the fire's own conditions — it forecasts.

**Final dataset:** 25,000 positives + ~19–25k negatives (the last negative export batches were still trickling in from Google Earth Engine at write time; results are stable).

---

## 4. The features

~52 features per place-day, all computed in **Google Earth Engine** (`Dataget_*.py`):

**Weather (GRIDMET):** precipitation sums (7/14/30/60/90/180/365-day), vapor-pressure deficit (VPD), Energy Release Component (ERC), 100-hr dead fuel moisture, max temperature, min relative humidity — multiple time windows each.

**Drought (GRIDMET/DROUGHT):** Palmer Drought Severity Index (PDSI) at 0/30/90/180-day lags.

**Vegetation greenness (MODIS MOD13Q1):** NDVI, EVI (250 m, 16-day).

**Human access:** population density (WorldPop), **distance-to-developed** (`DistDev`, from NLCD), NLCD developed fraction, **VIIRS nighttime lights**.

**Land cover (NLCD point fractions):** forest, shrub, grass, pasture, wetland, crop, developed.

**Neighborhood landscape context:** developed / forest / wetland / crop / pasture fractions within **0.5–5 km** buffers. *(The v11c/v11d wins.)*

**Vertical fuel structure:** **canopy height** (ETH GlobalCanopyHeight 10 m, 2020) and **tree cover** (Hansen). *(The v11g/v11h win — tall flammable pine vs. low scrub vs. open marsh.)*

**Terrain (SRTM):** elevation. *(Slope, aspect, ruggedness, topographic-position were tested and found NULL — the SE is flat.)*

**Engineered:** PDSI 90-day trajectory, VPD trend, precip deficit, 100-hr fuel-moisture trend, a composite "dryness" index.

---

## 5. The model

**Primary:** `HistGradientBoostingClassifier` (scikit-learn gradient-boosted trees):
`max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25`.

**Why trees:** the signal is tabular, heterogeneous, and interaction-heavy; gradient-boosted trees dominate this regime and need no scaling.

**Robustness across model families** (all on the honest metric): Logistic Regression 0.777, LightGBM 0.806, HistGradientBoosting 0.809, Random Forest 0.811. The result is **not model-specific** — even a linear model gets ~0.78. We also tried a SE-ResNet-18 CNN on image stacks early on (the v2/v3 era); trees on tabular features matched or beat it and are far more interpretable.

> Note: GPU was explored (Keras-3-on-PyTorch in a `pyrocast_cuda` conda env, for an RTX 5070 / CUDA, since TensorFlow has no native-Windows GPU). The winning models are CPU tree-ensembles, so the GPU path is only relevant to the early CNN experiments and to running Earth Engine exports.

---

## 6. The metric: blocked space+time cross-validation

This is the **north-star metric** and the only one we trust for "real-world accuracy." It simulates deploying the model on **a place it has never seen, in a year it has never seen**:

- **Spatial blocking:** group all points into **1° lat/lon grid blocks**; use `GroupKFold` so an entire block is either train or test — never split. → tests **new places**.
- **Temporal split:** train on **≤ 2019**, test on **2020** only. → tests the **future**.
- **Combined:** the test set is the intersection — held-out blocks *in* 2020. The model must extrapolate in **both** space and time.

On top of that, every headline gain is re-checked under **crutch controls** (Section 8). Reported numbers throughout this README are this blocked space+time AUC unless stated otherwise.

---

## 7. The struggle — a version-by-version war diary

This is the heart of the project. Each version either exposed a crutch or earned a real gain.

### v2 — the 0.95 that was a lie (biome shortcut)
A Sentinel-2 CNN scored **0.95 AUC**. Audit revealed a **biome shortcut**: it had learned which *ecoregions* burn, not where a fire would start. With geographic leakage removed it collapsed toward chance. **Lesson:** a high AUC with no spatial holdout means nothing.

### v3 — the first honest model
Rebuilt with **matched-temporal negatives** (same place, non-fire times) and proper holdouts. Honest scores: CNN **0.69**, logistic regression **0.73**. Humbling, but real.

### v6 — 0.93 (calendar-season leak)
Switched to MTBS large-fire ignitions; raw score **0.93**. Audit: the negatives weren't **season-matched**, so the model learned "fire season." Fixed by season-matching → became v7.

### v7 — 0.80 (remoteness artifact)
Season-matched MTBS scored **0.80** space+time. Audit: **MTBS only records *large* fires**, which occur in remote wildland. Population-matching the positives/negatives collapsed it to **~0.67**. The 0.80 was "big fires happen in empty places," not ignition skill. (Univariate temperature/VPD giveaways also collapsed under season-matching, 0.73 → 0.54.)

### v8 — the honest baseline (FIRMS, 0.716)
FIRMS satellite active-fire detections capture *real fire occurrence* (small + near people), not just big fires. Score **0.716 pop-matched** — *lower* than v7's 0.80, but **honest** (pop-matching cost only −0.005). The remoteness crutch was gone (`Pop_Density` univariate |AUC| fell 0.785 → 0.579). **This became the locked baseline.** Caveat: SE FIRMS is heavy on **prescribed fire** (a different process).

### v9 — feature engineering hits the wall (+0.001)
Added Burning Index, fm1000, wind, dry-day streaks, longer drought windows, extremes. Net gain: **+0.001** (noise). Weather/drought signal was already **saturated** by existing features. **Lesson:** the ceiling wasn't features — it was the data.

### v10b — the breakthrough (FPA-FOD real ignitions)
Switched positives to **FPA-FOD real wildfire ignitions** (no prescribed fire, actual ignitions, with cause) and added **human-access** features. Key design fix: draw negatives from *all* land (not just wildland) so "developed = fire" couldn't be exploited. Result: **0.811 as-is / 0.755 pop-matched** — the **first genuine gain** over baseline. Driver: `DistDev` (distance-to-developed) — and we showed it's a **causal** "where" signal, not just an artifact: human-caused fires sit a median **30 m** from development, lightning fires **67 m**, random background **95 m** → *location is set by the cause*, exactly what you'd expect if people genuinely start fires near where people are.

### v11 — pushing from 0.81 to 0.85 (the dev branch)
A disciplined sweep, each step verified against the crutch controls:

| Step | What was added | as-is | crutch-free | Verdict |
|---|---|---|---|---|
| v11a | day-of-week, weekend, seasonality | 0.811 | — | marginal (+0.002 genuine; flagged cyclical-doy as residual-season leakage and dropped it) |
| v11b | **case-crossover negatives** (same place, non-fire day) | — | — | the *when* is learnable (**0.73**) but **orthogonal** to the where-metric; cleanly quantified the where/when tradeoff |
| v11c | neighborhood fuel context + nightlights | **0.827** | 0.778 | ✅ verified (gain *grows* under matching) |
| v11d | crop/pasture/wetland fractions (**agricultural-burning** context) | **0.835** | 0.797 | ✅ verified |
| v11e | **scale positives 10k → 25k** | **0.840** | 0.803 | ✅ confirmed on **fresh untuned data** — overfitting concern resolved |
| v11f | terrain (slope/aspect/ruggedness/TPI) | 0.835 | — | ❌ **NULL** — the SE is flat; topography doesn't predict ignition here |
| v11g/h | **canopy height + tree cover** (vertical fuel) | **0.852** | 0.811 | ✅ verified, **unimpeachable** (pure vegetation, can't be reporting bias) |

**The two nulls are findings too:** weather features (v9) and terrain (v11f) genuinely don't add signal here. The wins were all about **better fuel and landscape characterization** plus **more data**.

---

## 8. The crutch hunt: is the skill real?

Before trusting 0.852 we ran an exhaustive "where could we be fooling ourselves?" audit (`crutch_hunt.py`, `validate_v10b.py`).

**Negative controls (sanity that the pipeline doesn't leak):**
- **Label-shuffle → AUC 0.515** (≈ random). If the harness leaked, shuffled labels would still score high. They don't.
- **Random-noise feature → AUC unchanged.** The model ignores garbage.

**Is a crutch hiding in a non-human factor?** No. Distribution-matching on elevation, NDVI, crop fraction, forest fraction barely moves the AUC (0.82–0.83). The only factors that move it when matched are **human-access** ones.

**Is human-access a crutch?** It's the dominant lever (the single strongest feature is `nbhd_dev_500m`, 0.743 alone) — and it's **spread across 5 features**, so matching on population alone *under-corrects*. So we measured the full bracket:

| How skeptical you want to be | AUC |
|---|---|
| Operational (human-access is real — people *cause* 94% of fires) | **0.852** |
| Remoteness removed (population-matched) | 0.811 |
| Strictest single control (matched on the strongest human feature) | 0.801 |
| **All human-access deleted + remoteness matched (the floor)** | **~0.79** |

Even with *every* human signal stripped out, there's **~0.79** of genuine environmental skill (weather + drought + fuel + canopy). The causality test (30 m / 67 m / 95 m, above) argues the human-access signal is mostly *real* rather than reporting bias. So the honest answer is a **range, 0.79–0.85**, not a single triumphant number — and we say so.

---

## 9. Current state — exact numbers

**Best model:** `best_model_v11h.joblib` — HistGradientBoosting on 25k FPA-FOD positives + ~19k matched negatives, 52 features (weather, drought, vegetation, human-access, neighborhood landscape context, agricultural context, vertical fuel structure).

**Blocked space+time (new place + future):**

| Metric | Value |
|---|---|
| **as-is AUC** | **0.852**  (95% CI **[0.844, 0.860]**) |
| pop-matched (crutch-free) | 0.811 |
| dev500-matched (strictest single) | 0.801 |
| environment-only floor (all human stripped) | 0.792 |
| leave-one-region-out (8 sub-regions) | mean 0.841, min 0.794 |
| Florida-only | 0.751 |
| calibration (Brier) | 0.176 (vs 0.241 baseline) — well-calibrated |
| operational: precision @ top-5% riskiest | ~0.88 *(at evaluation balance)* |

**The whole journey on one line:** `0.81 → 0.827 → 0.835 → 0.840 → 0.852` (as-is), `0.755 → 0.811` (crutch-free). **Every single gain survived the crutch controls.**

---

## 10. Honest caveats

- **Florida-specifically is weaker (~0.75).** The 0.852 leans on the *diversity* of the broader Southeast (varied canopy, terrain, land use). Flat, uniform, low-canopy Florida offers less to discriminate on. The canopy win (v11g/h) added **nothing** in Florida — it helped the varied northern SE. State this plainly.
- **Human-access is mostly-causal but has a reporting-bias tint.** We can't fully separate "people start fires here" from "fires here get reported." The causality test says mostly the former; the honest number is the **range** in Section 8.
- **It's a *where* model more than a *when* model.** Location skill is strong (~0.79 spatial); day-to-day timing is weak (~0.64–0.73) because human ignition timing is near-stochastic. The case-crossover experiment (v11b) showed the "when" *is* learnable (0.73) but is a separate axis from the headline metric — a natural next product is a **P(where) × P(when)** place-day risk map.
- **AUC ≠ real-world hit rate at the true base rate.** Fires are rare; at the real (low) prevalence, precision is far below the 88% quoted at evaluation balance. AUC measures *ranking quality*, which is what's transferable.

---

## 11. Reproducing this

Environment: Python with `scikit-learn`, `pandas`, `numpy`, `earthengine-api`, `lightgbm` (base env for modeling); a `pyrocast_cuda` conda env (Keras-3-on-PyTorch) for Earth Engine auth + any GPU/CNN work.

```
# 1. Pull real ignitions (FPA-FOD)            -> fpafod_se.csv
python fpafod_extract.py
# 2. Export features from Google Earth Engine  -> Drive -> Training Data Florida/v11e/, v11h_canopy/
python Dataget_v11e.py            # 25k positives + matched negatives, full feature stack
python Dataget_v11h_canopy.py     # canopy height + tree cover at the same points
#    (monitor_*.py download the GEE exports from Drive as they finish)
# 3. Evaluate (blocked space+time + crutch controls)
python model_v11h.py
# 4. Train + save the deployable model
python save_best_v11h.py          # -> best_model_v11h.joblib
# 5. Audit honesty
python crutch_hunt.py
python validate_v10b.py           # negative controls, calibration, operational, generalization
```

`EXPERIMENTS.md` is the full lab notebook — every experiment with its numbers, in order.

---

## 12. File guide

| File / pattern | Purpose |
|---|---|
| `EXPERIMENTS.md` | **The lab notebook** — chronological record of every experiment and result |
| `fpafod_extract.py` → `fpafod_se.csv` | Pull 52,633 real SE-US ignitions from FPA-FOD |
| `Dataget_fpafod_v10.py` | v10b generator (FPA-FOD positives + matched negatives + base features) |
| `Dataget_v11c/d/f/g_extra.py` | Cheap "extra feature" exports at fixed points (context, terrain, canopy) for fast A/B tests |
| `Dataget_v11e.py` | The scaled 25k full-feature export |
| `Dataget_v11h_canopy.py` | Canopy/tree-cover at the 25k points |
| `monitor_*.py` | Download GEE exports from Google Drive as batches finish |
| `model_v11h.py`, `model_ignition_v10.py` | Evaluators (blocked space+time + crutch controls) |
| `crutch_hunt.py`, `validate_v10b.py` | The honesty audits (negative controls, matching, calibration, generalization) |
| `Dataget_v11_crossover.py`, `model_v11_crossover.py` | The case-crossover "when" experiment |
| `save_best_v11h.py` → `best_model_v11h.joblib` | Train + save the final deployable model |
| `best_model_v10b/v11c/v11e/v11h.joblib` | Saved model checkpoints along the journey |

---
---

# PART II — The Plain-English Version (for everyone)

*Everything above, with no jargon. If you're a judge or just curious, read this part.*

## What we built

A program that looks at a spot on the map and a day, and says **"a wildfire is likely to start here"** — a day or two *before* it happens. Think of it as a weather forecast, but for "where will someone or something spark a new fire" across the Southeastern United States.

## Why this is way harder than it sounds

You'd think you just feed a computer a bunch of past fires and let it learn. The trap is that **the computer will happily learn the wrong thing and look brilliant doing it.** Our entire project was a fight against that.

Here's the analogy we kept coming back to. Imagine you want to predict which students will get an A. You train a model and it's 95% accurate — amazing! Then you discover it's just checking *which school they go to*. It never learned anything about the actual students; it memorized that "rich suburb school = A." It would be **useless** for a new student at a new school. That's exactly the trap a fire model falls into — it memorizes "this *region* burns" instead of learning "a fire will start *here, on this day*."

So the real work isn't getting a high score. **The real work is proving the score isn't a lie.**

## The cheating problem — and how we caught ourselves three times

We literally watched our own model cheat, three different ways:

1. **The biome cheat (our very first model: 95%).** It looked incredible. Then we realized it had just memorized which *types of landscape* burn — like guessing "A" from the school name. When we forced it to predict for *new areas* it had never seen, the 95% **evaporated**. Painful, but the most important lesson of the project.

2. **The calendar cheat (a later model: 93%).** This one had secretly learned "summer = fire." Of course fires happen in fire season — but that's not *useful*; everyone knows summer is dangerous. We fixed it by making sure the "no-fire" examples came from the *same months* as the real fires, so the model couldn't win just by knowing the season.

3. **The "middle of nowhere" cheat (another model: 80%).** This dataset only contained *big* fires, and big fires happen in remote wilderness. So the model learned "remote = fire." When we corrected for that, the 80% fell to **67%**. Again — a real-looking number that was mostly an accident of *how the data was collected*.

Each time, we **deleted the crutch and accepted the lower, honest score.** Our project's number went *down* (from a fake 95% to an honest 69%) before we earned it back up the right way.

## The whole journey, told like a story

- **We started at a fake 95%, fell to an honest 69%.** Ouch — but now we were measuring reality.
- **We hit a wall around 70–72%.** We tried adding every weather and drought feature we could think of. It barely moved. The lesson: the problem wasn't our cleverness, it was our **data**.
- **The breakthrough: better fire records.** We switched to a federal database of *real* wildfire ignitions (not just big fires, not prescribed burns) that also tells us the **cause**. 94% of these fires are started by **people**. So "how close is this to where people live and work" became a genuinely *useful* clue — and we proved it's real, not a cheat: human-caused fires happen right next to development (about 30 meters away), while lightning fires — which people don't start — happen much farther out (about 70 meters), and random spots are farther still (90 meters). The *cause* decides *where*. That's a real pattern, not a data accident. This got us to **honest 81%**.
- **Then we squeezed out real gains, one careful step at a time:**
  - Looking at the **landscape around** each spot (how much forest, wetland, and farmland is nearby) → up a bit.
  - Adding **farmland context** — turns out a *lot* of Southern fires come from agricultural and debris burning → up a bit more.
  - **Feeding it more data** (10,000 fires → 25,000). This was a big test: would the gains hold up on 15,000 fires the model had *never been tuned on*? **They did — they even got better.** That killed our worst fear (that we were secretly overfitting).
  - **Trying terrain** (hills, slopes, ruggedness) → **nothing.** The Southeast is flat; mountains matter for fire out West, not here. (A "nothing" result is still a real scientific finding.)
  - **The final winner: how *tall* the plants are.** A satellite measures tree-canopy height and tree cover. Tall, dense pine forest is very different fuel than short scrub or open marsh — and that vertical structure was a clue our model had been missing. This is the most trustworthy gain of all, because vegetation height **literally cannot** be a "reporting bias" — it's just measuring the plants. This pushed us over the line to **85%.**

## What our final number actually means

**85% (0.852), and we're 95% sure it's between 84.4% and 86.0%.**

In plain terms: if you hand the model one real fire location and one random spot, **85% of the time it correctly rates the fire location as riskier.** And practically: if you took the 5% of places our model flags as most dangerous on a given day, the real fires are overwhelmingly concentrated in that slice.

We're honest about a *range*, not a single number, depending on how strict you want to be:
- **85%** if you accept that "fires start near people" is a real fact about the world (it is — people cause 94% of them).
- **~79%** if you're maximally paranoid and delete *every* human-related clue, leaving only nature (weather, drought, fuel, tree cover). Even then, there's real skill.

## How do we KNOW we're not fooling ourselves (again)?

This is the part a judge should grill us on, so here are our answers up front:

- **The "shuffle" test.** We scrambled the fire/no-fire labels randomly and re-ran everything. The model dropped to **51.5% — basically a coin flip.** If our pipeline were secretly leaking the answer, scrambled labels would *still* score high. They didn't. This proves the machinery is clean.
- **The "garbage feature" test.** We added a column of pure random noise. The model **ignored it.** It's not just latching onto anything.
- **The "new place, new year" test — always.** We *never* score the model on places or years it trained on. Every number is for genuinely unseen ground.
- **The "delete the clue" test.** For every improvement, we re-checked it after surgically removing the suspicious shortcuts. Every gain we kept **survived** that — several actually got *stronger*, which is the opposite of what a cheat would do.

## What it's bad at (because honesty is the whole point)

- **Florida itself is harder (~75%).** Ironically, the project is named for Florida, but Florida is flat and uniform, so there's less for the model to grab onto. The 85% comes from the *variety* of the whole Southeast. We don't hide this.
- **It's much better at "where" than "when."** Pinpointing the *spot* is strong; pinpointing the exact *day* is weaker, because a human deciding to burn yard debris on a Tuesday is close to random. That's an honest limit of the problem, not a bug.
- **"85%" is a ranking score, not "85% of fires caught."** Fires are rare, so in the real world you'd still get false alarms. The model is excellent at *prioritizing* — telling you where to look first — which is exactly what fire managers need.

## Why this matters

Wildfire ignition prediction in the Southeast is an **under-studied problem** (most research targets the western US or fire *spread*, not Southeastern *ignition*). More than that, this project is a case study in **scientific honesty**: it would have been easy to report a flashy 95% and stop. Instead we hunted down our own cheating, again and again, and built a number we can actually defend — **0.852, and we can show our work for every decimal of it.**

---

*PyroCast — built with a strong bias toward catching ourselves cheating. The model is good; the honesty is the point.*
