# PyroCast v2 — Handoff

Branch: `model-v2-improved-features`  (pushed to origin)
Status: code complete, GEE export running, training pending data.

## What changed

**Feature stack: 15 → 19 channels** (Florida-specific tuning):

| Removed | Added | Reason |
|---|---|---|
| Slope | — | Florida is flat; slope is noise |
| — | ERC, FM100 (GRIDMET) | NFDRS drought / dead-fuel-moisture indices |
| — | LC_Forest, LC_Wetland, LC_Open (NLCD) | Distinguish flatwoods / swamp / scrub / prairie |

Full v2 BAND_NAMES order:

```
Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDMI,
Temp_Max, Humidity_Min, Wind_Speed, Precip, ERC, FM100,
Elevation, LC_Forest, LC_Wetland, LC_Open, Pop_Density
```

## What's running right now

10 GEE export tasks are running in your project `gleaming-glass-426122-k0`,
writing to Drive folder `Fire_Prediction_Dataset_Florida_v2`.

Check status anytime with:

```powershell
python monitor_gee_export.py --once
```

Or watch them and download as they complete:

```powershell
python monitor_gee_export.py
```

## What you do next (one command)

When you're ready to finish training (after GEE tasks finish, or just leave this
running and it'll wait):

```powershell
python run_v2_pipeline.py
```

This walks through, in order:

1. **download** — wait for GEE tasks, pull TFRecord shards from Drive to `Training Data Florida/`
2. **prepare** — merge shards and spatially split (no location leakage)
3. **leakage** — verify the split has zero geographic overlap
4. **train** — run `Training_Florida.py` (CPU, ~hours depending on machine)
5. **validate** — run `test_true_validation.py` for held-out metrics
6. **publish** — copy the trained `best_robust_fire_model_v2.keras` into `web/`

Logs land in `logs/pipeline_<timestamp>.log`. Each stage is idempotent — re-running
skips stages whose outputs already exist, so it's safe to interrupt and resume.

You can also start mid-pipeline:

```powershell
python run_v2_pipeline.py --start-from train
```

## Deploying

The branch is push-safe: `web/app.py` and `web/services/model_runner.py` are
**dual-mode**. They detect the loaded model's `input_shape` and either:

- Use v2 19-channel layout natively (when `best_robust_fire_model_v2.keras` is present), or
- Reproject the v2 GEE stack to the v1 15-channel layout (when the v1 file is loaded).

This means you can:

**Option A (safe):** Push this branch as-is. Railway will redeploy, but since
`best_robust_fire_model_v2.keras` doesn't exist on the server, it'll fall back
to the v1 model. Behavior is unchanged from current production.

**Option B (real upgrade):** After `run_v2_pipeline.py` completes:

1. Inspect `logs/pipeline_*.log` — look for the `test_true_validation.py` output.
   You want val AUC ≥ 0.91 (matching v1) and ideally higher precision/recall.
2. Decide how to ship the model file:
   - If `web/best_robust_fire_model_v2.keras` is **< 100 MB**, just commit it.
   - If **≥ 100 MB**, upload it to a GitHub release and set Railway env var
     `MODEL_URL=https://github.com/.../releases/download/v2/best_robust_fire_model_v2.keras`
     The startup code in `web/app.py` already auto-downloads from `MODEL_URL`.
3. Merge to main:
   ```powershell
   git checkout main
   git merge model-v2-improved-features
   git push origin main
   ```
   Railway redeploys automatically.

## Files added / changed

- `Dataget_Florida.py` — v2 feature stack and column list
- `Prepare_Florida_Data.py` — repo-relative paths, fingerprint uses `LC_Forest`
- `Training_Florida.py` — `CHANNELS=19`, repo-relative paths, output `best_robust_fire_model_v2.keras`
- `check_location_leakage.py` — v2 file paths and fingerprint bands
- `test_true_validation.py` — v2 layout
- `web/app.py` — prefers v2 model, falls back to v1; passes `expected_channels` to runner
- `web/services/gee_layer.py` — emits 19+1 channel stack with NLCD masks and ERC/FM100
- `web/services/model_runner.py` — dual-mode (v1 ↔ v2) with explicit channel reprojection
- `monitor_gee_export.py` — NEW: auto-downloads completed shards from Drive
- `run_v2_pipeline.py` — NEW: end-to-end orchestration

## Tradeoffs I made

- **No cloud training.** Training runs locally on CPU because the existing code
  forces CPU mode (`CUDA_VISIBLE_DEVICES=-1`) due to an RTX 5080 crash. If you
  want to use GPU, change that line in `Training_Florida.py`.
- **GFS doesn't carry ERC/FM100.** For real-time inference, `gee_layer._get_gfs_daily`
  pulls the most-recent-available GRIDMET image and grafts ERC/FM100 onto the GFS
  weather snapshot. Drought/fuel-moisture indices change slowly so a 1-3 day lag
  is fine — better than zeroing those channels and confusing the model.
- **Augmentation on land-cover masks.** The training augmenter still adds Gaussian
  noise (σ=0.02) to all channels then clips to [0,1]. Binary masks come out as
  ~0.0/1.0 with tiny jitter, which the network handles. Removing noise from those
  channels specifically would be a marginal improvement; not worth the complexity.
- **Leaving Pop_Density.** Its causal connection to fire *behavior* is weak, but
  it's a real proxy for human ignition sources, which dominate Florida starts.
  Kept it.
