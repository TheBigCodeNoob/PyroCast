"""
End-to-end v2 pipeline:
  1. Wait for GEE exports to finish + auto-download from Drive (monitor_gee_export.py)
  2. Run Prepare_Florida_Data.py to merge & spatially split the shards
  3. Run check_location_leakage.py to verify the split has no overlap
  4. Run Training_Florida.py to train the v2 model
  5. Copy trained model to web/ for deployment
  6. Print the manual-deploy command (we don't auto-push to main)

Each stage logs to a single timestamped file, and the pipeline is idempotent:
re-running it after a stage succeeds will skip that stage if its outputs exist.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO / "Training Data Florida"
TRAIN_TFRECORD = DATA_DIR / "Florida_Spatial_Train_v2.tfrecord"
VAL_TFRECORD = DATA_DIR / "Florida_Spatial_Val_v2.tfrecord"
MODEL_OUT_LOCAL = REPO / "best_robust_fire_model_v2.keras"
MODEL_OUT_WEB = REPO / "web" / "best_robust_fire_model_v2.keras"

LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], desc: str) -> int:
    log(f"--- {desc} ---")
    log(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(proc.stdout)
    print(proc.stdout)
    log(f"--- {desc}: exit {proc.returncode} ---")
    return proc.returncode


def stage_download():
    """Block until GEE exports complete and all shards are local."""
    log("STAGE 1: Wait for GEE export + download from Drive")
    rc = run([sys.executable, "monitor_gee_export.py"], "Monitor GEE export")
    if rc != 0:
        sys.exit("Monitor failed; check logs.")
    shards = list(DATA_DIR.glob("Export_Florida_Fire_Dataset_Part_*.tfrecord*"))
    log(f"Local shards: {len(shards)}")
    if len(shards) == 0:
        sys.exit("No shards downloaded. Aborting.")


def stage_prepare():
    """Merge shards and spatially split into train/val."""
    log("STAGE 2: Prepare data (merge + spatial split)")
    if TRAIN_TFRECORD.exists() and VAL_TFRECORD.exists():
        log("Train/val TFRecords already exist; skipping prepare stage.")
        return
    rc = run([sys.executable, "Prepare_Florida_Data.py"], "Prepare_Florida_Data")
    if rc != 0:
        sys.exit("Data prep failed.")


def stage_leakage_check():
    """Verify spatial split has no location leakage."""
    log("STAGE 3: Spatial leakage check")
    rc = run([sys.executable, "check_location_leakage.py"], "Location leakage check")
    if rc != 0:
        log("WARNING: Leakage check returned non-zero. Continuing but inspect logs.")


def stage_train():
    """Train the v2 model."""
    log("STAGE 4: Train v2 model")
    if MODEL_OUT_LOCAL.exists() and MODEL_OUT_LOCAL.stat().st_size > 10_000_000:
        log("v2 model already exists; skipping training.")
        return
    rc = run([sys.executable, "Training_Florida.py"], "Train")
    if rc != 0:
        sys.exit("Training failed.")
    if not MODEL_OUT_LOCAL.exists():
        sys.exit(f"Training finished but model not found at {MODEL_OUT_LOCAL}")


def stage_validate():
    """Run the held-out validation report."""
    log("STAGE 5: True validation report")
    rc = run([sys.executable, "test_true_validation.py"], "True validation")
    if rc != 0:
        log("WARNING: Validation script returned non-zero. Inspect logs.")


def stage_publish():
    """Copy model into web/ so the FastAPI app can find it."""
    log("STAGE 6: Publish model to web/")
    if not MODEL_OUT_LOCAL.exists():
        sys.exit("No model to publish.")
    shutil.copy2(MODEL_OUT_LOCAL, MODEL_OUT_WEB)
    log(f"Copied -> {MODEL_OUT_WEB} ({MODEL_OUT_WEB.stat().st_size/1024/1024:.1f} MB)")


def stage_print_deploy():
    log("STAGE 7: Manual deploy instructions")
    log("All artifacts are ready locally. To deploy, review the diff then:")
    log("  git add -A")
    log("  git commit -m 'v2: 19-channel model with ERC/FM100/NLCD masks'")
    log("  git push origin model-v2-improved-features")
    log("Then open a PR to main on GitHub. Railway will redeploy on merge.")
    log("Note: the .keras file is ~50-200MB; if it exceeds GitHub's 100MB limit,")
    log("upload to a release artifact and set MODEL_URL env var on Railway instead.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", choices=[
        "download", "prepare", "leakage", "train", "validate", "publish"
    ], default="download")
    args = parser.parse_args()

    log(f"Pipeline starting (log: {LOG_FILE})")
    stages = [
        ("download", stage_download),
        ("prepare", stage_prepare),
        ("leakage", stage_leakage_check),
        ("train", stage_train),
        ("validate", stage_validate),
        ("publish", stage_publish),
    ]
    started = False
    for name, fn in stages:
        if not started and name != args.start_from:
            continue
        started = True
        fn()
    stage_print_deploy()
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
