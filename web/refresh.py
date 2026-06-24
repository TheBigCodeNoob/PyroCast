"""Scheduled refresh: pull the latest weather/satellite data, recompute Florida fire risk for
the dense grid, and update the files the web app serves. Run this every 6-12 hours (cron /
Windows Task Scheduler). Steps: export today's FL grid from Earth Engine -> download ->
score -> write web/data/. The web app just reads the precomputed files, so it stays instant.
  python web/refresh.py            # uses today's date
  python web/refresh.py 2026-04-15 # specific date
"""
import sys, subprocess, time, glob, pathlib, datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONDA = r'C:\Users\nonna\miniconda3\Scripts\conda.exe'
ENV = 'pyrocast_cuda'
GRID_DIR = ROOT / 'Training Data Florida' / 'FLgrid'
DATE = next((a for a in sys.argv[1:] if '-' in a), None)


def run(cmd, **kw):
    print('>', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


def main():
    t0 = time.time()
    # 1. trigger the Earth Engine export for the target date
    run([CONDA, 'run', '--no-capture-output', '-n', ENV, 'python', 'Dataget_fl_grid.py'] + ([DATE] if DATE else []))
    # 2. download as batches finish (robust by-name downloader), up to ~2h
    for _ in range(40):
        run([CONDA, 'run', '--no-capture-output', '-n', ENV, 'python', 'dl_by_name.py', 'Fire_FLgrid_', str(GRID_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n = len(glob.glob(str(GRID_DIR) + '/*.csv'))
        print(f'  downloaded {n}/18 grid batches', flush=True)
        if n >= 16:
            break
        time.sleep(180)
    # 3. score the grid -> web/data/
    run(['python', str(ROOT / 'web' / 'compute_risk.py')])
    print(f'refresh done in {int(time.time()-t0)}s at {datetime.datetime.utcnow().isoformat()}Z', flush=True)


if __name__ == '__main__':
    main()
