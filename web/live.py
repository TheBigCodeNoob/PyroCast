"""PyroCast LIVE: background jobs that make the deployed tool current and self-auditing.

  REFRESH (every 72h): pull current conditions from GEE for the ~9.4k known land cells,
    re-score, and regenerate the served map via compute_risk. Fail-safe: any error leaves the
    last good map in place.
  AUDIT (daily): watch NASA FIRMS for new significant Florida fire events; for each, rebuild
    the risk map as it stood 7 days BEFORE the fire and record where the ignition ranked.
    Results accumulate in web/data/scorecard.json (+ annotated maps in web/data/audit/) and
    are served at /scorecard. Idempotent: after a redeploy it re-audits from START_DATE.

EE auth: GOOGLE_APPLICATION_CREDENTIALS (key-file path) if set, else any env var whose value
is a service-account JSON blob (written to a temp file). No creds -> jobs idle harmlessly.
"""
import os, sys, json, math, time, tempfile, datetime, threading, traceback
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import pyrocast_core as core            # shares the app's single in-memory model bundle

DATA = os.path.join(HERE, 'data')
AUDIT_DIR = os.path.join(DATA, 'audit')
STATE = os.path.join(DATA, 'live_state.json')
SCORE = os.path.join(DATA, 'scorecard.json')
START_DATE = '2026-10-16'          # the system's public birthday: audit every event from here on
REFRESH_EVERY_H = 72
AUDIT_EVERY_H = 24
MIN_DET = 6                        # FIRMS detections for an event to count as "significant"
LEAD_DAYS = 7                      # audit scores the map as it stood this many days pre-fire
MAX_AUDITS_PER_CYCLE = 4
CHUNK = 350
FL = (-87.7, 24.5, -79.8, 31.05)   # lon0, lat0, lon1, lat1 (must match the served grid)

_G = None                          # lazy module handle for Dataget_fl_grid
_warned = False
_log = lambda *a: print('[live]', *a, flush=True)


# ---------------- EE auth (lazy, fail-safe) ----------------
def _ee_init():
    global _G, _warned
    if _G is not None:
        return True
    try:
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            for k, v in os.environ.items():
                if isinstance(v, str) and '"private_key"' in v and 'service_account' in v:
                    p = os.path.join(tempfile.gettempdir(), 'gee_key.json')
                    with open(p, 'w') as fh:
                        fh.write(v)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = p
                    _log(f'using service-account JSON from env var {k}')
                    break
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            if not _warned:
                _log('no GEE credentials found; live jobs idle'); _warned = True
            return False
        # ee.Initialize() only auto-uses PERSISTENT (interactive) credentials, which don't exist
        # on Railway — a service-account key must be passed explicitly. Bind it into
        # ee.Initialize so the Dataget modules' own import-time ee.Initialize(project=...) works.
        import ee, functools
        key_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        email = json.load(open(key_path)).get('client_email')
        if email:
            creds = ee.ServiceAccountCredentials(email, key_path)
            ee.Initialize(creds, project='gleaming-glass-426122-k0')
            ee.Initialize = functools.partial(ee.Initialize, creds)
            _log(f'EE initialized as {email}')
        # Dataget_fl_grid parses argv at import; Dataget_v11e opens fpafod_se.csv relative to CWD
        argv, cwd = sys.argv, os.getcwd()
        sys.argv = [sys.argv[0], '2022-01-01']
        try:
            os.chdir(ROOT)
            import Dataget_fl_grid as G
            _G = G
        finally:
            sys.argv = argv; os.chdir(cwd)
        return True
    except Exception as e:
        _log(f'EE init failed: {type(e).__name__}: {e}'); return False


# ---------------- extraction helpers ----------------
def _ms(date):
    y, m, d = (int(x) for x in str(date).split('-'))
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _extract(points, target_ms):
    """Full feature stack for (lon,lat) points at target time via chunked getDownloadURL."""
    import ee
    frames, nchunks = [], (len(points) + CHUNK - 1) // CHUNK
    for i in range(0, len(points), CHUNK):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]),
             {'label': 0, 'cause': -1, 'target_time': target_ms}) for lo, la in points[i:i + CHUNK]])
        feats = fc.map(_G.grid_feat).filter(ee.Filter.notNull(['NDVI']))
        for attempt in range(3):
            try:
                url = feats.getDownloadURL(filetype='CSV', selectors=_G.COLUMNS)
                frames.append(pd.read_csv(url)); break
            except Exception as e:
                _log(f'  chunk {i//CHUNK+1} attempt {attempt+1}: {type(e).__name__}: {str(e)[:90]}')
                time.sleep(20)
        if (i // CHUNK) % 5 == 4 or i + CHUNK >= len(points):
            _log(f'  extracted chunk {i//CHUNK+1}/{nchunks}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _land_points(step_mult=1):
    """The known FL land cells (from the served grid). step_mult=3 -> ~1k-cell coarse lattice."""
    g = pd.read_csv(os.path.join(DATA, 'fl_grid_full.csv'), usecols=['lon', 'lat'])
    if step_mult > 1:
        ix = (np.round((g.lon + 87.7) / 0.04) % step_mult == 0) & \
             (np.round((g.lat - 24.5) / 0.04) % step_mult == 0)
        g = g[ix]
    return list(zip(g.lon.round(3), g.lat.round(3)))


# ---------------- job 1: live refresh ----------------
def refresh_once():
    if not _ee_init():
        return False
    target = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
    pts = _land_points()
    _log(f'REFRESH: extracting {len(pts)} land cells @ {target}')
    df = _extract(pts, _ms(target))
    if len(df) < 0.8 * len(pts):
        _log(f'REFRESH aborted: only {len(df)} rows extracted; keeping last good map'); return False
    tmp = tempfile.mkdtemp(prefix='flgrid_')
    df.to_csv(os.path.join(tmp, 'live_grid.csv'), index=False)
    import compute_risk
    compute_risk.main(tmp)
    _log(f'REFRESH done: map regenerated for conditions as of {target}')
    return True


# ---------------- job 2: FIRMS watch + self-audit ----------------
def _firms_day(day):
    import ee
    d0 = ee.Date(str(day))
    mask = ee.ImageCollection('FIRMS').filterDate(d0, d0.advance(1, 'day')).select('T21').max().gt(0)
    pts = (ee.Image.pixelLonLat().updateMask(mask)
           .sample(region=ee.Geometry.Rectangle([FL[0], FL[1], FL[2], FL[3]]), scale=1000, geometries=False)
           .getInfo().get('features', []))
    return [(round(p['properties']['longitude'], 3), round(p['properties']['latitude'], 3)) for p in pts]


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1)


def audit_once():
    if not _ee_init():
        return False
    os.makedirs(AUDIT_DIR, exist_ok=True)
    sc = _load(SCORE, {'start': START_DATE, 'events': []})
    st = _load(STATE, {})
    today = datetime.date.today()
    day = datetime.date.fromisoformat(st.get('firms_day', START_DATE))
    open_ev = st.get('open_events', [])
    while day < today:                                     # catch up day by day (idempotent)
        try:
            det = _firms_day(day)
        except Exception as e:
            _log(f'FIRMS {day}: {type(e).__name__}: {str(e)[:90]}'); break
        for lon, lat in det:
            hit = next((ev for ev in open_ev
                        if abs(ev['lon'] - lon) < 0.08 and abs(ev['lat'] - lat) < 0.08
                        and (day - datetime.date.fromisoformat(ev['last'])).days <= 3), None)
            if hit:
                hit['n'] += 1; hit['last'] = str(day)
            else:
                open_ev.append({'lon': lon, 'lat': lat, 'date': str(day), 'last': str(day), 'n': 1})
        if det:
            _log(f'FIRMS {day}: {len(det)} detections, {len(open_ev)} open events')
        day += datetime.timedelta(days=1)
        st['firms_day'] = str(day)
    # events quiet >3 days are over: audit the significant ones (keep on failure, retry next cycle)
    keep, audited = [], 0
    for ev in open_ev:
        over = (today - datetime.date.fromisoformat(ev['last'])).days > 3
        eid = f"{ev['date']}_{ev['lon']}_{ev['lat']}"
        if not over:
            keep.append(ev); continue
        if ev['n'] < MIN_DET or any(e['id'] == eid for e in sc['events']):
            continue                                       # minor event or already audited: drop
        if audited >= MAX_AUDITS_PER_CYCLE:
            keep.append(ev); continue
        try:
            _audit_event(ev, sc); audited += 1
            _save(SCORE, sc)
        except Exception as e:
            _log(f'audit of {eid}: {type(e).__name__}: {e}')
            keep.append(ev)
    st['open_events'] = keep
    _save(STATE, st)
    return True


def _audit_event(ev, sc):
    import calibrate
    pre = (datetime.date.fromisoformat(ev['date']) - datetime.timedelta(days=LEAD_DAYS)).isoformat()
    _log(f"AUDIT: event {ev['date']} @ {ev['lat']},{ev['lon']} (n={ev['n']}) -> map as of {pre}")
    pts = _land_points(step_mult=3)                        # ~1k-cell statewide reference grid
    tms = _ms(pre)
    grid = _extract(pts, tms)
    fp = _extract([(ev['lon'], ev['lat'])], tms)
    if len(grid) < 300 or len(fp) == 0:
        raise RuntimeError(f'extraction too thin (grid={len(grid)}, fire={len(fp)})')
    gs = core.score(grid); fs = float(core.score(fp)[0])
    fpct = float((gs < fs).mean())
    tier = int(np.searchsorted(calibrate.TIER_PCT, fpct, side='right'))
    img = f"ev_{ev['date']}_{ev['lat']}_{ev['lon']}.png"
    try:
        _render_audit(grid, gs, ev, pre, fpct, calibrate.TIER_NAMES[tier], os.path.join(AUDIT_DIR, img))
    except Exception as e:
        _log(f'  render failed ({e}); recording score anyway'); img = None
    sc['events'].append({'id': f"{ev['date']}_{ev['lon']}_{ev['lat']}", 'date': ev['date'],
                         'lon': ev['lon'], 'lat': ev['lat'], 'detections': ev['n'],
                         'pre_date': pre, 'pctile': round(fpct * 100, 1),
                         'tier': calibrate.TIER_NAMES[tier], 'img': img})
    _log(f"  -> ignition ranked {fpct*100:.0f}th percentile ({calibrate.TIER_NAMES[tier]})")


def _render_audit(grid, gs, ev, pre, fpct, tier, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
    import calibrate
    pct = pd.Series(gs).rank(pct=True).values
    W = 600; H = int(W * (FL[3] - FL[1]) / (FL[2] - FL[0]))
    gx, gy = np.meshgrid(np.linspace(FL[0], FL[2], W), np.linspace(FL[1], FL[3], H))
    P = grid[['lon', 'lat']].values
    gz = griddata(P, pct, (gx, gy), method='linear').ravel()
    dd, _ = cKDTree(P).query(np.column_stack([gx.ravel(), gy.ravel()]))
    gz[dd > 0.15] = np.nan
    gz = gz.reshape(H, W)
    valid = ~np.isnan(gz)
    land = binary_opening(binary_closing(valid, np.ones((3, 3)), iterations=2), np.ones((3, 3)), iterations=2)
    w = gaussian_filter(valid.astype(float), 2.0)
    sm = gaussian_filter(np.where(valid, gz, 0), 2.0) / np.where(w > 1e-3, w, 1)
    sm[~land] = np.nan
    cmap = ListedColormap(calibrate.TIER_COLORS)
    norm = BoundaryNorm([0.0] + calibrate.TIER_PCT + [1.0001], cmap.N)
    rgba = cmap(norm(np.clip(np.nan_to_num(sm, nan=0.0), 0, 1)))
    rgba[np.isnan(sm)] = (0.05, 0.07, 0.09, 1.0)
    fig, ax = plt.subplots(figsize=(5.4, 5.6), facecolor='#0e1116')
    ax.imshow(rgba, extent=[FL[0], FL[2], FL[1], FL[3]], origin='lower')
    ax.plot(ev['lon'], ev['lat'], marker='*', ms=20, mfc='white', mec='black', mew=1.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('#0e1116')
    ax.set_title(f"fire {ev['date']}  ·  map as of {pre}\nignition ranked {fpct*100:.0f}th percentile ({tier})",
                 color='white', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor='#0e1116'); plt.close(fig)


# ---------------- job 3: daily pulse snapshots + change reports ----------------
HIST_DIR = os.path.join(DATA, 'history')
PULSE_LAG = 10                      # conditions date = today - lag (same availability lag as refresh)
PULSE_FLOOR = '2026-09-25'          # earliest conditions date to (re)build - gives ~3 weeks of timeline
MAX_PULSE_PER_TICK = 2              # bound backfill work per scheduler tick (~8 min each)
HIGH_PCT = 0.75                     # High tier lower edge (calibrate.TIER_PCT[2])
HYST = 0.03                         # hysteresis so cells hovering at the edge don't flicker in reports
# (column, plain-english label, format)
DRIVERS = [('vpd_7', 'air dryness (VPD)', 'pct'), ('erc_7', 'fire-energy index (ERC)', 'pct'),
           ('fm100_30', 'dead-fuel moisture', 'pct'), ('pr_30', 'rain last 30 days', 'mm'),
           ('pdsi_0', 'drought index (PDSI)', 'abs'), ('smap_surface', 'soil moisture', 'pct'),
           ('ndmi', 'vegetation moisture (NDMI)', 'abs'), ('lst_day', 'land-surface temp', 'degC'),
           ('rmin_30', 'min humidity (30d)', 'pct')]
CITIES = [('Pensacola', -87.22, 30.42), ('Crestview', -86.57, 30.76), ('Panama City', -85.66, 30.16),
          ('Marianna', -85.23, 30.77), ('Apalachicola', -85.00, 29.73), ('Tallahassee', -84.28, 30.44),
          ('Perry', -83.58, 30.12), ('Live Oak', -82.98, 30.30), ('Lake City', -82.64, 30.19),
          ('Gainesville', -82.32, 29.65), ('Starke', -82.11, 29.94), ('Palatka', -81.64, 29.65),
          ('Jacksonville', -81.66, 30.33), ('St. Augustine', -81.31, 29.90), ('Ocala', -82.14, 29.19),
          ('Inverness', -82.33, 28.84), ('Brooksville', -82.39, 28.56), ('Leesburg', -81.88, 28.81),
          ('Deltona', -81.26, 28.90), ('Daytona Beach', -81.02, 29.21), ('Titusville', -80.81, 28.61),
          ('Orlando', -81.38, 28.54), ('Kissimmee', -81.41, 28.29), ('Lakeland', -81.95, 28.04),
          ('Tampa', -82.46, 27.95), ('Bartow', -81.84, 27.90), ('Melbourne', -80.61, 28.08),
          ('Vero Beach', -80.40, 27.64), ('Fort Pierce', -80.33, 27.45), ('Sebring', -81.44, 27.50),
          ('Wauchula', -81.81, 27.55), ('Sarasota', -82.53, 27.34), ('Arcadia', -81.86, 27.22),
          ('Okeechobee', -80.83, 27.24), ('Port St. Lucie', -80.35, 27.27), ('Punta Gorda', -82.05, 26.93),
          ('Clewiston', -80.93, 26.75), ('LaBelle', -81.44, 26.76), ('Belle Glade', -80.67, 26.68),
          ('Fort Myers', -81.87, 26.64), ('Immokalee', -81.42, 26.42), ('West Palm Beach', -80.05, 26.71),
          ('Naples', -81.79, 26.14), ('Fort Lauderdale', -80.14, 26.12), ('Miami', -80.19, 25.76),
          ('Homestead', -80.48, 25.47), ('Key West', -81.78, 24.56)]


def _place(lon, lat):
    """Human-readable location: 'around Ocala' or '23 km NE of Arcadia'."""
    name, clon, clat = min(CITIES, key=lambda c: math.hypot((lon - c[1]) * 88.0, (lat - c[2]) * 111.0))
    dkm = math.hypot((lon - clon) * 88.0, (lat - clat) * 111.0)
    if dkm < 9:
        return f'around {name}'
    ang = (math.degrees(math.atan2(lon - clon, lat - clat)) + 360) % 360
    comp = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((ang + 22.5) // 45) % 8]
    return f'{dkm:.0f} km {comp} of {name}'


def _clusters(cells):
    """Group (lon,lat) cells into contiguous areas (8-neighborhood on the 0.12-deg lattice)."""
    todo, out = set(cells), []
    while todo:
        seed = todo.pop(); grp = [seed]; queue = [seed]
        while queue:
            x = queue.pop()
            near = [c for c in todo if abs(c[0] - x[0]) < 0.19 and abs(c[1] - x[1]) < 0.19]
            for c in near:
                todo.discard(c); grp.append(c); queue.append(c)
        out.append(grp)
    return sorted(out, key=len, reverse=True)


def _render_pulse(df, path):
    """Transparent-ocean tier raster of the coarse pulse grid (same look as the main map)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
    import calibrate
    pts = df[['lon', 'lat']].values
    lon0, lon1 = float(df.lon.min()), float(df.lon.max())
    lat0, lat1 = float(df.lat.min()), float(df.lat.max())
    W = 700; H = int(round(W * (lat1 - lat0) / (lon1 - lon0)))
    gx, gy = np.meshgrid(np.linspace(lon0, lon1, W), np.linspace(lat0, lat1, H))
    gz = griddata(pts, df['pct'].values, (gx, gy), method='linear').ravel()
    dist, _ = cKDTree(pts).query(np.column_stack([gx.ravel(), gy.ravel()]))
    gz[dist > 0.15] = np.nan
    gz = gz.reshape(H, W)
    valid = ~np.isnan(gz)
    land = binary_opening(binary_closing(valid, np.ones((3, 3)), iterations=2), np.ones((3, 3)), iterations=2)
    w = gaussian_filter(valid.astype(float), sigma=2)
    sm = gaussian_filter(np.where(valid, gz, 0.0), sigma=2) / np.where(w > 1e-3, w, 1.0)
    sm[~land] = np.nan
    sm = sm[::-1, :]
    cmap = ListedColormap(calibrate.TIER_COLORS)
    norm = BoundaryNorm([0.0] + calibrate.TIER_PCT + [1.0001], cmap.N)
    rgba = cmap(norm(np.clip(np.nan_to_num(sm, nan=0.0), 0, 1)))
    rgba[..., 3] = np.where(np.isnan(sm), 0.0, 0.88)
    plt.imsave(str(path), rgba)
    return [[lat0, lon0], [lat1, lon1]]


def _pulse_build(d):
    """Extract + score + render one daily snapshot for conditions date d."""
    _log(f'PULSE: building snapshot for conditions {d}')
    pts = _land_points(step_mult=3)
    df = _extract(pts, _ms(d))
    if len(df) < 300:
        raise RuntimeError(f'extraction too thin ({len(df)} rows)')
    df['score'] = core.score(df)
    df['pct'] = df['score'].rank(pct=True).round(4)
    keep = ['lon', 'lat', 'score', 'pct'] + [c for c, _, _ in DRIVERS if c in df.columns]
    df[keep].to_csv(os.path.join(HIST_DIR, f'pulse_{d}.csv'), index=False)
    bounds = _render_pulse(df, os.path.join(HIST_DIR, f'pulse_{d}.png'))
    _save(os.path.join(HIST_DIR, f'pulse_{d}.json'), {'bounds': bounds, 'n': int(len(df))})
    _log(f'  pulse {d} done ({len(df)} cells)')


def _fmt_driver(label, kind, prev, now):
    dlt = now - prev
    arrow = 'up' if dlt > 0 else 'down'
    if kind == 'pct':
        if abs(prev) < 1e-6:
            return None
        return f'{label} {arrow} {abs(dlt) / abs(prev) * 100:.0f}%'
    if kind == 'mm':
        return f'{label}: {dlt:+.0f} mm'
    if kind == 'degC':
        return f'{label} {dlt:+.1f} C'
    return f'{label} {dlt:+.2f}'


def _area_record(m, grp):
    import calibrate
    sub = m[[(lo, la) in grp for lo, la in zip(m.lon, m.lat)]]
    lon, lat = float(sub.lon.mean()), float(sub.lat.mean())
    tier = int(np.searchsorted(calibrate.TIER_PCT, float(sub['pct_b'].mean()), side='right'))  # tier NOW
    drivers = []
    for col, label, kind in DRIVERS:
        a, b = col + '_a', col + '_b'
        if a not in m.columns or b not in m.columns:
            continue
        prev, now = float(np.nanmean(sub[a])), float(np.nanmean(sub[b]))
        if np.isnan(prev) or np.isnan(now):
            continue
        std = float(np.nanstd(m[b])) or 1e-9
        z = abs(now - prev) / std
        if z < 0.2:
            continue
        txt = _fmt_driver(label, kind, prev, now)
        if txt:
            drivers.append((z, txt))
    drivers.sort(reverse=True)
    return {'name': _place(lon, lat), 'lat': round(lat, 3), 'lon': round(lon, 3),
            'cells': len(grp), 'tier': calibrate.TIER_NAMES[tier],
            'drivers': [t for _, t in drivers[:3]]}


def _pulse_report(d, dprev):
    """Compare consecutive snapshots -> named areas newly High+ / no longer High+, with drivers."""
    a = pd.read_csv(os.path.join(HIST_DIR, f'pulse_{dprev}.csv'))
    b = pd.read_csv(os.path.join(HIST_DIR, f'pulse_{d}.csv'))
    m = a.merge(b, on=['lon', 'lat'], suffixes=('_a', '_b'))
    if len(m) < 300:
        raise RuntimeError('snapshot lattices barely overlap')
    newh = m[(m.pct_b >= HIGH_PCT) & (m.pct_a < HIGH_PCT - HYST)]
    eased = m[(m.pct_a >= HIGH_PCT) & (m.pct_b < HIGH_PCT - HYST)]

    def areas(rows):
        cells = list(zip(rows.lon, rows.lat))
        out = []
        for grp in _clusters(cells):
            # ignore single-cell blips unless the move was big
            if len(grp) < 2:
                sub = m[[(lo, la) in grp for lo, la in zip(m.lon, m.lat)]]
                if abs(float(sub.pct_b.mean()) - float(sub.pct_a.mean())) < 0.12:
                    continue
            out.append(_area_record(m, grp))
        return out[:6]

    # NOTE: tiers are percentile bands, so the COUNT of High+ cells is constant by construction;
    # the meaningful statewide signal is the raw mean score shift.
    rep = {'date': d, 'prev': dprev, 'n_cells': int(len(m)),
           'mean_score_prev': round(float(m.score_a.mean()), 4),
           'mean_score_now': round(float(m.score_b.mean()), 4),
           'new_high': areas(newh), 'eased': areas(eased)}
    _save(os.path.join(HIST_DIR, f'report_{d}.json'), rep)
    _log(f"  report {dprev} -> {d}: {len(rep['new_high'])} new high, {len(rep['eased'])} eased")


def pulse_once():
    if not _ee_init():
        return False
    os.makedirs(HIST_DIR, exist_ok=True)
    target = datetime.date.today() - datetime.timedelta(days=PULSE_LAG)
    floor = datetime.date.fromisoformat(PULSE_FLOOR)
    have = {f[6:16] for f in os.listdir(HIST_DIR) if f.startswith('pulse_') and f.endswith('.csv')}
    missing, day = [], target
    while day >= floor:                              # newest first: today's snapshot lands first
        if str(day) not in have:
            missing.append(str(day))
        day -= datetime.timedelta(days=1)
    for d in missing[:MAX_PULSE_PER_TICK]:
        try:
            _pulse_build(d)
        except Exception as e:
            _log(f'pulse {d}: {type(e).__name__}: {e}')
    dates = sorted({f[6:16] for f in os.listdir(HIST_DIR) if f.startswith('pulse_') and f.endswith('.csv')})
    for i in range(1, len(dates)):
        if not os.path.exists(os.path.join(HIST_DIR, f'report_{dates[i]}.json')):
            try:
                _pulse_report(dates[i], dates[i - 1])
            except Exception as e:
                _log(f'report {dates[i]}: {type(e).__name__}: {e}')
    return True


# ---------------- scheduler ----------------
def _seed_state():
    """Fresh container (no state file): trust the shipped map's own computed date, so a
    redeploy doesn't trigger an immediate 70-min re-extraction of data it already has."""
    st = _load(STATE, {})
    if 'last_refresh' not in st:
        try:
            d = json.load(open(os.path.join(DATA, 'fl_risk.json')))['meta']['computed']
            y, m, dd = (int(x) for x in d.split('-'))
            st['last_refresh'] = datetime.datetime(y, m, dd, 12).timestamp()
            _save(STATE, st)
            _log(f'seeded last_refresh from shipped map (computed {d})')
        except Exception:
            pass


def _loop():
    _seed_state()
    while True:
        try:
            st = _load(STATE, {})
            now = time.time()
            if now - st.get('last_refresh', 0) > REFRESH_EVERY_H * 3600:
                ok = refresh_once()
                st = _load(STATE, {})
                # on failure, retry in ~6h instead of waiting the full cycle
                st['last_refresh'] = now if ok else now - (REFRESH_EVERY_H - 6) * 3600
                _save(STATE, st)
            st = _load(STATE, {})
            if now - st.get('last_audit', 0) > AUDIT_EVERY_H * 3600:
                ok = audit_once()
                st = _load(STATE, {})
                st['last_audit'] = now if ok else now - (AUDIT_EVERY_H - 6) * 3600
                _save(STATE, st)
            pulse_once()      # self-gating: no-op when every snapshot up to today-10d exists
        except Exception:
            _log('scheduler error:\n' + traceback.format_exc())
        time.sleep(1800)


def start():
    if os.environ.get('PYROCAST_LIVE', '1') == '0':
        _log('disabled via PYROCAST_LIVE=0'); return
    threading.Thread(target=_loop, daemon=True, name='pyrocast-live').start()
    _log(f'scheduler started (refresh {REFRESH_EVERY_H}h / audit {AUDIT_EVERY_H}h)')


if __name__ == '__main__':
    # manual one-shots for local testing: python web/live.py refresh|audit|pulse
    if 'refresh' in sys.argv:
        refresh_once()
    if 'audit' in sys.argv:
        audit_once()
    if 'pulse' in sys.argv:
        pulse_once()
