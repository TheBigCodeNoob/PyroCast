"""v11a (free, no export): test cheap engineered features on v10b — day-of-week,
weekend (humans burn on weekends), smooth seasonality. Judged on blocked space+time.
Watch: if cyclical-doy helps a LOT it implies residual season leakage (we matched on
MONTH, not exact day) — flag it. day-of-week helping is a genuine 'when' signal."""
import glob, datetime, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in glob.glob('Training Data Florida/v10b/*.csv')], ignore_index=True).dropna().reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
# cheap new features
dow = [(datetime.date(int(y), 1, 1) + datetime.timedelta(days=int(d) - 1)).weekday() for y, d in zip(df.year, df.doy)]
df['dow'] = dow
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['doy_sin'] = np.sin(2 * np.pi * df.doy / 365.0)
df['doy_cos'] = np.cos(2 * np.pi * df.doy / 365.0)
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy']
BASE = [c for c in df.columns if c not in META + ['dow', 'is_weekend', 'doy_sin', 'doy_cos']]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values


def st(cols):
    X = np.nan_to_num(df[cols].values.astype('float32')); oof = np.full(len(y), np.nan)
    for tg, eg in GroupKFold(5).split(X, y, block):
        tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
        if len(tr) < 100 or y[tr].sum() < 20:
            continue
        m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); return roc_auc_score(y[mask], oof[mask])


# weekend rate check (is there a real human-fire weekend effect?)
print('weekend share: fire={:.3f} neg={:.3f}'.format(df[y == 1].is_weekend.mean(), df[y == 0].is_weekend.mean()))
print('  (human fires only: {:.3f})'.format(df[(df.cause == 1)].is_weekend.mean()))
b = st(BASE); print(f'\nbaseline (v10b feats)           {b:.4f}')
print(f'+ day-of-week                   {st(BASE+["dow"]):.4f}')
print(f'+ is_weekend                    {st(BASE+["is_weekend"]):.4f}')
print(f'+ cyclical doy (watch leakage)  {st(BASE+["doy_sin","doy_cos"]):.4f}')
print(f'+ ALL cheap                     {st(BASE+["dow","is_weekend","doy_sin","doy_cos"]):.4f}')
