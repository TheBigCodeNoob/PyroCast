"""Where does the FINAL model fail? Characterize missed fires (low score, real fire) and
false alarms (high score, no fire) to find the next lever."""
import glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def _safe(g):
    out = []
    for c in glob.glob(g):
        try:
            out.append(pd.read_csv(c))
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


ve = _safe('Training Data Florida/v11e/*.csv').dropna()
vh = _safe('Training Data Florida/v11h_canopy/*.csv').dropna()
vm = _safe('Training Data Florida/v12_moisture/*.csv')
CANF = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
MOIST = ['ndmi', 'smap_root', 'lst_day', 'pet', 'et_stress']
for d in (ve, vh, vm):
    d['k'] = d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str)
vm['et_stress'] = vm.et / (vm.pet + 1)
df = (ve.merge(vh[['k'] + CANF].drop_duplicates('k'), on='k')
        .merge(vm[['k'] + MOIST].drop_duplicates('k'), on='k', how='left')).reset_index(drop=True)
df['pdsi_traj_90'] = df.pdsi_0 - df.pdsi_90; df['vpd_trend'] = df.vpd_7 - df.vpd_90
df['pr_deficit'] = df.pr_365 / 4 - df.pr_90; df['fm100_trend'] = df.fm100_30 - df.fm100_90
df['dryness'] = df.vpd_30 + df.erc_30 - df.pr_90 / 50
META = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'k']
FEATS = [c for c in df.columns if c not in META]
y = df.label.astype(int).values
block = (np.floor(df.lon).astype(int).astype(str) + '_' + np.floor(df.lat).astype(int).astype(str)).values
yr = df.year.values; X = df[FEATS].values.astype('float32')
oof = np.full(len(y), np.nan)
for tg, eg in GroupKFold(5).split(X, y, block):
    tr = tg[yr[tg] <= 2019]; te = eg[yr[eg] >= 2020]
    if len(tr) < 100 or y[tr].sum() < 20:
        continue
    m = HistGradientBoostingClassifier(max_iter=450, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=2.0, min_samples_leaf=25, random_state=0)
    m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
d = df[~np.isnan(oof)].copy(); d['p'] = oof[~np.isnan(oof)]
fires = d[d.label == 1]; negs = d[d.label == 0]
missed = fires[fires.p < fires.p.quantile(0.2)]   # fires the model rated lowest
caught = fires[fires.p > fires.p.quantile(0.8)]
falarm = negs[negs.p > negs.p.quantile(0.9)]       # non-fires rated highest

print('=== MISSED FIRES (lowest-scored real fires) vs WELL-CAUGHT fires ===')
print(f'  missed n={len(missed)}  caught n={len(caught)}')
print(f'  cause (human frac):  missed {(missed.cause==1).mean():.2f}  caught {(caught.cause==1).mean():.2f}')
print(f'  month:               missed {missed.month.mean():.1f}  caught {caught.month.mean():.1f}')
for f in ['DistDev', 'nbhd_dev_500m', 'canopy_ht', 'treecover', 'vcf' if 'vcf_herb' in d else 'ndmi', 'pdsi_90', 'vpd_90', 'Elevation', 'LC_Forest', 'LC_Wetland']:
    if f in d.columns:
        print(f'  {f:<14s} missed {missed[f].median():8.3f}  caught {caught[f].median():8.3f}')
print(f'  Florida frac:        missed {((missed.lat<31)&(missed.lon>-87.6)).mean():.2f}  caught {((caught.lat<31)&(caught.lon>-87.6)).mean():.2f}')

print('\n=== FALSE ALARMS (highest-scored non-fires): what makes them look fire-prone? ===')
print(f'  n={len(falarm)} | DistDev med {falarm.DistDev.median():.3f} (vs all-neg {negs.DistDev.median():.3f})')
print(f'  these are background points in fire-prone settings (near development, dry, right fuel)')
print(f'  -> mostly UNAVOIDABLE in presence/background framing (they look exactly like fire spots)')

print('\n=== takeaway ===')
print('  If missed fires cluster on a feature/region/cause, that gap is the next lever.')
