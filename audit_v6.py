import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.concat([pd.read_csv(c) for c in sorted(glob.glob('Training Data Florida/v6/*.csv'))], ignore_index=True).dropna().reset_index(drop=True)
y = df.label.astype(int).values
cells = (np.round(df.lon*4)/4).astype(str)+'_'+(np.round(df.lat*4)/4).astype(str)
groups = cells.values

# Engineered (same as model)
eps=0.1
df['pdsi_traj_90']=df.pdsi_0-df.pdsi_90; df['pdsi_traj_180']=df.pdsi_0-df.pdsi_180
df['vpd_trend']=df.vpd_7-df.vpd_90; df['erc_trend']=df.erc_7-df.erc_90
df['pr_recent_ratio']=df.pr_30/(df.pr_90+eps); df['pr_deficit']=df.pr_365/4.0-df.pr_90
df['fm100_trend']=df.fm100_30-df.fm100_90; df['dryness']=df.vpd_30+df.erc_30-df.pr_90/50.0
SPATIAL=['Elevation','Pop_Density','LC_Forest','LC_Shrub','LC_Grass','LC_Pasture','LC_Wetland','LC_Crop','LC_Developed','NDVI','NDMI']
TEMPORAL=[c for c in df.columns if c not in (['lon','lat','label']+SPATIAL)]

print("=== Pop_Density distribution (degenerate-feature check) ===")
for lab,nm in [(1,'FIRE'),(0,'NOFIRE')]:
    s=df.Pop_Density[df.label==lab]
    print(f"  {nm}: mean={s.mean():.2f} median={s.median():.2f} %zero={(s==0).mean()*100:.1f}%  p90={s.quantile(.9):.1f}")
print("  (overlap matters: if both have similar spread, it's real signal, not a flag)")

def hgb(): return HistGradientBoostingClassifier(max_iter=600,learning_rate=0.03,max_leaf_nodes=63,l2_regularization=2.0,min_samples_leaf=25,early_stopping=False,random_state=0)
def cv_auc(cols):
    X=np.nan_to_num(df[cols].values.astype(np.float32))
    oof=np.zeros(len(y))
    for a,b in GroupKFold(5).split(X,y,groups):
        m=hgb(); m.fit(X[a],y[a]); oof[b]=m.predict_proba(X[b])[:,1]
    return roc_auc_score(y,oof)

print("\n=== Robustness: drop the dominant feature, does skill hold? (spatial-CV) ===")
print(f"  FULL                         {cv_auc(SPATIAL+TEMPORAL):.4f}")
print(f"  FULL minus Pop_Density       {cv_auc([c for c in SPATIAL+TEMPORAL if c!='Pop_Density']):.4f}")
print(f"  WHERE only                   {cv_auc(SPATIAL):.4f}")
print(f"  WHERE minus Pop_Density      {cv_auc([c for c in SPATIAL if c!='Pop_Density']):.4f}")
print(f"  Pop_Density ALONE            {cv_auc(['Pop_Density']):.4f}")
print(f"  TEMPORAL only                {cv_auc(TEMPORAL):.4f}")

# Seasonality proxy: tmmx_7 (recent max temp) separates fire-season from off-season
print("\n=== Seasonality proxy (is WHEN just 'is it fire season'?) ===")
for lab,nm in [(1,'FIRE'),(0,'NOFIRE')]:
    print(f"  {nm}: tmmx_7 mean={df.tmmx_7[df.label==lab].mean():.1f}K  erc_30={df.erc_30[df.label==lab].mean():.1f}  fm100_90={df.fm100_90[df.label==lab].mean():.2f}  pr_90={df.pr_90[df.label==lab].mean():.1f}mm")
