"""Reproduce the Florida ensemble headline on spatially and temporally held-out data."""
import glob
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')


def load_data():
    def read(pattern):
        files = glob.glob(pattern)
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    base = read('Training Data Florida/v11e/*.csv').dropna()
    canopy = read('Training Data Florida/v11h_canopy/*.csv').dropna()
    moisture = read('Training Data Florida/v12_moisture/*.csv')
    human = read('Training Data Florida/v13_human/*.csv')
    canopy_features = ['canopy_ht', 'treecover', 'canopy_ht_2km', 'treecover_2km']
    moisture_features = ['ndmi', 'smap_root', 'lst_day', 'pet']

    for frame in (base, canopy, moisture, human):
        frame['key'] = frame.lon.round(5).astype(str) + '_' + frame.lat.round(5).astype(str)
    moisture['et_stress'] = moisture.et / (moisture.pet + 1)
    data = (base.merge(canopy[['key'] + canopy_features].drop_duplicates('key'), on='key')
            .merge(moisture[['key'] + moisture_features + ['et_stress']].drop_duplicates('key'),
                   on='key', how='left')
            .merge(human[['key', 'built']].dropna().drop_duplicates('key'), on='key', how='left'))
    data = data[(data.lat < 31.0) & (data.lon > -87.6) & (data.lon < -79.8)].reset_index(drop=True)
    data['pdsi_traj_90'] = data.pdsi_0 - data.pdsi_90
    data['vpd_trend'] = data.vpd_7 - data.vpd_90
    data['pr_deficit'] = data.pr_365 / 4 - data.pr_90
    data['fm100_trend'] = data.fm100_30 - data.fm100_90
    data['dryness'] = data.vpd_30 + data.erc_30 - data.pr_90 / 50
    meta = ['lon', 'lat', 'label', 'cause', 'month', 'year', 'doy', 'key']
    features = [column for column in data.columns if column not in meta]
    return data, features


def block_bootstrap_interval(y, scores, blocks, n_bootstrap=1000, seed=0):
    _, inverse = np.unique(blocks, return_inverse=True)
    n_blocks = inverse.max() + 1
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        weights = counts[inverse]
        if weights[y == 1].sum() and weights[y == 0].sum():
            estimates.append(roc_auc_score(y, scores, sample_weight=weights))
    if not estimates:
        raise ValueError('No valid bootstrap samples contained both classes.')
    return np.percentile(estimates, [2.5, 97.5])


def main():
    data, features = load_data()
    y = data.label.astype(int).to_numpy()
    year = data.year.to_numpy()
    blocks = (np.floor(data.lon / 0.6).astype(int).astype(str) + '_'
              + np.floor(data.lat / 0.6).astype(int).astype(str)).to_numpy()
    X = np.nan_to_num(data[features].to_numpy(dtype='float32'))
    model_factories = {
        'hgb': lambda: HistGradientBoostingClassifier(
            max_iter=450, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=2.0, min_samples_leaf=25, random_state=0),
        'rf': lambda: RandomForestClassifier(
            n_estimators=300, max_features='sqrt', min_samples_leaf=3,
            n_jobs=-1, random_state=0),
        'lgb': lambda: LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            reg_lambda=3.0, min_child_samples=30, random_state=0, verbose=-1),
    }
    predictions = {name: np.full(len(y), np.nan) for name in model_factories}
    for fold, (train_groups, test_groups) in enumerate(
            GroupKFold(5).split(X, y, blocks), start=1):
        train = train_groups[year[train_groups] <= 2019]
        test = test_groups[year[test_groups] >= 2020]
        if (len(train) < 100 or y[train].sum() < 20
                or (y[train] == 0).sum() < 20 or len(test) == 0):
            print(f'Fold {fold}: skipped (train={len(train)}, test={len(test)})')
            continue
        print(f'Fold {fold}: train={len(train)}, held-out future={len(test)}')
        for name, make_model in model_factories.items():
            model = make_model()
            model.fit(X[train], y[train])
            predictions[name][test] = model.predict_proba(X[test])[:, 1]

    mask = ~np.isnan(predictions['hgb'])
    if not mask.any() or len(np.unique(y[mask])) != 2:
        raise ValueError('No valid held-out predictions for both classes.')
    test_y = y[mask]
    ensemble = np.mean([predictions[name][mask] for name in model_factories], axis=0)
    test_blocks = blocks[mask]
    interval = block_bootstrap_interval(test_y, ensemble, test_blocks)
    print(f'Rows={len(data)}, fires={int(y.sum())}, features={len(features)}')
    print(f'Held-out n={int(mask.sum())}, fires={int(test_y.sum())}, '
          f'spatial blocks={len(np.unique(test_blocks))}')
    for name in model_factories:
        print(f'{name} blocked space+time AUROC={roc_auc_score(test_y, predictions[name][mask]):.4f}')
    print(f'Ensemble blocked space+time AUROC={roc_auc_score(test_y, ensemble):.4f}')
    print(f'Ensemble 95% spatial-block bootstrap interval=[{interval[0]:.4f}, {interval[1]:.4f}]')


if __name__ == '__main__':
    main()
