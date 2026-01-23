import pandas as pd

df = pd.read_csv('data/raw/fire_catalog.csv')
df_2010 = df[df['year'] >= 2010]

print(f'Total perimeters (2010+): {len(df_2010)}')

# Check duplicates
dups = df_2010.groupby('fire_id').size()
print(f'\nFires with multiple perimeters: {(dups > 1).sum()} fires')
print(f'Max perimeters for one fire: {dups.max()}')
print(f'Avg perimeters per fire: {dups.mean():.1f}')

# What if we keep ALL perimeters as separate training samples?
df_1k = df_2010[df_2010['acres'] >= 1000]
df_1500 = df_2010[df_2010['acres'] >= 1500]

print(f'\n1000+ acres:')
print(f'  Total perimeters (training samples): {len(df_1k)}')
print(f'  Unique fires: {df_1k["fire_id"].nunique()}')

print(f'\n1500+ acres:')
print(f'  Total perimeters (training samples): {len(df_1500)}')
print(f'  Unique fires: {df_1500["fire_id"].nunique()}')

print(f'\n\nFINAL CONFIGURATION (NO DEDUPLICATION):')
print(f'=' * 60)
print(f'Fire perimeter snapshots to process: {len(df_1k)}')
print(f'From {df_1k["fire_id"].nunique()} unique fire events')
print(f'')
print(f'Training samples:')
print(f'  Base: {len(df_1k)}')
print(f'  After 4x augmentation: ~{len(df_1k) * 4:,}')
print(f'  After 8x augmentation: ~{len(df_1k) * 8:,}')
print(f'')
print(f'Estimated processing time:')
print(f'  {len(df_1k)} snapshots × 25 sec = {len(df_1k) * 25 / 3600:.1f} hours')
print(f'=' * 60)
