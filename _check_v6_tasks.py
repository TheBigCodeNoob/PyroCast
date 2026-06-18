import ee
ee.Initialize(project='gleaming-glass-426122-k0')
from collections import Counter
ts = [t['metadata'] for t in ee.data.listOperations()
      if t.get('metadata',{}).get('description','').startswith('Fire_Ignition_v6_')]
print('total v6 tasks:', len(ts))
for t in sorted(ts, key=lambda m: m.get('description','')):
    print(f"  {t.get('description'):<28s} {t.get('state'):<12s}")
print('states:', dict(Counter(t['state'] for t in ts)))
