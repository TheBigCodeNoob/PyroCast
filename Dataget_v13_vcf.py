"""
v13c: MODIS Vegetation Continuous Fields (MOD44B) — continuous fuel decomposition,
year-matched. % tree cover, % non-tree vegetation (grass/shrub = fine fast-igniting fuels),
% non-vegetated (bare = non-fuel). Distinct from canopy height, NLCD categories, and NDMI.
  python Dataget_v13_vcf.py --test | python Dataget_v13_vcf.py
"""
import ee, sys
from Dataget_v11e import POS, pos_fc, negatives, POS_BATCHES, NEG_BATCHES

EXPORT_FOLDER = 'Fire_v13_vcf'
COLUMNS = ['lon', 'lat', 'label', 'cause', 'year', 'doy', 'vcf_tree', 'vcf_herb', 'vcf_bare', 'vcf_herb_2km']
VCF = ee.ImageCollection('MODIS/061/MOD44B')


def extra(feature):
    pt = feature.geometry(); td = ee.Date(feature.get('target_time')); yr = td.get('year')
    col = VCF.filterDate(ee.Date.fromYMD(yr, 1, 1).advance(-1, 'year'), ee.Date.fromYMD(yr, 12, 31))
    img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sort('system:time_start', False).first(),
          ee.Image.constant([0, 0, 0]).rename(['Percent_Tree_Cover', 'Percent_NonTree_Vegetation', 'Percent_NonVegetated'])))
    tree = img.select('Percent_Tree_Cover').min(100)
    herb = img.select('Percent_NonTree_Vegetation').min(100)
    bare = img.select('Percent_NonVegetated').min(100)

    def at(im, band):
        return im.rename(band).reduceRegion(ee.Reducer.first(), pt, 250).get(band)
    coords = pt.coordinates()
    return ee.Feature(pt, {
        'lon': coords.get(0), 'lat': coords.get(1), 'label': feature.get('label'), 'cause': feature.get('cause'),
        'year': yr, 'doy': td.getRelative('day', 'year'),
        'vcf_tree': at(tree, 'vcf_tree'), 'vcf_herb': at(herb, 'vcf_herb'), 'vcf_bare': at(bare, 'vcf_bare'),
        'vcf_herb_2km': herb.rename('h').reduceRegion(ee.Reducer.mean(), pt.buffer(2000), 250).get('h'),
    })


def run_test():
    one = ee.Feature(pos_fc(0, 5).map(extra).first()).toDictionary().getInfo()
    print('sample:', {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in one.items()})
    print('missing:', [c for c in COLUMNS if c not in one])


def run_export():
    chunk = (len(POS) + POS_BATCHES - 1) // POS_BATCHES
    for i in range(POS_BATCHES):
        ee.batch.Export.table.toDrive(collection=pos_fc(i * chunk, (i + 1) * chunk).map(extra), description=f'Fire_v13vcf_pos_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v13vcf_pos_{i+1}')
    neg = negatives(303)
    for i in range(NEG_BATCHES):
        lo, hi = i / NEG_BATCHES, (i + 1) / NEG_BATCHES
        nb = neg.filter(ee.Filter.And(ee.Filter.gte('part', lo), ee.Filter.lt('part', hi)))
        ee.batch.Export.table.toDrive(collection=nb.map(extra), description=f'Fire_v13vcf_neg_{i+1}', folder=EXPORT_FOLDER, fileFormat='CSV', selectors=COLUMNS).start()
        print(f'  submitted Fire_v13vcf_neg_{i+1}')
    print(f"-> '{EXPORT_FOLDER}'")


if __name__ == '__main__':
    (run_test if '--test' in sys.argv else run_export)()
