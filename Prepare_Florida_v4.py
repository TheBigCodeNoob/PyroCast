"""
Merge the 12 v4 shards into a location-leakage-free spatial split for CNN training.
Reads gz shards from 'Training Data Florida/v4/' (recursive), splits by location
fingerprint (static channels), writes GZIP-compressed train/val TFRecords to save disk.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import math
import random
import tensorflow as tf
import pathlib as _pl

_REPO = _pl.Path(__file__).resolve().parent
RAW_GLOB = str(_REPO / "Training Data Florida" / "v4" / "**" / "*.tfrecord*")
OUT_TRAIN = str(_REPO / "Training Data Florida" / "Florida_Spatial_Train_v4.tfrecord.gz")
OUT_VAL = str(_REPO / "Training Data Florida" / "Florida_Spatial_Val_v4.tfrecord.gz")
VAL_SPLIT_PCT = 0.15
SEED = 42
BANDS_TO_CHECK = ['Elevation', 'LC_Forest', 'LC_Wetland', 'LC_Open']  # static -> location id
SHUFFLE_BUFFER = 1000


def fingerprint(rec):
    fd = {k: tf.io.VarLenFeature(tf.float32) for k in BANDS_TO_CHECK}
    p = tf.io.parse_single_example(rec, fd)
    vals = []
    for k in BANDS_TO_CHECK:
        v = float(tf.reduce_mean(tf.sparse.to_dense(p[k], default_value=0.0)).numpy())
        vals.append(0.0 if math.isnan(v) else round(v, 3))
    return tuple(vals)


def comp(f):
    return 'GZIP' if f.endswith('.gz') else None


def main():
    files = glob.glob(RAW_GLOB, recursive=True)
    if not files:
        raise SystemExit(f"No v4 shards at {RAW_GLOB}")
    print(f"Found {len(files)} shards.")

    random.seed(SEED)
    loc_map, total = {}, 0
    print("[Pass 1] assigning locations...")
    for f in files:
        for rec in tf.data.TFRecordDataset(f, compression_type=comp(f)):
            fp = fingerprint(rec.numpy())
            if fp not in loc_map:
                loc_map[fp] = 'val' if random.random() < VAL_SPLIT_PCT else 'train'
            total += 1
            if total % 1000 == 0:
                print(f"  scanned {total}", end='\r')
    print(f"\n  total={total}, locations={len(loc_map)}")

    print("[Pass 2] writing GZIP split...")
    opt = tf.io.TFRecordOptions(compression_type='GZIP')
    wtr = tf.io.TFRecordWriter(OUT_TRAIN, opt)
    wva = tf.io.TFRecordWriter(OUT_VAL, opt)
    tbuf, vbuf, ntr, nva = [], [], 0, 0

    def flush(buf, w):
        random.shuffle(buf)
        for r in buf:
            w.write(r)
        return []

    for f in files:
        for rec in tf.data.TFRecordDataset(f, compression_type=comp(f)):
            b = rec.numpy()
            if loc_map[fingerprint(b)] == 'train':
                tbuf.append(b); ntr += 1
                if len(tbuf) >= SHUFFLE_BUFFER: tbuf = flush(tbuf, wtr)
            else:
                vbuf.append(b); nva += 1
                if len(vbuf) >= SHUFFLE_BUFFER: vbuf = flush(vbuf, wva)
    if tbuf: flush(tbuf, wtr)
    if vbuf: flush(vbuf, wva)
    wtr.close(); wva.close()
    print(f"Done. train={ntr}, val={nva}")
    print(f"  {OUT_TRAIN}\n  {OUT_VAL}")


if __name__ == "__main__":
    main()
