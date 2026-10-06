#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Regenerate the 29 CAISR-dynamics features for every record (submission 2204)."""
import csv, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, _HERE)
from caisr_features import extract_caisr, CAISR_KEYS

ROOT = _ROOT

def one(r):
    p = (f"{ROOT}/training_data/algorithmic_annotations/{r['SiteID']}/"
         f"{r['BidsFolder']}_ses-{r['SessionID']}_caisr_annotations.edf")
    if not os.path.exists(p):
        return np.full(len(CAISR_KEYS), np.nan, dtype=np.float32)
    return extract_caisr(p)

if __name__ == "__main__":
    rows = list(csv.DictReader(open(f"{ROOT}/training_data/demographics.csv")))
    with Pool(8) as pool:
        feats = pool.map(one, rows, chunksize=8)
    X = np.vstack(feats).astype(np.float32)
    np.savez(f"{ROOT}/caisr_features_cache.npz", features=X, keys=np.array(CAISR_KEYS))
    valid = int(np.isfinite(X).all(1).sum())
    print(f"saved caisr_features_cache.npz {X.shape}; fully-valid rows: {valid}/{len(X)}")
