#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
import csv, os, numpy as np
import sys; sys.path.insert(0, _ROOT)
from multiprocessing import Pool
import team_code as tc
from helper_code import load_demographics, load_signal_data, load_diagnoses

ROOT = "training_data"
rows = list(csv.DictReader(open(f"{ROOT}/demographics.csv")))

def one(r):
    site, bf, ses = r['SiteID'], r['BidsFolder'], r['SessionID']
    try:
        patient_data = load_demographics(f"{ROOT}/demographics.csv", bf, ses)
        demo = tc.extract_demographic_features(patient_data)
        phys_file = f"{ROOT}/physiological_data/{site}/{bf}_ses-{ses}.edf"
        if not os.path.exists(phys_file):
            return None
        phys_data, phys_fs = load_signal_data(phys_file)
        phys = tc.extract_physiological_features(phys_data, phys_fs)
        algo_file = f"{ROOT}/algorithmic_annotations/{site}/{bf}_ses-{ses}_caisr_annotations.edf"
        algo_data, _ = load_signal_data(algo_file)
        algo = tc.extract_algorithmic_annotations_features(algo_data)
        return np.hstack([demo, phys, algo]).astype(np.float32)
    except Exception as e:
        import sys; print(f"ERR {bf}: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    with Pool(8) as pool:
        feats = pool.map(one, rows, chunksize=4)
    ok = [f is not None for f in feats]
    def lab(r):
        v = r['Cognitive_Impairment'].strip().lower(); return 1.0 if v in ('true','1') else 0.0
    X = np.vstack([f for f in feats if f is not None]).astype(np.float32)
    y = np.array([lab(r) for r,o in zip(rows,ok) if o])
    sites = np.array([r['SiteID'] for r,o in zip(rows,ok) if o])
    ages = np.array([float(r['Age']) if r['Age'] else np.nan for r,o in zip(rows,ok) if o])
    np.savez("features_cache.npz", features=X, labels=y, sites=sites, ages=ages)
    print("saved:", X.shape, "kept", sum(ok), "/", len(rows))
