#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Stream: download one PSG -> extract 8bbbcae features -> delete. Parallel, low disk."""
import csv, os, sys, subprocess, time, glob, tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = _ROOT
sys.path.insert(0, _HERE); sys.path.insert(0, ROOT)
import team_code_spectral as tc
from helper_code import load_signal_data

DS  = "physionet/physionetchallenge2026data"
KAG = f"{ROOT}/venv/bin/kaggle"
ENV = {**os.environ, "KAGGLE_KEY": os.environ["KAGGLE_KEY"],
       "KAGGLE_USERNAME": os.environ["KAGGLE_USERNAME"]}
CSVP = f"{ROOT}/channel_table.csv"
OUTNPY = f"{_HERE}/spectral_partial"
os.makedirs(OUTNPY, exist_ok=True)
WORKERS = 4
N_PHYS, N_ALGO = 54, 17

rows = list(csv.DictReader(open(f"{ROOT}/training_data/demographics.csv")))

def work(i_r):
    i, r = i_r
    out = f"{OUTNPY}/{i:05d}.npy"
    if os.path.exists(out):
        return None
    s, bf, ses = r["SiteID"], r["BidsFolder"], r["SessionID"]
    demo = tc.extract_demographic_features(r)
    # algorithmic features from the annotations already on disk
    ap = f"{ROOT}/training_data/algorithmic_annotations/{s}/{bf}_ses-{ses}_caisr_annotations.edf"
    try:
        ad, _ = load_signal_data(ap)
        algo = tc.extract_algorithmic_annotations_features(ad)
    except Exception:
        algo = np.full(N_ALGO, np.nan)
    # physiological: download -> extract -> delete
    phys = np.full(N_PHYS, np.nan)
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        rel = f"physiological_data/{s}/{bf}_ses-{ses}.edf"
        for attempt in range(3):
            p = subprocess.run([KAG, "datasets", "download", DS, "-f", rel, "-p", td, "--force"],
                               env=ENV, capture_output=True, text=True, timeout=900)
            if p.returncode == 0:
                break
            time.sleep(3 * (attempt + 1))
        for z in glob.glob(f"{td}/*.zip"):
            subprocess.run(["unzip", "-oq", z, "-d", td]); os.remove(z)
        edf = glob.glob(f"{td}/*.edf")
        if edf:
            try:
                d, fs = load_signal_data(edf[0])
                phys = tc.extract_physiological_features(d, fs, csv_path=CSVP)
            except Exception as e:
                sys.stderr.write(f"extract fail {bf}: {e}\n")
    np.save(out, np.hstack([demo, phys, algo]).astype(np.float32))
    return None

todo = [(i, r) for i, r in enumerate(rows) if not os.path.exists(f"{OUTNPY}/{i:05d}.npy")]
print(f"{len(rows)} records; {len(todo)} to process with {WORKERS} workers", flush=True)
t0, done = time.time(), 0
with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(work, j) for j in todo]
    for f in as_completed(futs):
        try: f.result()
        except Exception as e: sys.stderr.write(f"worker error: {e}\n")
        done += 1
        if done % 25 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"[{done}/{len(todo)}] {el/60:.1f} min, ETA {el/done*(len(todo)-done)/60:.1f} min", flush=True)

parts = sorted(glob.glob(f"{OUTNPY}/*.npy"))
X = np.vstack([np.load(p) for p in parts])
np.savez(f"{ROOT}/spectral_features_cache.npz", features=X)
print(f"DONE: spectral_features_cache.npz {X.shape}", flush=True)
