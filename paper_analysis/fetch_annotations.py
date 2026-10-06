#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Download CAISR annotation EDFs in parallel (API-overhead bound)."""
import csv, os, subprocess, time, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = _ROOT
OUT  = f"{ROOT}/training_data/algorithmic_annotations"
DS   = "physionet/physionetchallenge2026data"
KAG  = f"{ROOT}/venv/bin/kaggle"
ENV  = {**os.environ,
        "KAGGLE_KEY": os.environ["KAGGLE_KEY"],
        "KAGGLE_USERNAME": os.environ["KAGGLE_USERNAME"]}
WORKERS = 8

rows = list(csv.DictReader(open(f"{ROOT}/training_data/demographics.csv")))
todo = []
for r in rows:
    s, bf, ses = r["SiteID"], r["BidsFolder"], r["SessionID"]
    base = f"{bf}_ses-{ses}_caisr_annotations.edf"
    dst  = f"{OUT}/{s}/{base}"
    if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
        todo.append((f"algorithmic_annotations/{s}/{base}", f"{OUT}/{s}"))

print(f"{len(rows)} records; {len(todo)} to fetch with {WORKERS} workers", flush=True)
for _, d in todo:
    os.makedirs(d, exist_ok=True)

def get(job):
    rel, dstdir = job
    for attempt in range(3):
        p = subprocess.run([KAG, "datasets", "download", DS, "-f", rel, "-p", dstdir, "--force"],
                           env=ENV, capture_output=True, text=True, timeout=240)
        if p.returncode == 0:
            return None
        time.sleep(2 * (attempt + 1))
    return rel

t0, done, failed = time.time(), 0, []
with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(get, j) for j in todo]
    for f in as_completed(futs):
        r = f.result()
        if r:
            failed.append(r)
        done += 1
        if done % 100 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"[{done}/{len(todo)}] {el/60:.1f} min, ETA {el/done*(len(todo)-done)/60:.1f} min, "
                  f"{len(failed)} failed", flush=True)

for z in glob.glob(f"{OUT}/*/*.zip"):
    subprocess.run(["unzip", "-oq", z, "-d", os.path.dirname(z)])
    os.remove(z)
n = len(glob.glob(f"{OUT}/*/*.edf"))
print(f"DONE: {n} annotation files, {len(failed)} failures", flush=True)
