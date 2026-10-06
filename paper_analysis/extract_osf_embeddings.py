#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Extract OSF foundation-model patient-level embeddings for all training records.
Referencing-aware 12-channel mapping, sleep-agnostic uniform epoch sampling.
Checkpoints to osf_embeddings_cache.npz so partial LOSO probing is possible."""
import os, sys, csv, glob, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import mne
from scipy.signal import resample_poly
from math import gcd
import torch
torch.backends.mkldnn.enabled = False  # oneDNN conv SIGFPE workaround

SCRATCH = _os.environ.get("OSF_WORKDIR", ".")  # holds the cloned OSF repo (osf/) and its weights
OSF_DIR = _os.environ.get("OSF_DIR", f"{SCRATCH}/osf")
sys.path.insert(0, OSF_DIR)
from osf.backbone.vit1d_cls import vit_base

ROOT = "training_data"
OUT = "osf_embeddings_cache.npz"
CH = ["ECG","EMG_Chin","EMG_LLeg","EMG_RLeg","ABD","THX","NP","SN",
      "EOG_E1_A2","EOG_E2_A1","EEG_C3_A2","EEG_C4_A1"]
SR, EPS, SEQ = 64, 30, 1920
N_SAMPLE = 64  # epochs sampled uniformly across the night (de-risk: reduced for speed)

# single-channel modalities: candidate substrings
SINGLE = {
 "ECG":["ekg","ecg"], "EMG_Chin":["chin1-chin2","chin"], "EMG_LLeg":["lat","l leg","left leg","lleg"],
 "EMG_RLeg":["rat","r leg","right leg","rleg"], "ABD":["abd"], "THX":["chest","thor","thx"],
 "NP":["ptaf","nasal","cflow","airflow","flow"], "SN":["snore","snor"],
}
def _idx(low, cands, exact_first=True):
    for cand in cands:
        for i,c in enumerate(low):
            if c==cand: return i
    for cand in cands:
        for i,c in enumerate(low):
            if cand in c: return i
    return None

def build_signals(data, names):
    """Return [12, T] with referencing; missing -> zeros."""
    low=[n.lower() for n in names]
    T=data.shape[1]
    out=np.zeros((12,T),dtype=np.float32)
    def mast(*cands): return _idx(low,list(cands))
    m1=mast("m1","a1"); m2=mast("m2","a2")
    def eeg(direct_subs, elec, mst):
        j=_idx(low,direct_subs)
        if j is not None: return data[j]
        e=_idx(low,[elec])
        if e is not None and mst is not None: return data[e]-data[mst]
        if e is not None: return data[e]
        return None
    def eog(direct_subs, elec, mst):
        j=_idx(low,direct_subs)
        if j is not None: return data[j]
        e=_idx(low,[elec])
        if e is not None and mst is not None: return data[e]-data[mst]
        if e is not None: return data[e]
        return None
    sig={}
    sig["EEG_C3_A2"]=eeg(["c3-m2","c3-a2","c3m2"],"c3",m2)
    sig["EEG_C4_A1"]=eeg(["c4-m1","c4-a1","c4m1"],"c4",m1)
    sig["EOG_E1_A2"]=eog(["e1-m2","e1-a2","loc"],"e1",m2)
    sig["EOG_E2_A1"]=eog(["e2-m1","e2-a1","roc"],"e2",m1)
    for slot,cands in SINGLE.items():
        k=_idx(low,cands); sig[slot]=data[k] if k is not None else None
    for si,slot in enumerate(CH):
        s=sig.get(slot)
        if s is None: continue
        s=np.asarray(s,dtype=np.float32)
        m,sd=np.nanmean(s),np.nanstd(s)+1e-8
        out[si]=np.nan_to_num((s-m)/sd)
    return np.clip(out,-6,6)

_BB=None
def _init():
    global _BB
    torch.set_num_threads(2)
    p=torch.load(f"{OSF_DIR}/osf_backbone.pth",map_location="cpu",weights_only=False)
    m=p["metadata"]
    _BB=vit_base(num_leads=12,seq_len=1920,patch_size=m["patch_size_time"],
                 lead_wise=m["lead_wise"],patch_size_ch=m["patch_size_ch"])
    _BB.load_state_dict(p["state_dict"]); _BB.eval()

def embed_one(rec):
    site,bf,ses=rec
    f=f"{ROOT}/physiological_data/{site}/{bf}_ses-{ses}.edf"
    if not os.path.exists(f): return np.full(1536,np.nan,dtype=np.float32)
    try:
        raw=mne.io.read_raw_edf(f,preload=False,verbose="ERROR")
        # keep only channels that could map to an OSF slot -> cut RAM ~2x
        KEEP=("c3","c4","m1","m2","a1","a2","e1","e2","chin","lat","rat","leg",
              "ekg","ecg","abd","chest","thor","thx","ptaf","nasal","cflow","airflow","flow","snor")
        keep=[n for n in raw.ch_names if any(t in n.lower() for t in KEEP)]
        raw.pick(keep)
        raw.load_data(verbose="ERROR")
        sf=int(round(raw.info["sfreq"]))
        names=list(raw.ch_names)
        data=raw.get_data().astype(np.float32)  # [n_ch, T] at native sf
        del raw
        X0=build_signals(data,names)             # [12,T] referenced+zscored at native sf
        del data
        if sf!=SR:                               # fast polyphase resample 12ch only
            g=gcd(sf,SR); X=resample_poly(X0,SR//g,sf//g,axis=1).astype(np.float32)
        else:
            X=X0
        n_ep=X.shape[1]//SEQ
        if n_ep==0: return np.full(1536,np.nan,dtype=np.float32)
        ep_idx=np.linspace(0,n_ep-1,min(N_SAMPLE,n_ep)).astype(int)
        ep_idx=np.unique(ep_idx)
        batch=np.stack([X[:,e*SEQ:(e+1)*SEQ] for e in ep_idx],0)  # [k,12,SEQ]
        xt=torch.tensor(batch)
        embs=[]
        with torch.no_grad():
            for i in range(0,len(xt),64):
                cls,_=_BB.forward_encoding(xt[i:i+64],return_sequence=False)
                embs.append(cls.numpy())
        E=np.concatenate(embs,0)  # [k,768]
        return np.concatenate([E.mean(0),E.std(0)]).astype(np.float32)  # 1536
    except Exception as e:
        sys.stderr.write(f"ERR {bf}: {e}\n")
        return np.full(1536,np.nan,dtype=np.float32)

if __name__=="__main__":
    from multiprocessing import Pool
    rows=list(csv.DictReader(open(f"{ROOT}/demographics.csv")))
    def lab(r):
        v=r["Cognitive_Impairment"].strip().lower(); return 1.0 if v in("true","1") else 0.0
    recs=[(r["SiteID"],r["BidsFolder"],r["SessionID"]) for r in rows]
    labels=np.array([lab(r) for r in rows],dtype=np.float32)
    sites=np.array([r["SiteID"] for r in rows])
    ages=np.array([float(r["Age"]) if r["Age"] else np.nan for r in rows])
    sex=np.array([1.0 if r["Sex"].strip().lower().startswith("m") else 0.0 for r in rows])
    N=len(recs)
    feats=np.full((N,1536),np.nan,dtype=np.float32)
    # DE-RISK subset: all of small sites + all S0001 positives + capped S0001 negatives
    S0001_NEG_CAP=90
    subset=[]
    s0_neg=[]
    for i,s in enumerate(sites):
        if s in ("I0002","I0006"): subset.append(i)
        elif s=="S0001":
            if labels[i]==1: subset.append(i)
            else: s0_neg.append(i)
    s0_neg=list(np.array(s0_neg)[np.linspace(0,len(s0_neg)-1,S0001_NEG_CAP).astype(int)])
    subset+=s0_neg
    subset=sorted(set(subset))
    # interleave site order so partial checkpoints span all 3 sites (enables early LOSO)
    import collections
    buckets=collections.defaultdict(list)
    for i in subset: buckets[sites[i]].append(i)
    order=[]
    ptrs={s:0 for s in buckets}
    while len(order)<len(subset):
        for s in buckets:
            if ptrs[s]<len(buckets[s]):
                order.append(buckets[s][ptrs[s]]); ptrs[s]+=1
    N=len(order)
    print(f"DE-RISK subset: {N} records | "+" ".join(f"{s}={len(buckets[s])}(pos {int(labels[buckets[s]].sum())})" for s in buckets),flush=True)
    t0=time.time(); done=0
    with Pool(3,initializer=_init) as pool:
        for oi,vec in zip(order,pool.imap(embed_one,[recs[i] for i in order],chunksize=1)):
            feats[oi]=vec; done+=1
            if done%25==0 or done==N:
                el=time.time()-t0
                np.savez(OUT,features=feats,labels=labels,sites=sites,ages=ages,sex=sex,
                         keys=np.array(["mean%d"%i for i in range(768)]+["std%d"%i for i in range(768)]))
                valid=int(np.isfinite(feats).all(1).sum())
                print(f"[{done}/{N}] valid={valid} elapsed={el/60:.1f}min eta={el/done*(N-done)/60:.1f}min",flush=True)
    print("DONE",OUT,feats.shape,flush=True)
