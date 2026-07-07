#!/usr/bin/env python3
"""Lean deployment extractor: computes EXACTLY the 3 YASA-derived features used by the
15-feature model — cpl_phase_sin, rem_pct, aper_slope_c — in a single EDF load. Logic
mirrors yasa_features.extract_yasa (coupling) and yasa_features_extra (rem_pct, aperiodic
slope) so values match the validated caches. Stages come from caisr prob channels (available
at test time). Returns np.array([cpl_phase_sin, rem_pct, aper_slope_c]) with NaN on failure
(the model's SimpleImputer handles NaN)."""
import numpy as np
import mne, yasa
from scipy import signal as ssig
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

STAGE_NAMES = ['w', 'n1', 'n2', 'n3', 'r']
EEG_C = ['C4-M1', 'C3-M2', 'C4-A1', 'C3-A2']   # central -> aperiodic slope
EEG_F = ['F3-M2', 'F4-M1', 'F3-A2', 'F4-A1']   # frontal -> SO-spindle coupling
DEPLOY_KEYS = ['cpl_phase_sin', 'rem_pct', 'aper_slope_c', 'cpl_mrl_diff']

def _pick(raw, names):
    for n in names:
        if n in raw.ch_names:
            return n
    return None

def _stage(caisr_path, sf, n):
    rc = mne.io.read_raw_edf(caisr_path, preload=True, verbose='ERROR')
    probs = []
    for s in STAGE_NAMES:
        ch = f'caisr_prob_{s}'
        if ch not in rc.ch_names:
            return None
        probs.append(rc.get_data(picks=ch)[0])
    stage2 = np.vstack(probs).argmax(0)
    hyp = np.repeat(stage2, int(round(sf / rc.info['sfreq'])))
    L = min(len(hyp), n)
    return hyp[:L].astype(int), L

def _aper_slope(x, sf, mask):
    if mask.sum() < sf * 60:
        return np.nan
    f, pxx = ssig.welch(x[mask], sf, nperseg=int(sf * 4))
    band = (f >= 2) & (f <= 40) & (pxx > 0)
    if band.sum() < 10:
        return np.nan
    return float(np.polyfit(np.log10(f[band]), np.log10(pxx[band]), 1)[0])

def _mrl(ph):
    """mean resultant length of SO-spindle coupling phases (needs >=5 events)."""
    if len(ph) < 5:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * ph))))


def extract_deploy(edf_path, caisr_path, sf=100.0):
    # [cpl_phase_sin, rem_pct, aper_slope_c, cpl_mrl_diff]
    out = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        cc = _pick(raw, EEG_C); cf = _pick(raw, EEG_F)
        if cc is None or cf is None:
            return out
        raw.pick([cc, cf]); raw.load_data(verbose='ERROR'); raw.resample(sf)
        st = _stage(caisr_path, sf, raw.n_times)
        if st is None:
            return out
        hyp, L = st
        c = raw.get_data(picks=cc, units='uV')[0][:L]
        f = raw.get_data(picks=cf, units='uV')[0][:L]
        n2n3 = np.isin(hyp, [2, 3])
        # rem_pct
        out[1] = float((hyp == 4).sum()) / max(len(hyp), 1) * 100
        if n2n3.sum() / (sf * 60) < 5:
            return out
        # aper_slope_c (central, N2+N3)
        out[2] = _aper_slope(c, sf, n2n3)
        # SO-spindle coupling on frontal, N2+N3 (one detection feeds both coupling features)
        sw = yasa.sw_detect(f, sf, hypno=hyp, include=(2, 3), coupling=True)
        if sw is not None:
            s = sw.summary()
            if 'PhaseAtSigmaPeak' in s:
                ph = s['PhaseAtSigmaPeak'].values
                out[0] = float(np.sin(np.angle(np.mean(np.exp(1j * ph)))))
                # cpl_mrl_diff: change in coupling strength across the night (late - early).
                # Split detected slow waves by onset time at the sleep-period midpoint. Gated
                # at >=10 min N2+N3 to match the offline cycle cache exactly.
                if 'Start' in s and n2n3.sum() / (sf * 60) >= 10:
                    sleep_idx = np.where((hyp >= 1) & (hyp <= 4))[0]
                    if len(sleep_idx) >= 2:
                        mid = (sleep_idx[0] + sleep_idx[-1]) / 2.0
                        start_samp = s['Start'].values * sf
                        early = start_samp < mid
                        mrl_e = _mrl(ph[early]); mrl_l = _mrl(ph[~early])
                        if np.isfinite(mrl_e) and np.isfinite(mrl_l):
                            out[3] = mrl_l - mrl_e
        return out
    except Exception:
        return out
