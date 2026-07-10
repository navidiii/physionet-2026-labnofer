#!/usr/bin/env python3
"""Rich, SITE-INVARIANT features from CAISR algorithmic annotations.

Rationale (see leaderboard autopsy): raw-EEG spectral / YASA features carry each
recording site's amplifier + montage fingerprint, so a model that leans on them learns a
site shortcut that breaks on the unseen validation/test site (and evaporates entirely on
single-electrode montages). The CAISR annotation channels, by contrast, are produced by the
SAME algorithm on every site (identical 11 channels @ 2 Hz across S0001/I0002/I0006 and, by
construction, on the hidden I0004/I0007), so features derived from them are montage- and
amplifier-invariant -- exactly the failure mode that killed our previous attempts does not
apply here. Sleep-architecture disruption, fragmentation, arousal/respiratory/limb burden
and their night-time dynamics are all literature-backed markers of future cognitive decline
and are computable purely from these annotations.

extract_caisr(caisr_path) -> np.ndarray of scalar features (NaN on failure), aligned to
CAISR_KEYS. The pipeline's SimpleImputer handles NaN.
"""
import numpy as np
import mne
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

STAGE_PROB = ['caisr_prob_w', 'caisr_prob_n1', 'caisr_prob_n2', 'caisr_prob_n3', 'caisr_prob_r']
# stage codes after argmax: 0=W 1=N1 2=N2 3=N3 4=REM

CAISR_KEYS = [
    # composition (fractions of total recording)
    'frac_w', 'frac_n1', 'frac_n2', 'frac_n3', 'frac_r', 'sleep_eff', 'tst_hr',
    # fragmentation / dynamics
    'trans_per_hr', 'trans_entropy', 'awak_per_hr', 'waso_min',
    'sol_min', 'rem_lat_min',
    # bout structure (minutes)
    'bout_n2', 'bout_n3', 'bout_r', 'bout_w', 'n_rem_bouts',
    # temporal distribution
    'n3_frac_first', 'rem_frac_last',
    # event burden (per hour of sleep)
    'arou_idx', 'arou_idx_rem', 'arou_idx_nrem',
    'resp_idx', 'resp_idx_rem', 'resp_idx_nrem', 'limb_idx',
    # night dynamics (2nd half / 1st half of sleep)
    'arou_ratio', 'resp_ratio',
]

EPOCH_S = 30.0


def _event_count(mask):
    """number of rising edges (event onsets) in a boolean mask."""
    m = mask.astype(np.int8)
    return int(np.sum((m[1:] == 1) & (m[:-1] == 0)) + (1 if len(m) and m[0] == 1 else 0))


def _events_in(mask, sel):
    """count event onsets whose onset sample falls within boolean selector `sel`."""
    m = mask.astype(np.int8)
    onset = np.zeros_like(m, dtype=bool)
    if len(m):
        onset[0] = m[0] == 1
        onset[1:] = (m[1:] == 1) & (m[:-1] == 0)
    return int(np.sum(onset & sel))


def _bouts(ep, stage):
    """list of bout lengths (in epochs) for a given stage in the epoch hypnogram."""
    idx = (ep == stage).astype(np.int8)
    if idx.sum() == 0:
        return []
    d = np.diff(np.concatenate([[0], idx, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(ends - starts)


def extract_caisr(caisr_path):
    f = {}
    try:
        raw = mne.io.read_raw_edf(caisr_path, preload=True, verbose='ERROR')
        sf = raw.info['sfreq']
        chs = raw.ch_names
        if any(c not in chs for c in STAGE_PROB):
            return _to_array(f)
        probs = np.vstack([raw.get_data(picks=c)[0] for c in STAGE_PROB])
        samp_hyp = probs.argmax(0).astype(np.int8)          # 2 Hz, 0..4
        n = len(samp_hyp)
        spe = int(round(sf * EPOCH_S))                       # samples per 30s epoch
        n_ep = n // spe
        if n_ep < 20:                                        # < 10 min of data
            return _to_array(f)
        # epoch hypnogram by majority vote within each 30s window
        ep = samp_hyp[:n_ep * spe].reshape(n_ep, spe)
        ep_hyp = np.array([np.bincount(e, minlength=5).argmax() for e in ep], dtype=np.int8)

        sleep_ep = np.isin(ep_hyp, [1, 2, 3, 4])
        tst_ep = int(sleep_ep.sum())
        if tst_ep < 20:                                      # < 10 min sleep
            return _to_array(f)
        tst_hr = tst_ep * EPOCH_S / 3600.0
        total_ep = n_ep

        # composition
        for name, st in zip(['frac_w', 'frac_n1', 'frac_n2', 'frac_n3', 'frac_r'], range(5)):
            f[name] = float(np.mean(ep_hyp == st))
        f['sleep_eff'] = float(tst_ep) / total_ep
        f['tst_hr'] = tst_hr

        # sleep onset / offset (first & last sleep epoch)
        sl_idx = np.where(sleep_ep)[0]
        onset, offset = int(sl_idx[0]), int(sl_idx[-1])
        f['sol_min'] = onset * EPOCH_S / 60.0                # sleep onset latency (min)
        rem_idx = np.where(ep_hyp == 4)[0]
        rem_after = rem_idx[rem_idx >= onset]
        f['rem_lat_min'] = float((rem_after[0] - onset) * EPOCH_S / 60.0) if len(rem_after) else np.nan

        # transitions / fragmentation within the sleep period [onset, offset]
        period = ep_hyp[onset:offset + 1]
        if len(period) > 1:
            f['trans_per_hr'] = float(np.sum(period[1:] != period[:-1])) / max(tst_hr, 1e-6)
            T = np.zeros((5, 5))
            for a, b in zip(period[:-1], period[1:]):
                if a != b:
                    T[a, b] += 1
            tot = T.sum()
            if tot > 0:
                p = T[T > 0] / tot
                f['trans_entropy'] = float(-np.sum(p * np.log2(p)))
            awak = np.sum((period[1:] == 0) & (period[:-1] != 0))
            f['awak_per_hr'] = float(awak) / max(tst_hr, 1e-6)
            f['waso_min'] = float(np.sum(period == 0)) * EPOCH_S / 60.0

        # bout structure (minutes)
        for name, st in zip(['bout_n2', 'bout_n3', 'bout_r', 'bout_w'], [2, 3, 4, 0]):
            b = _bouts(ep_hyp, st)
            f[name] = (float(np.mean(b)) * EPOCH_S / 60.0) if b else 0.0
        f['n_rem_bouts'] = float(len(_bouts(ep_hyp, 4)))

        # temporal distribution: N3 front-loading, REM back-loading
        thirds = np.array_split(np.arange(total_ep), 3)
        n3_total = np.sum(ep_hyp == 3)
        rem_total = np.sum(ep_hyp == 4)
        f['n3_frac_first'] = float(np.sum(ep_hyp[thirds[0]] == 3) / n3_total) if n3_total > 0 else np.nan
        f['rem_frac_last'] = float(np.sum(ep_hyp[thirds[2]] == 4) / rem_total) if rem_total > 0 else np.nan

        # ---- events (kept at native 2 Hz) ----
        def get(ch):
            return raw.get_data(picks=ch)[0] if ch in chs else None
        arou, resp, limb = get('arousal_caisr'), get('resp_caisr'), get('limb_caisr')
        samp_sleep = np.isin(samp_hyp, [1, 2, 3, 4])
        samp_rem = samp_hyp == 4
        samp_nrem = np.isin(samp_hyp, [1, 2, 3])
        L = n
        def eidx(mask, sel):
            if mask is None:
                return np.nan
            return _events_in(mask[:L] > 0.5, sel[:L]) / max(tst_hr, 1e-6)
        f['arou_idx'] = eidx(arou, samp_sleep)
        f['arou_idx_rem'] = eidx(arou, samp_rem)
        f['arou_idx_nrem'] = eidx(arou, samp_nrem)
        f['resp_idx'] = eidx(resp, samp_sleep)
        f['resp_idx_rem'] = eidx(resp, samp_rem)
        f['resp_idx_nrem'] = eidx(resp, samp_nrem)
        f['limb_idx'] = eidx(limb, samp_sleep)

        # night dynamics: 2nd-half / 1st-half event rate within sleep period
        mid = (onset + offset) // 2 * spe
        def half_ratio(mask):
            if mask is None:
                return np.nan
            m = mask[:L] > 0.5
            s1 = samp_sleep.copy(); s1[mid:] = False
            s2 = samp_sleep.copy(); s2[:mid] = False
            r1 = _events_in(m, s1) / max(int(s1.sum()), 1)
            r2 = _events_in(m, s2) / max(int(s2.sum()), 1)
            return (r2 / r1) if r1 > 0 else np.nan
        f['arou_ratio'] = half_ratio(arou)
        f['resp_ratio'] = half_ratio(resp)

        return _to_array(f)
    except Exception:
        return _to_array(f)


def _to_array(f):
    return np.array([f.get(k, np.nan) for k in CAISR_KEYS], dtype=np.float32)
