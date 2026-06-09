"""Shared aDBS sensing + stim-settings extraction logic.

Companion to ``adaptpd_sensing_waveform_viewer.ipynb``. The per-patient detection
functions below are lifted verbatim from that notebook's helper cell so both the
single-patient viewer and the cohort notebook stay in sync. Cohort helpers at the
bottom run the detection across every patient and assemble one combined DataFrame.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-patient detection (verbatim from the viewer notebook helper cell)
# ---------------------------------------------------------------------------
def _tail(s: str) -> str:
    return s.split('.')[-1] if isinstance(s, str) else s


def parse_file_stem(stem: str) -> dict:
    m = re.match(r'^([a-zA-Z0-9]+)_(.+[AP]M)_(.+)$', stem)
    if not m:
        raise ValueError(f'Unexpected filename stem: {stem}')
    return {
        'patient': m.group(1),
        'timestamp': pd.to_datetime(m.group(2), format=r'%Y-%m-%d_%H_%M_%S%p'),
        'visit': m.group(3),
    }


# SensingElectrodeConfigDef.ZERO_AND_TWO + Left -> ZERO_TWO_LEFT (matches CalibrationTests.Channel)
CONFIG_TO_PAIR = {
    'ZERO_AND_TWO': 'ZERO_TWO',
    'ZERO_AND_THREE': 'ZERO_THREE',
    'ONE_AND_THREE': 'ONE_THREE',
    'ONE_AND_TWO': 'ONE_TWO',
}


def waveform_channel_from_sensing(config: str, hemisphere: str) -> str:
    key = _tail(config)  # e.g. ZERO_AND_TWO from SensingElectrodeConfigDef.ZERO_AND_TWO
    if key not in CONFIG_TO_PAIR:
        raise ValueError(f'Unknown sensing config: {config}')
    return f"{CONFIG_TO_PAIR[key]}_{hemisphere.upper()}"


def _pretty_lead_location(loc: str) -> str:
    """LeadLocationDef.Stn -> 'STN', LeadLocationDef.Gpi -> 'GPi'."""
    tail = (_tail(loc) or '') if loc else ''
    return {'STN': 'STN', 'GPI': 'GPi'}.get(tail.upper(), tail)


def discover_lead_locations(meta: pd.DataFrame) -> dict:
    """Map hemisphere -> stim target (STN / GPi) from LeadConfiguration.

    Prefers *consent* exports; falls back to any file that carries LeadConfiguration.
    """
    ordered = meta.copy()
    ordered['is_consent'] = ordered['visit'].str.contains('consent', case=False, na=False)
    ordered = ordered.sort_values(['is_consent', 'timestamp'], ascending=[False, True])
    for _, row in ordered.iterrows():
        with open(row['path']) as f:
            d = json.load(f)
        lc = d.get('LeadConfiguration') or {}
        for state in ('Final', 'Initial'):
            out = {}
            for lead in lc.get(state) or []:
                hemi = _tail(lead.get('Hemisphere'))
                loc = _pretty_lead_location(lead.get('LeadLocation'))
                if hemi in ('Left', 'Right') and loc:
                    out[hemi] = loc
            if out:
                return out
    return {}


def _visit_priority(visit: str) -> int:
    """Prefer programming session over later eval exports."""
    v = visit.upper()
    if 'ADBS_SETUP' in v:
        return 0
    if 'ADBSEVAL' in v:
        return 1
    if 'CDBS_BASELINE' in v:
        return 2
    return 9


def _adapt_program_score(group: dict) -> int:
    """Score how likely this bank is the real aDBS program (not a spare/unconfigured bank)."""
    score = 0
    if group.get('ActiveGroup'):
        score += 100  # selected on IPG at session end — not the same as 'adapt', but strong signal
    name = (group.get('GroupName') or '').strip().lower()
    if any(tok in name for tok in ('adbs', 'adapt')):
        score += 50
    if name in ('do not use', 'b only', 'b', 'baseline', ''):
        score -= 25
    for sc in (group.get('ProgramSettings') or {}).get('SensingChannel') or []:
        st = str(sc.get('AdaptiveTherapyStatus') or '')
        if 'RUNNING' in st:
            score += 80
        elif 'SUSPENDED' in st:
            score += 40
        elif 'NOT_CONFIGURED' in st:
            score -= 60
        elif 'DISABLED' in st:
            score -= 40
        if sc.get('Mode'):
            score += 25
    return score


def _active_contacts(sc: dict) -> str:
    """Compact stim montage from ElectrodeState, e.g. '-(SenSight_1a, ...) +(Case)'."""
    cath, anod = [], []
    for e in sc.get('ElectrodeState') or []:
        state = str(e.get('ElectrodeStateResult'))
        name = _tail(e.get('Electrode'))
        if 'Negative' in state:
            cath.append(name)
        elif 'Positive' in state:
            anod.append(name)
    parts = []
    if cath:
        parts.append('-(' + ', '.join(cath) + ')')
    if anod:
        parts.append('+(' + ', '.join(anod) + ')')
    return ' '.join(parts)


def _sensing_row(group: dict, sc: dict, hemisphere: str, visit: str, timestamp, path, groups_state: str, score: int) -> dict:
    cfg = sc.get('Channel')
    nested = ((sc.get('SensingSetup') or {}).get('ChannelSignalResult') or {}).get('Channel')
    setup = sc.get('SensingSetup') or {}
    at = sc.get('AdaptiveTherapy') or {}
    return {
        'visit': visit,
        'timestamp': timestamp,
        'file': path.name,
        'groups_state': groups_state,
        'group_id': group.get('GroupId'),
        'group_name': group.get('GroupName'),
        'ipg_bank_selected': group.get('ActiveGroup'),
        'adapt_program_score': score,
        'hemisphere': hemisphere,
        'config': cfg,
        'config_short': _tail(cfg),
        'waveform_channel': waveform_channel_from_sensing(cfg, hemisphere),
        'device_reports_channel': _tail(nested) if nested else None,
        'adaptive_status': sc.get('AdaptiveTherapyStatus'),
        'mode': _tail(sc.get('Mode')),
        'lfp_beta_center_hz': setup.get('FrequencyInHertz'),
        'lfp_averaging_ms': setup.get('AveragingDurationInMilliSeconds'),
        # --- aDBS stim settings (the amplitude 'phases' + LFP thresholds gating them) ---
        'rate_hz': sc.get('RateInHertz'),
        'pulse_width_us': sc.get('PulseWidthInMicroSecond'),
        'active_contacts': _active_contacts(sc),
        'amp_lower_ma': sc.get('LowerLimitInMilliAmps'),
        'amp_upper_ma': sc.get('UpperLimitInMilliAmps'),
        'amp_lower_capture_ma': sc.get('LowerCaptureAmplitudeInMilliAmps'),
        'amp_upper_capture_ma': sc.get('UpperCaptureAmplitudeInMilliAmps'),
        'amp_suspend_ma': sc.get('SuspendAmplitudeInMilliAmps'),
        'lfp_lower_threshold': sc.get('LowerLfpThreshold'),
        'lfp_upper_threshold': sc.get('UpperLfpThreshold'),
        'lfp_measured_lower': sc.get('MeasuredLowerLfp'),
        'lfp_measured_upper': sc.get('MeasuredUpperLfp'),
        'transition_up_ms': sc.get('TransitionUpInMilliSeconds'),
        'transition_down_ms': sc.get('TransitionDownInMilliSeconds'),
        'upper_threshold_onset_ms': at.get('UpperThresholdOnsetInMilliSeconds'),
        'lower_threshold_onset_ms': at.get('LowerThresholdOnsetInMilliSeconds'),
        'startup_delay_ms': at.get('AdaptiveStartupDelayInMilliSeconds'),
        'detection_blanking_ms': at.get('DetectionBlankingDurationInMilliSeconds'),
    }


def discover_adbs_sensing(meta: pd.DataFrame, min_score: int = 20) -> pd.DataFrame:
    """
    Auto-detect aDBS sensing from setup / eval / baseline JSONs.

    ipg_bank_selected (exported as ActiveGroup in raw JSON) means this bank is the one
    selected on the programmer at session end — not whether aDBS is running. We pick the
    highest-scoring adapt program per hemisphere, preferring ADBS_SETUP over later visits.
    """
    visit_re = re.compile(r'ADBS_SETUP|aDBSEval|CDBS_BASELINE', re.I)
    candidates = meta[meta['visit'].str.contains(visit_re)].copy()
    if candidates.empty:
        raise RuntimeError('No ADBS_SETUP / baseline / eval JSON found for this patient')

    candidates['visit_priority'] = candidates['visit'].map(_visit_priority)
    candidates = candidates.sort_values(['visit_priority', 'timestamp'])

    for _, row in candidates.iterrows():
        with open(row['path']) as f:
            d = json.load(f)
        file_rows = []
        for state in ('Final', 'Initial'):
            state_bonus = 10 if state == 'Final' else 0
            for g in d.get('Groups', {}).get(state) or []:
                score = _adapt_program_score(g) + state_bonus
                if score < min_score:
                    continue
                for sc in (g.get('ProgramSettings') or {}).get('SensingChannel') or []:
                    hemi = _tail(sc.get('HemisphereLocation', ''))
                    if hemi not in ('Left', 'Right'):
                        continue
                    file_rows.append(
                        _sensing_row(g, sc, hemi, row['visit'], row['timestamp'], row['path'], state, score)
                    )
        if file_rows:
            df = pd.DataFrame(file_rows)
            best = df.loc[df.groupby('hemisphere')['adapt_program_score'].idxmax()]
            return best.sort_values('hemisphere').reset_index(drop=True)

    raise RuntimeError('No adapt-program SensingChannel found; inspect Groups in ADBS_SETUP JSON')


def summarize_visit_chronic_lfp(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Visit / extend-access files carry cumulative LFPTrendLogs (home beta band power).
    Each reading has only DateTime, LFP, AmplitudeInMilliAmps — no channel tag.
    Channel + beta center frequency come from the active program's SensingChannel on that visit.
    """
    visit_re = re.compile(r'VISIT|EXTEND_ACCESS', re.I)
    visits = meta[meta['visit'].str.contains(visit_re, na=False)].sort_values('timestamp')
    rows = []
    for _, row in visits.iterrows():
        with open(row['path']) as f:
            d = json.load(f)
        lfp_logs = (d.get('DiagnosticData') or {}).get('LFPTrendLogs') or {}
        n_trend = sum(len(entries) for windows in lfp_logs.values() for entries in windows.values())
        las = (d.get('EventSummary') or {}).get('LfpAndAmplitudeSummary') or []
        las_by_hemi = {_tail(x.get('Hemisphere')): x for x in las}
        found_active = False
        groups_used = (d.get('Groups') or {}).get('Final') or []
        groups_state = 'Final'
        if not any(g.get('ActiveGroup') for g in groups_used):
            groups_used = (d.get('Groups') or {}).get('Initial') or []
            groups_state = 'Initial'
        for g in groups_used:
            if not g.get('ActiveGroup'):
                continue
            found_active = True
            gid = _tail(g.get('GroupId'))
            for sc in (g.get('ProgramSettings') or {}).get('SensingChannel') or []:
                hemi = _tail(sc.get('HemisphereLocation'))
                ss = sc.get('SensingSetup') or {}
                summary = las_by_hemi.get(hemi) or {}
                rows.append({
                    'visit': row['visit'],
                    'timestamp': row['timestamp'],
                    'file': row['path'].name,
                    'groups_state': groups_state,
                    'active_group_id': gid,
                    'group_name': g.get('GroupName'),
                    'hemisphere': hemi,
                    'config_short': _tail(sc.get('Channel')),
                    'waveform_channel': waveform_channel_from_sensing(sc.get('Channel'), hemi),
                    'lfp_beta_center_hz': ss.get('FrequencyInHertz'),
                    'lfp_averaging_ms': ss.get('AveragingDurationInMilliSeconds'),
                    'adaptive_status': _tail(sc.get('AdaptiveTherapyStatus')),
                    'n_lfptrend_readings': n_trend,
                    'pct_between_lfp_thresholds': summary.get('BetweenThresholdPercent'),
                    'pct_time_adbs_running': summary.get('TimeAdaptiveRunningPercent'),
                })
        if not found_active and n_trend:
            rows.append({
                'visit': row['visit'], 'timestamp': row['timestamp'], 'file': row['path'].name,
                'groups_state': None, 'active_group_id': None, 'group_name': None,
                'hemisphere': None, 'config_short': None, 'waveform_channel': None,
                'lfp_beta_center_hz': None, 'lfp_averaging_ms': None, 'adaptive_status': None,
                'n_lfptrend_readings': n_trend,
                'pct_between_lfp_thresholds': None, 'pct_time_adbs_running': None,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['timestamp', 'hemisphere']).reset_index(drop=True)


def iter_time_domain_clips(data: dict, stream_keys=('CalibrationTests', 'SenseChannelTests', 'BrainSenseTimeDomain')):
    for stream in stream_keys:
        for idx, clip in enumerate(data.get(stream) or []):
            y = clip.get('TimeDomainData')
            if not y:
                continue
            sr = clip.get('SampleRateInHz') or 250
            t0 = pd.to_datetime(clip.get('FirstPacketDateTime'))
            n = len(y)
            t = t0 + pd.to_timedelta(np.arange(n) / sr, unit='s')
            yield {
                'stream': stream,
                'clip_index': idx,
                'channel': clip.get('Channel'),
                'pass': clip.get('Pass'),
                'gain': clip.get('Gain'),
                'sample_rate_hz': sr,
                'n_samples': n,
                'duration_s': n / sr,
                'first_packet': t0,
                'time': t,
                'voltage': np.asarray(y, dtype=float),
            }


# ---------------------------------------------------------------------------
# Cohort helpers
# ---------------------------------------------------------------------------
_PATIENT_RE = re.compile(r'^([a-zA-Z0-9]+)_')


def list_patients(data_root) -> list[str]:
    """All unique device serials (patient IDs) with >=1 JSON export under data_root."""
    counts = Counter(
        m.group(1)
        for p in Path(data_root).glob('*.json')
        if (m := _PATIENT_RE.match(p.name))
    )
    return sorted(counts)


def build_meta(data_root, patient: str) -> pd.DataFrame:
    """File index (one row per JSON) for a single patient, sorted by timestamp."""
    rows = []
    for p in sorted(Path(data_root).glob(f'{patient}_*.json')):
        try:
            info = parse_file_stem(p.stem)
        except ValueError:
            continue
        rows.append({**info, 'path': p, 'size_mb': p.stat().st_size / 1e6})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)


def extract_patient(data_root, patient: str) -> pd.DataFrame:
    """One row per hemisphere: detected aDBS sensing channel + stim settings + lead target."""
    meta = build_meta(data_root, patient)
    if meta.empty:
        raise RuntimeError('no parseable JSON files')
    sensing = discover_adbs_sensing(meta)
    leads = discover_lead_locations(meta)
    out = sensing.copy()
    out.insert(0, 'patient', patient)
    out['stim_target'] = out['hemisphere'].map(leads)
    out['n_files'] = len(meta)
    return out


def build_cohort(data_root, patients=None):
    """Run extract_patient over every patient. Returns (cohort_df, failures_df)."""
    if patients is None:
        patients = list_patients(data_root)
    frames, failures = [], []
    for pt in patients:
        try:
            frames.append(extract_patient(data_root, pt))
        except Exception as e:  # record and continue across the cohort
            failures.append({'patient': pt, 'error': f'{type(e).__name__}: {e}'})
    if frames:
        with warnings.catch_warnings():  # benign all-NA-column dtype FutureWarning
            warnings.simplefilter('ignore', FutureWarning)
            cohort = pd.concat(frames, ignore_index=True)
    else:
        cohort = pd.DataFrame()
    return cohort, pd.DataFrame(failures)


# ---------------------------------------------------------------------------
# Longitudinal (per-visit) extraction: programming from Visit >= 3 onwards
# ---------------------------------------------------------------------------
def visit_number(visit: str):
    """Numeric visit index from a visit label, e.g. 'VISIT_3_NA' -> 3 (None if not a VISIT_n)."""
    m = re.search(r'VISIT[_ ]?(\d+)', str(visit), re.I)
    return int(m.group(1)) if m else None


def is_visit_3plus(visit: str, include_extend: bool = True) -> bool:
    """True for VISIT_3.. and (optionally) EXTEND_ACCESS exports."""
    n = visit_number(visit)
    if n is not None:
        return n >= 3
    return include_extend and 'EXTEND_ACCESS' in str(visit).upper()


def extract_file_sensing(path, visit, timestamp, min_score: int = 20) -> pd.DataFrame:
    """Best-scoring adaptive program's sensing + stim settings per hemisphere for ONE export."""
    with open(path) as f:
        d = json.load(f)
    rows = []
    groups = d.get('Groups') or {}
    for state in ('Final', 'Initial'):
        bonus = 10 if state == 'Final' else 0
        for g in groups.get(state) or []:
            score = _adapt_program_score(g) + bonus
            if score < min_score:
                continue
            for sc in (g.get('ProgramSettings') or {}).get('SensingChannel') or []:
                hemi = _tail(sc.get('HemisphereLocation', ''))
                if hemi not in ('Left', 'Right'):
                    continue
                rows.append(_sensing_row(g, sc, hemi, visit, timestamp, Path(path), state, score))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    best = df.loc[df.groupby('hemisphere')['adapt_program_score'].idxmax()]
    return best.sort_values('hemisphere').reset_index(drop=True)


def extract_patient_timeline(data_root, patient: str, visit_filter=None) -> pd.DataFrame:
    """One row per hemisphere x export for a patient (full programming timeline).

    visit_filter: optional predicate on the visit label; None = every export.
    Only exports that actually contain an adaptive program contribute rows.
    """
    meta = build_meta(data_root, patient)
    if meta.empty:
        raise RuntimeError('no parseable JSON files')
    leads = discover_lead_locations(meta)
    sel = meta if visit_filter is None else meta[meta['visit'].map(visit_filter)]
    if sel.empty:
        raise RuntimeError('no matching exports')
    frames = []
    for _, r in sel.sort_values('timestamp').iterrows():
        f = extract_file_sensing(r['path'], r['visit'], r['timestamp'])
        if f.empty:
            continue
        f.insert(0, 'patient', patient)
        f['visit_number'] = visit_number(r['visit'])
        f['stim_target'] = f['hemisphere'].map(leads)
        frames.append(f)
    if not frames:
        raise RuntimeError('no adaptive program found in matching exports')
    with warnings.catch_warnings():  # benign all-NA-column dtype FutureWarning
        warnings.simplefilter('ignore', FutureWarning)
        out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['timestamp', 'hemisphere']).reset_index(drop=True)


def extract_patient_visits(data_root, patient: str, include_extend: bool = True) -> pd.DataFrame:
    """Visit >= 3 (optionally EXTEND_ACCESS) longitudinal rows for a patient."""
    return extract_patient_timeline(
        data_root, patient, lambda v: is_visit_3plus(v, include_extend)
    )


def build_cohort_timeline(data_root, patients=None, visit_filter=None):
    """Run extract_patient_timeline over every patient. Returns (cohort_df, failures_df).

    visit_filter=None -> every export (full timeline); pass a predicate to restrict.
    """
    if patients is None:
        patients = list_patients(data_root)
    frames, failures = [], []
    for pt in patients:
        try:
            frames.append(extract_patient_timeline(data_root, pt, visit_filter))
        except Exception as e:  # record and continue across the cohort
            failures.append({'patient': pt, 'error': f'{type(e).__name__}: {e}'})
    if frames:
        with warnings.catch_warnings():  # benign all-NA-column dtype FutureWarning
            warnings.simplefilter('ignore', FutureWarning)
            cohort = pd.concat(frames, ignore_index=True)
    else:
        cohort = pd.DataFrame()
    if not cohort.empty:
        cohort = cohort.sort_values(['patient', 'timestamp', 'hemisphere']).reset_index(drop=True)
    return cohort, pd.DataFrame(failures)


def build_cohort_visits(data_root, patients=None, include_extend: bool = True):
    """Visit >= 3 longitudinal cohort table. Returns (cohort_df, failures_df)."""
    return build_cohort_timeline(
        data_root, patients, lambda v: is_visit_3plus(v, include_extend)
    )
