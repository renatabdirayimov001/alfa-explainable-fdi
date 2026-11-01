"""Leakage audit for the processed ALFA dataset.

Applies two threshold rules to signals that expose the injected faults:
  * engine alarm: vfr_hud throttle == 0 for >= 1 s (airborne, after the first 20 s)
  * surface alarm: any control-surface rc-out channel (1, 3, 4, 5) with zero variance over 3 s
Prints the alarm time relative to the fault onset for every flight.
Usage: python leakage_audit.py /path/to/processed
"""
import os, sys
import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else '../processed'
for f in sorted(os.listdir(root)):
    if 'no_ground_truth' in f:
        continue
    d = os.path.join(root, f)
    onset = np.inf
    for st in ['engines', 'aileron', 'rudder', 'elevator']:
        p = os.path.join(d, f'{f}-failure_status-{st}.csv')
        if os.path.exists(p):
            s = pd.read_csv(p)
            if len(s):
                onset = min(onset, s['%time'].min() * 1e-9)
    h = pd.read_csv(os.path.join(d, f'{f}-mavros-vfr_hud.csv'))
    t, thr = h['%time'].values * 1e-9, h['field.throttle'].values
    zero = (thr == 0) & (t > t[0] + 20)
    thr_alarm, run = None, 0.0
    for i in range(1, len(t)):
        run = run + (t[i] - t[i - 1]) if (zero[i] and zero[i - 1]) else 0.0
        if run >= 1.0:
            thr_alarm = t[i]
            break
    r = pd.read_csv(os.path.join(d, f'{f}-mavros-rc-out.csv'))
    tr = r['%time'].values * 1e-9
    ch = r[[f'field.channels{k}' for k in (1, 3, 4, 5)]].values.astype(float)
    rc_alarm = None
    for i in range(len(tr)):
        if tr[i] < tr[0] + 20:
            continue
        m = (tr > tr[i] - 3) & (tr <= tr[i])
        if m.sum() >= 6 and (ch[m].std(0) == 0).any():
            rc_alarm = tr[i]
            break
    rel = lambda a: None if a is None else (round(a - onset, 2) if np.isfinite(onset) else 'alarm in normal flight')
    print(f'{f:70s} throttle rule: {rel(thr_alarm)} | rc-out rule: {rel(rc_alarm)}')
