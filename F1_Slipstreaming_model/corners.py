"""
Corner events: the apparent downforce / grip effect of following.

Corners are found automatically from the median speed-vs-lap-fraction
profile of all valid laps in the session (local speed minima with enough
prominence and an apex speed >= CORNER_MIN_APEX_KPH, i.e. aero-relevant
corners). For each corner and each driver-lap we measure:

  v_min      minimum speed within +/- CORNER_APEX_HALFWIDTH_M of the apex
  v_entry    speed at the start of the approach window
  brake_dist distance before the apex at which the brake first comes on
  t_window   time taken from approach start to corner exit

"Following" at a corner = nearest car ahead (by crossing-time order at the
same lap fraction) is within CORNER_FOLLOW_GAP_S at BOTH approach start and
apex, DRS closed for CORNER_DRS_LOOKBACK_M before the apex, green-flag lap.
"Clean air" = nearest car ahead >= CLEAN_AIR_GAP_S at both points.
Baseline = median over the same driver's clean valid laps within
+/- BASELINE_LAP_WINDOW laps.

Time gaps come from lap-fraction crossing times, so this works in bends
where a straight-line "car ahead" test fails.

Reported (positive = worse for the follower):
  apex_speed_loss_pct = (v_clean - v_follow) / v_clean * 100
  corner_time_delta_s = time_follow - time_clean
  brake_shift_m       = brake_dist_follow - brake_dist_clean (earlier braking)
Downforce loss % is derived later (analyse_eras.py) from apex_speed_ratio
with an assumed downforce-to-weight ratio, as a labelled model estimate.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import config


@dataclass
class CornerEvent:
    year: int
    event: str
    session: str
    corner_id: int
    corner_rel: float
    driver: str
    lap_number: float
    gap_s: float
    v_apex_follow_kph: float
    v_apex_clean_kph: float
    apex_speed_ratio: float
    apex_speed_loss_pct: float
    v_entry_follow_kph: float
    v_entry_clean_kph: float
    entry_speed_delta_pct: float
    brake_shift_m: float
    corner_time_delta_s: float
    n_clean_laps: int


def _lap_arrays(trace):
    out = {}
    for lap, g in trace.groupby("LapNumber"):
        g = g.sort_values("SessionTime_s")
        if len(g) < 10:
            continue
        out[lap] = dict(
            rel=np.maximum.accumulate(g["RelDist"].values),
            t=g["SessionTime_s"].values,
            v=g["Speed"].values,
            brake=g["Brake"].values.astype(bool),
            drs=g["DRS"].values,
            valid=bool(g["LapValid"].iloc[0]),
        )
    return out


def detect_corners(laps_by_driver, lap_len_m):
    grid = np.linspace(0.0, 1.0, 1000)
    profiles = []
    for laps in laps_by_driver.values():
        for a in laps.values():
            if a["valid"] and (a["rel"][-1] - a["rel"][0]) > 0.95:
                profiles.append(np.interp(grid, a["rel"], a["v"]))
    if len(profiles) < 5:
        return []
    ref = np.median(np.vstack(profiles), axis=0)
    ref = np.convolve(ref, np.ones(5) / 5.0, mode="same")
    peaks, _ = find_peaks(-ref, prominence=config.CORNER_MIN_PROMINENCE_KPH,
                          distance=max(1, int(0.012 * len(grid))))
    corners = []
    for p in peaks:
        ra = float(grid[p])
        if ref[p] < config.CORNER_MIN_APEX_KPH:
            continue
        r0 = ra - config.CORNER_APPROACH_M / lap_len_m
        r1 = ra + config.CORNER_EXIT_M / lap_len_m
        if r0 < 0.01 or r1 > 0.99:
            continue
        corners.append(dict(r_start=r0, r_apex=ra, r_end=r1, v_ref=float(ref[p])))
    return corners


def _corner_metrics(a, c, L):
    rel, t, v = a["rel"], a["t"], a["v"]
    r0, ra, r1 = c["r_start"], c["r_apex"], c["r_end"]
    if rel[0] > r0 or rel[-1] < r1:
        return None
    hw = config.CORNER_APEX_HALFWIDTH_M / L
    m = (rel >= ra - hw) & (rel <= ra + hw)
    if m.sum() < 2:
        return None
    mb = (rel >= r0) & (rel <= ra) & a["brake"]
    brake_dist = float((ra - rel[mb][0]) * L) if mb.any() else np.nan
    md = (rel >= ra - config.CORNER_DRS_LOOKBACK_M / L) & (rel <= ra)
    return dict(
        t_start=float(np.interp(r0, rel, t)),
        t_apex=float(np.interp(ra, rel, t)),
        t_end=float(np.interp(r1, rel, t)),
        v_min=float(v[m].min()),
        v_entry=float(np.interp(r0, rel, v)),
        brake_dist=brake_dist,
        drs_open=bool((a["drs"][md] >= 10).any()),
    )


def _gap_ahead(df, col):
    """Time gap to the nearest other-driver crossing earlier than ours."""
    gaps = np.full(len(df), np.inf)
    drv = df["driver"].values
    ts_all = df[col].values
    for d in np.unique(drv):
        sel = drv == d
        others = np.sort(ts_all[(~sel) & np.isfinite(ts_all)])
        if len(others) == 0:
            continue
        ts = ts_all[sel]
        idx = np.searchsorted(others, ts) - 1
        g = np.where(idx >= 0, ts - others[np.clip(idx, 0, None)], np.inf)
        g = np.where(np.isfinite(ts), g, np.inf)
        gaps[sel] = g
    return gaps


def detect_corner_events(traces, lap_len_m, year, event, era):
    if not np.isfinite(lap_len_m) or lap_len_m <= 0:
        print("  Corners: lap length unknown, skipped")
        return []
    laps_by_driver = {d: _lap_arrays(t) for d, t in traces.items()}
    corners = detect_corners(laps_by_driver, lap_len_m)
    events, no_base, rejected = [], 0, 0

    for ci, c in enumerate(corners):
        rows = []
        for d, laps in laps_by_driver.items():
            for lap, a in laps.items():
                m = _corner_metrics(a, c, lap_len_m)
                if m is not None:
                    rows.append(dict(driver=d, lap=lap, valid=a["valid"], **m))
        if len(rows) < 10:
            continue
        df = pd.DataFrame(rows)
        df["gap_start"] = _gap_ahead(df, "t_start")
        df["gap_apex"] = _gap_ahead(df, "t_apex")
        df["window_t"] = df["t_end"] - df["t_start"]

        G, CLEAN = config.CORNER_FOLLOW_GAP_S, config.CLEAN_AIR_GAP_S
        follow = (df["valid"] & (df["gap_start"] <= G) & (df["gap_apex"] <= G)
                  & ~df["drs_open"] & df["v_min"].notna())
        clean = (df["valid"] & (df["gap_start"] >= CLEAN) & (df["gap_apex"] >= CLEAN)
                 & df["v_min"].notna())

        for idx in df.index[follow]:
            r = df.loc[idx]
            base = df[clean & (df["driver"] == r["driver"])
                      & ((df["lap"] - r["lap"]).abs() > 0)
                      & ((df["lap"] - r["lap"]).abs() <= config.BASELINE_LAP_WINDOW)]
            if len(base) < config.MIN_CLEAN_LAPS_CORNER:
                no_base += 1
                continue
            v_clean = float(base["v_min"].median())
            if v_clean <= 0:
                continue
            ratio = r["v_min"] / v_clean
            loss = (1.0 - ratio) * 100.0
            if abs(loss) > config.MAX_PLAUSIBLE_APEX_DELTA_PCT:
                rejected += 1
                continue
            ve_clean = float(base["v_entry"].median())
            bd_clean = base["brake_dist"].median()
            brake_shift = (r["brake_dist"] - bd_clean
                           if np.isfinite(r["brake_dist"]) and np.isfinite(bd_clean) else np.nan)
            events.append(CornerEvent(
                year=year, event=event, session=era, corner_id=ci,
                corner_rel=c["r_apex"], driver=r["driver"], lap_number=r["lap"],
                gap_s=float((r["gap_start"] + r["gap_apex"]) / 2.0),
                v_apex_follow_kph=float(r["v_min"]), v_apex_clean_kph=v_clean,
                apex_speed_ratio=float(ratio), apex_speed_loss_pct=float(loss),
                v_entry_follow_kph=float(r["v_entry"]), v_entry_clean_kph=ve_clean,
                entry_speed_delta_pct=float((r["v_entry"] - ve_clean) / ve_clean * 100.0),
                brake_shift_m=float(brake_shift),
                corner_time_delta_s=float(r["window_t"] - base["window_t"].median()),
                n_clean_laps=int(len(base)),
            ))

    print(f"  Corners: {len(corners)} detected, {len(events)} following events "
          f"({no_base} skipped: too few clean laps; {rejected} discarded: "
          f"|apex change| > {config.MAX_PLAUSIBLE_APEX_DELTA_PCT:.0f}%)")
    return events
