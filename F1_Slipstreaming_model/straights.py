"""
Straight-line slipstream events: speed gain, effective C_D ratio, drag
reduction and time gained.

For each stretch where a driver follows within FOLLOWING_DISTANCE_THRESHOLD_M
of a car AHEAD (in line, full throttle, DRS closed, >= MIN_SPEED_KPH, near
constant speed, green-flag lap) for >= MIN_FOLLOWING_DURATION_S, compare the
driver's median speed with their OWN clean-air speed at the same part of
the lap (different valid lap within +/- BASELINE_LAP_WINDOW laps, same gear).

  speed_delta_pct    = (v_wake - v_clean) / v_clean * 100
  cd_ratio           = (v_clean / v_wake) ** 3        (equal-power assumption)
  drag_reduction_pct = (1 - cd_ratio) * 100
  time_gain_s        = L / v_clean - (time actually taken over L)

The C_D ratio is an *effective* ratio (parasitic + induced drag) and relies
on the equal-power assumption; gear-matching is a partial control for it.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

import config

K = 3


@dataclass
class StraightEvent:
    year: int
    event: str
    session: str
    following_driver: str
    leading_driver: str
    lap_number: float
    session_time_s: float
    rel_dist: float
    gap_m: float
    gear: float
    v_wake_kph: float
    v_clean_kph: float
    n_baseline_samples: int
    speed_delta_pct: float
    cd_ratio: float
    drag_reduction_pct: float
    time_gain_s: float
    time_gain_s_per_km: float


def nearest_car_ahead_gap(driver_trace, other_traces):
    """
    Nearest OTHER car that is ahead along this car's heading and within
    LATERAL_OFFSET_MAX_M of its line. The other car's position is linearly
    interpolated to this car's exact timestamps (no nearest-sample error).
    """
    out = driver_trace.copy()
    n = len(out)
    x = out["X"].values
    y = out["Y"].values
    t = out["SessionTime_s"].values

    hx = np.full(n, np.nan)
    hy = np.full(n, np.nan)
    if n > 2 * K:
        hx[K:-K] = x[2 * K:] - x[:-2 * K]
        hy[K:-K] = y[2 * K:] - y[:-2 * K]
    norm = np.hypot(hx, hy)
    with np.errstate(divide="ignore", invalid="ignore"):
        good = norm > 5.0                       # moved > 0.5 m over window
        hx = np.where(good, hx / norm, np.nan)
        hy = np.where(good, hy / norm, np.nan)

    gap = np.full(n, np.nan)
    lead = np.full(n, None, dtype=object)

    for code, ot in other_traces.items():
        if len(ot) < 2:
            continue
        ots = ot["SessionTime_s"].values
        ox = ot["X"].values
        oy = ot["Y"].values
        i1 = np.clip(np.searchsorted(ots, t), 1, len(ots) - 1)
        i0 = i1 - 1
        t0, t1 = ots[i0], ots[i1]
        span = t1 - t0
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(span > 0, (t - t0) / span, 0.0)
        inside = (t >= ots[0]) & (t <= ots[-1]) & (span <= 0.6)

        px = ox[i0] + (ox[i1] - ox[i0]) * frac
        py = oy[i0] + (oy[i1] - oy[i0]) * frac
        dx = px - x
        dy = py - y
        along = dx * hx + dy * hy
        lateral = np.abs(-dx * hy + dy * hx)
        dist = np.hypot(dx, dy) / 10.0          # FastF1 X,Y are in 1/10 m

        ok = inside & (along > 0) & (lateral / 10.0 <= config.LATERAL_OFFSET_MAX_M)
        dist = np.where(ok, dist, np.nan)
        better = ~np.isnan(dist) & (np.isnan(gap) | (dist < gap))
        gap[better] = dist[better]
        lead[better] = code

    out["gap_m"] = gap
    out["leading_driver"] = lead
    return out


def _event_mask(tr):
    return (
        (tr["gap_m"] < config.FOLLOWING_DISTANCE_THRESHOLD_M)
        & (tr["Throttle"] >= config.MIN_THROTTLE_PCT)
        & (tr["DRS"] < 10)
        & (tr["Speed"] >= config.MIN_SPEED_KPH)
        & (tr["Accel"].abs() <= config.MAX_ACCEL_MS2)
        & tr["LapValid"].astype(bool)
    )


def _stretches(mask, min_len):
    res, start = [], None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                res.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        res.append((start, len(mask) - 1))
    return res


def find_clean_air_baseline(trace, lap_number, rel_lo, rel_hi, gear):
    tol = config.BASELINE_RELDIST_TOL
    lap_gap = (trace["LapNumber"] - lap_number).abs()
    cond = (
        trace["LapValid"].astype(bool)
        & (lap_gap > 0) & (lap_gap <= config.BASELINE_LAP_WINDOW)
        & (trace["RelDist"] >= rel_lo - tol) & (trace["RelDist"] <= rel_hi + tol)
        & (trace["Speed"] >= config.MIN_SPEED_KPH)
        & (trace["Throttle"] >= config.MIN_THROTTLE_PCT)
        & (trace["DRS"] < 10)
        & (trace["Accel"].abs() <= config.MAX_ACCEL_MS2)
    )
    if np.isfinite(gear):
        cond &= (trace["Gear"] == gear)
    c = trace[cond]
    if c.empty:
        return None, 0
    clean = c["gap_m"].isna() | (c["gap_m"] > (c["Speed"] / 3.6) * config.CLEAN_AIR_GAP_S)
    c = c[clean]
    if len(c) < config.MIN_BASELINE_SAMPLES:
        return None, len(c)
    return float(c["Speed"].median()), len(c)


def detect_straight_events(traces, year, event, era):
    events = []
    no_base = rejected = 0

    for code, trace in traces.items():
        others = {c: t for c, t in traces.items() if c != code}
        wg = nearest_car_ahead_gap(trace, others)

        mask = _event_mask(wg).values.copy()
        laps = wg["LapNumber"].values
        mask[np.r_[False, np.diff(laps) != 0]] = False   # break at lap change

        t = wg["SessionTime_s"].values
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.25
        min_samples = max(1, int(config.MIN_FOLLOWING_DURATION_S / dt))

        for s, e in _stretches(mask, min_samples):
            seg = wg.iloc[s:e + 1]
            v_wake = float(seg["Speed"].median())
            gear_mode = seg["Gear"].mode()
            gear = float(gear_mode.iloc[0]) if len(gear_mode) and np.isfinite(gear_mode.iloc[0]) else np.nan
            v_clean, nb = find_clean_air_baseline(
                wg, seg["LapNumber"].iloc[0], seg["RelDist"].min(), seg["RelDist"].max(), gear)
            if v_clean is None or v_clean <= 0 or v_wake <= 0:
                no_base += 1
                continue
            delta = (v_wake - v_clean) / v_clean * 100.0
            if abs(delta) > config.MAX_PLAUSIBLE_SPEED_DELTA_PCT:
                rejected += 1
                continue

            L = float(seg["Distance"].iloc[-1] - seg["Distance"].iloc[0])
            t_follow = float(seg["SessionTime_s"].iloc[-1] - seg["SessionTime_s"].iloc[0])
            if L <= 0 or t_follow <= 0:
                continue
            t_gain = L / (v_clean / 3.6) - t_follow
            cd_ratio = (v_clean / v_wake) ** 3
            mid = seg.iloc[len(seg) // 2]
            events.append(StraightEvent(
                year=year, event=event, session=era, following_driver=code,
                leading_driver=mid["leading_driver"], lap_number=mid["LapNumber"],
                session_time_s=mid["SessionTime_s"], rel_dist=mid["RelDist"],
                gap_m=float(seg["gap_m"].mean()), gear=gear, v_wake_kph=v_wake,
                v_clean_kph=v_clean, n_baseline_samples=nb, speed_delta_pct=delta,
                cd_ratio=cd_ratio, drag_reduction_pct=(1 - cd_ratio) * 100.0,
                time_gain_s=t_gain, time_gain_s_per_km=t_gain / (L / 1000.0),
            ))

    print(f"  Straights: {len(events)} events ({no_base} skipped: no clean baseline; "
          f"{rejected} discarded: speed gain > {config.MAX_PLAUSIBLE_SPEED_DELTA_PCT:.0f}%)")
    return events
