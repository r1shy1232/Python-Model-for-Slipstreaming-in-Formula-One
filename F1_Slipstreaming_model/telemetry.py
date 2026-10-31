"""
Shared telemetry preparation for the straight-line and corner modules.

Per driver, one continuous trace across the race with columns:
  SessionTime_s, Speed, Throttle, DRS, Gear, Brake, X, Y, Distance,
  RelDist (fraction of that lap, 0-1), LapNumber, LapValid, Accel, Driver

Changes vs v2:
  - Car position (X, Y) is LINEARLY INTERPOLATED to the car-data timestamps
    instead of matched to the nearest sample. The old nearest-sample match
    could be up to ~0.15 s out, i.e. ~8-13 m of position error at 300 km/h,
    which is comparable to the 20 m gap threshold.
  - Gear and Brake are kept (gear-matching control; braking-point analysis).
  - Lap length is estimated per session (needed to convert RelDist <-> m).
"""

import numpy as np
import pandas as pd
import fastf1

K = 3  # half-window (samples) for acceleration / heading estimates


def load_session(year: int, event: str, session_code: str):
    session = fastf1.get_session(year, event, session_code)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    return session


def _prep_lap(lap, driver_code):
    """Return (DataFrame, lap_distance_m) for one lap, or (None, None)."""
    try:
        car = lap.get_car_data().add_distance()
        try:
            pos = lap.get_pos_data(pad=1, pad_side="both")
        except Exception:
            pos = lap.get_pos_data()
    except Exception:
        return None, None
    if car.empty or pos.empty:
        return None, None

    ct = car["SessionTime"].dt.total_seconds().values
    pt = pos["SessionTime"].dt.total_seconds().values
    order = np.argsort(pt)
    pt = pt[order]
    px = pos["X"].values[order]
    py = pos["Y"].values[order]

    inside = (ct >= pt[0]) & (ct <= pt[-1])
    x = np.where(inside, np.interp(ct, pt, px), np.nan)
    y = np.where(inside, np.interp(ct, pt, py), np.nan)

    dist = car["Distance"].values
    dmax = np.nanmax(dist) if len(dist) else np.nan
    if not np.isfinite(dmax) or dmax <= 0:
        return None, None

    valid = bool(
        lap["LapNumber"] > 1
        and pd.isna(lap["PitInTime"])
        and pd.isna(lap["PitOutTime"])
        and str(lap["TrackStatus"]) == "1"
    )

    gear = car["nGear"].values if "nGear" in car.columns else np.full(len(car), np.nan)
    brake = car["Brake"].values.astype(bool) if "Brake" in car.columns else np.zeros(len(car), bool)

    df = pd.DataFrame({
        "SessionTime_s": ct,
        "Speed": car["Speed"].values,
        "Throttle": car["Throttle"].values,
        "DRS": car["DRS"].values,
        "Gear": gear,
        "Brake": brake,
        "X": x,
        "Y": y,
        "Distance": dist,
        "RelDist": dist / dmax,
        "LapNumber": lap["LapNumber"],
        "LapValid": valid,
        "Driver": driver_code,
    })
    return df, (dmax if valid else None)


def _driver_trace(session, driver_code):
    laps = session.laps.pick_drivers(driver_code)
    frames, lap_lens = [], []
    for _, lap in laps.iterrows():
        if pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
            continue
        df, dmax = _prep_lap(lap, driver_code)
        if df is None:
            continue
        frames.append(df)
        if dmax is not None:
            lap_lens.append(dmax)
    if not frames:
        return pd.DataFrame(), []

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["Speed", "RelDist"])
    full = full.sort_values("SessionTime_s").reset_index(drop=True)
    full["LapValid"] = full["LapValid"].astype(bool)

    # longitudinal acceleration over a +/- K sample window
    t = full["SessionTime_s"].values
    v = full["Speed"].values / 3.6
    a = np.full(len(full), np.nan)
    if len(full) > 2 * K:
        dt = t[2 * K:] - t[:-2 * K]
        dv = v[2 * K:] - v[:-2 * K]
        with np.errstate(divide="ignore", invalid="ignore"):
            a[K:-K] = np.where((dt > 0) & (dt < 3.0), dv / dt, np.nan)
    full["Accel"] = a
    return full, lap_lens


def build_traces(session):
    """Return ({driver: trace_df}, lap_length_m)."""
    traces, all_lens = {}, []
    for code in session.laps["Driver"].unique().tolist():
        tr, lens = _driver_trace(session, code)
        if not tr.empty:
            traces[code] = tr
            all_lens += lens
    lap_len = float(np.median(all_lens)) if all_lens else float("nan")
    return traces, lap_len
