"""
Offline self-tests with synthetic cars (no FastF1 / internet needed).
Run from src/:  python test_synthetic.py
"""
import os
import tempfile
import numpy as np
import pandas as pd

import config
from straights import detect_straight_events, nearest_car_ahead_gap
from corners import detect_corner_events
import analyse_eras


# ----------------------------------------------------------------- straights
N = 60


def s_lap(t0, v_kph, lap, x_off=0.0, y_off=0.0, valid=True, ramp=None, gear=8.0, driver="A"):
    t = t0 + np.arange(N) * 0.1
    v = np.full(N, float(v_kph)) if ramp is None else np.linspace(ramp[0], ramp[1], N)
    xm = np.cumsum(v / 3.6 * 0.1) + x_off
    rel = (xm - xm[0]) / (xm[-1] - xm[0])
    return pd.DataFrame(dict(
        SessionTime_s=t, X=xm * 10, Y=np.full(N, y_off * 10), Speed=v, Throttle=100.0,
        DRS=0, Gear=gear, Brake=False, Distance=xm - xm[0], RelDist=rel,
        LapNumber=lap, LapValid=valid, Driver=driver))


def with_accel(df, K=3):
    t = df["SessionTime_s"].values
    v = df["Speed"].values / 3.6
    a = np.full(len(df), np.nan)
    a[K:-K] = (v[2 * K:] - v[:-2 * K]) / (t[2 * K:] - t[:-2 * K])
    df["Accel"] = a
    return df


def build(laps):
    return with_accel(pd.concat(laps, ignore_index=True))


def test_straights():
    a3 = s_lap(0, 309, 3)
    b3 = s_lap(0, 309, 3, x_off=15, driver="B")
    a4 = s_lap(100, 300, 4)
    b4 = s_lap(100, 300, 4, y_off=500, driver="B")
    ev = detect_straight_events({"A": build([a3, a4]), "B": build([b3, b4])}, 2022, "T", "post2022")
    assert len(ev) == 1 and ev[0].following_driver == "A", ev
    assert abs(ev[0].speed_delta_pct - 3.0) < 0.1 and 8.0 < ev[0].drag_reduction_pct < 9.0
    assert ev[0].time_gain_s > 0
    print("S1 PASSED: follower flagged, 3% -> ~8.5% drag reduction, positive time gain")

    # interpolation: other car sampled at DIFFERENT instants must still give ~15 m
    b3s = s_lap(0.05, 309, 3, x_off=15 + 309 / 3.6 * 0.05, driver="B")  # shifted 0.05 s
    res = nearest_car_ahead_gap(build([a3]), {"B": build([b3s])})
    g = res["gap_m"].dropna()
    assert len(g) > 20 and abs(g.mean() - 15.0) < 0.5, g.mean()
    print("S2 PASSED: time-offset samples interpolated to ~15 m gap")

    behind = nearest_car_ahead_gap(build([a3]), {"B": build([s_lap(0, 309, 3, x_off=-15, driver="B")])})
    assert behind["gap_m"].isna().all()
    beside = nearest_car_ahead_gap(build([a3]), {"B": build([s_lap(0, 309, 3, x_off=5, y_off=8, driver="B")])})
    assert beside["gap_m"].isna().all()
    print("S3 PASSED: car behind / beside ignored")

    acc3 = s_lap(0, 0, 3, ramp=(150, 300))
    accb = s_lap(0, 0, 3, x_off=15, ramp=(150, 300), driver="B")
    ev = detect_straight_events({"A": build([acc3, a4]), "B": build([accb, b4])}, 2022, "T", "post2022")
    assert len(ev) == 0
    inv = detect_straight_events({"A": build([s_lap(0, 309, 1, valid=False), a4]),
                                  "B": build([s_lap(0, 309, 1, x_off=15, valid=False, driver="B"), b4])},
                                 2022, "T", "post2022")
    assert len(inv) == 0
    print("S4 PASSED: acceleration zone and invalid lap rejected")

    a4g = s_lap(100, 300, 4, gear=7.0)   # baseline in a different gear
    ev = detect_straight_events({"A": build([a3, a4g]), "B": build([b3, b4])}, 2022, "T", "post2022")
    assert len(ev) == 0
    print("S5 PASSED: gear-mismatched baseline rejected")


# ------------------------------------------------------------------- corners
L = 5000.0
R = np.linspace(0, 1, 800)


def speed_profile(follow=False):
    v = 300 - 140 * np.exp(-((R - 0.5) / 0.03) ** 2) - 20 * np.exp(-((R - 0.8) / 0.02) ** 2)
    if follow:
        v = v * (1 - 0.02 * np.exp(-((R - 0.5) / 0.05) ** 2))
    return v


def c_lap(t0, lap, follow=False, driver="A", valid=True):
    v = speed_profile(follow)
    dt = np.diff(R) * L / (v[:-1] / 3.6)
    t = t0 + np.r_[0, np.cumsum(dt)]
    return pd.DataFrame(dict(
        SessionTime_s=t, Speed=v, Throttle=100.0, DRS=0, Gear=6.0,
        Brake=(R > 0.465) & (R < 0.5), X=np.nan, Y=np.nan, Distance=R * L, RelDist=R,
        LapNumber=lap, LapValid=valid, Driver=driver))


def test_corners():
    laps = range(2, 13)
    A = pd.concat([c_lap(100 * n, n, follow=(n in (5, 6))) for n in laps], ignore_index=True)
    B = pd.concat([c_lap(100 * n - (0.2 if n in (5, 6) else 20.0), n, driver="B") for n in laps],
                  ignore_index=True)
    ev = detect_corner_events({"A": A, "B": B}, L, 2022, "T", "post2022")
    assert len(ev) > 0, "no corner events"
    assert all(e.driver == "A" for e in ev), "leader B must not be flagged"
    assert {e.lap_number for e in ev} == {5, 6}, {e.lap_number for e in ev}
    big = [e for e in ev if abs(e.corner_rel - 0.5) < 0.02]
    flat = [e for e in ev if abs(e.corner_rel - 0.8) < 0.02]
    assert big and flat, "expected both corners"
    for e in big:
        assert abs(e.apex_speed_loss_pct - 2.0) < 0.35, e.apex_speed_loss_pct
        assert e.corner_time_delta_s > 0
        assert abs(e.brake_shift_m) < 1.0
    for e in flat:
        assert abs(e.apex_speed_loss_pct) < 0.3, e.apex_speed_loss_pct
    print(f"C1 PASSED: {len(ev)} corner events; 2% apex loss recovered "
          f"({big[0].apex_speed_loss_pct:.2f}%), flat-out corner ~0%, leader not flagged")


# ------------------------------------------------------------------ analysis
def test_analysis():
    rng = np.random.default_rng(1)
    srows, crows = [], []
    for era, drag_mu, loss_mu in (("pre2022", 6.0, 2.0), ("post2022", 4.0, 1.0)):
        for circ in ["A GP", "B GP", "C GP", "D GP", "E GP", "F GP"]:
            for drv in ["X", "Y", "Z"]:
                for _ in range(8):
                    d = rng.normal(drag_mu, 2.0)
                    srows.append(dict(session=era, event=circ, year=2021 if era == "pre2022" else 2022,
                                      following_driver=drv, drag_reduction_pct=d,
                                      speed_delta_pct=d / 3.0, time_gain_s_per_km=d / 100.0))
                    ls = rng.normal(loss_mu, 0.8)
                    crows.append(dict(session=era, event=circ, year=2021 if era == "pre2022" else 2022,
                                      driver=drv, apex_speed_ratio=1 - ls / 100.0, v_apex_clean_kph=200.0,
                                      apex_speed_loss_pct=ls, corner_time_delta_s=ls / 10.0,
                                      entry_speed_delta_pct=0.0, brake_shift_m=ls))
    S, C = pd.DataFrame(srows), pd.DataFrame(crows)
    with tempfile.TemporaryDirectory() as tmp:
        diffs, paired, pen = analyse_eras.run_all(S, C, tmp)
        for f in ["era_difference_pooled.csv", "era_difference_paired_circuits.csv", "penalty_ratio.csv",
                  "fig_paired_circuits.png", "fig_pooled_distributions.png", "fig_penalty_ratio.png"]:
            assert os.path.exists(os.path.join(tmp, f)), f
    d = diffs[diffs["metric"] == "drag_reduction_pct"].iloc[0]
    assert d["diff_post_minus_pre"] < -1.0 and d["ci_excludes_zero"], d
    p = paired[paired["metric"] == "drag_reduction_pct"].iloc[0]
    assert p["n_circuits"] == 6 and p["circuits_lower_post"] == 6, p
    assert len(pen) == 2 * len(config.DOWNFORCE_D250_VALUES)
    # df-loss formula check: D_c = 2*(200/250)^2 = 1.28, ratio = 0.98
    Dc = 2.0 * (200 / 250) ** 2
    expect = (1 + Dc) * (1 - 0.98 ** 2) / Dc * 100
    got = analyse_eras.add_downforce_columns(pd.DataFrame(dict(
        apex_speed_ratio=[0.98], v_apex_clean_kph=[200.0]))).iloc[0][analyse_eras.CENTRAL]
    assert abs(expect - got) < 1e-9
    print(f"A1 PASSED: era difference recovered ({d['diff_post_minus_pre']:.2f} pts, CI excludes 0), "
          f"6/6 circuits paired, files + figures written, downforce formula checked")


def test_extras():
    """Matched-corner and gap analyses recover planted effects."""
    rng = np.random.default_rng(3)
    rows = []
    for circ in ["A GP", "B GP", "C GP"]:
        for cid, rel in enumerate([0.2, 0.5, 0.8]):
            for era, mu in (("pre2022", 6.0), ("post2022", 4.0)):
                for drv in "XYZ":
                    for _ in range(3):
                        ls = rng.normal(mu, 0.7)
                        rows.append(dict(session=era, event=circ, year=2021 if era == "pre2022" else 2022,
                                         corner_id=cid, corner_rel=rel + rng.normal(0, 0.002), driver=drv,
                                         apex_speed_ratio=1 - ls / 100, v_apex_clean_kph=200.0,
                                         apex_speed_loss_pct=ls, corner_time_delta_s=ls / 10))
    C = pd.DataFrame(rows)
    S = pd.DataFrame(dict(session=["pre2022"] * 40 + ["post2022"] * 40, event="A GP", year=2021,
                          following_driver="X", gap_m=rng.uniform(2, 19, 80),
                          drag_reduction_pct=rng.normal(3, 3, 80)))
    with tempfile.TemporaryDirectory() as tmp:
        ex = analyse_eras.run_extras(S, C, tmp)
        ms = ex["matched_summary"].iloc[0]
        assert ms["n_corners"] == 9 and ms["n_lower_in_2022"] == 9, ms
        assert abs(ms["mean_diff"] + 2.0) < 0.5 and ms["ci_excludes_zero"], ms
        for f in ["matched_corners.csv", "matched_corners_summary.csv", "fig_matched_corners.png",
                  "straight_gap_bands.csv", "straight_gap_adjusted.csv"]:
            assert os.path.exists(os.path.join(tmp, f)), f
    print(f"X1 PASSED: 9 matched corners, planted -2.0 pt effect recovered ({ms['mean_diff']:.2f}); gap tables written")


if __name__ == "__main__":
    test_straights()
    test_corners()
    test_analysis()
    test_extras()
    print("\nALL SYNTHETIC TESTS PASSED")