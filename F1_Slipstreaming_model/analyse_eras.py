"""
Compare pre-2022 (2021) and post-2022 (2022) slipstream effects.

USAGE (from inside src/, after run_analysis.py):
    python analyse_eras.py

Metric conventions (positive = worse for the follower / bigger wake effect):
  drag_reduction_pct   larger = bigger slipstream benefit on straights
  apex_speed_loss_pct  larger = more cornering speed lost when following
  df_loss_pct_D*       modelled downforce loss % (see below)

DOWNFORCE MODEL (assumption-based, shown as a sensitivity range):
  Same corner, same line, same tyre grip mu:  a_lat = v^2 / R = mu (g + DF/m)
  => with D = DF / (m g) in clean air, D_c = D250 * (v_clean / 250 kph)^2 and
     ratio = v_follow / v_clean:
       D_w = (1 + D_c) * ratio^2 - 1
       downforce_loss = (1 - D_w / D_c) = (1 + D_c)(1 - ratio^2) / D_c
  D250 (downforce-to-weight at 250 km/h) is NOT measured; config.py gives a
  range. Driver caution and tyre effects are folded into "apparent" loss.

COMBINED METRICS (era level, medians):
  penalty ratio       = median downforce loss % / median drag reduction %
  efficiency retained = (1 - DF loss) / (1 - drag reduction)

STATISTICS: events within one driver's weekend are not independent, so
bootstrap CIs resample (circuit, driver) clusters within each era; circuit
level comparisons are paired (same circuit, 2021 vs 2022).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config

PRE, POST = "pre2022", "post2022"
CENTRAL = f"df_loss_pct_D{config.DOWNFORCE_D250_CENTRAL:g}"

STRAIGHT_METRICS = ["drag_reduction_pct", "speed_delta_pct", "time_gain_s_per_km"]
CORNER_METRICS = ["apex_speed_loss_pct", CENTRAL, "corner_time_delta_s",
                  "entry_speed_delta_pct", "brake_shift_m"]


def add_downforce_columns(C):
    C = C.copy()
    if C.empty:
        return C
    ratio = C["apex_speed_ratio"].values
    for D250 in config.DOWNFORCE_D250_VALUES:
        Dc = D250 * (C["v_apex_clean_kph"].values / 250.0) ** 2
        C[f"df_loss_pct_D{D250:g}"] = (1.0 + Dc) * (1.0 - ratio ** 2) / Dc * 100.0
    return C


def _clusters(df, metric):
    d = df.dropna(subset=[metric])
    return [g[metric].values for _, g in d.groupby(["event", _driver_col(d)])]


def _driver_col(d):
    return "following_driver" if "following_driver" in d.columns else "driver"


def _boot_medians(clusters, rng, n):
    k = len(clusters)
    if k == 0:
        return np.full(n, np.nan)
    out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        out[i] = np.median(np.concatenate([clusters[j] for j in idx]))
    return out


def _ci(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def era_difference(df, metric, rng):
    """Pooled median difference (post - pre) with cluster-bootstrap 95% CI."""
    pre, post = df[df["session"] == PRE], df[df["session"] == POST]
    cp, cq = _clusters(pre, metric), _clusters(post, metric)
    if not cp or not cq:
        return dict(metric=metric, n_pre=0, n_post=0)
    b = _boot_medians(cq, rng, config.BOOTSTRAP_N) - _boot_medians(cp, rng, config.BOOTSTRAP_N)
    lo, hi = _ci(b)
    return dict(
        metric=metric,
        n_pre=int(pre[metric].notna().sum()), n_post=int(post[metric].notna().sum()),
        median_pre=float(pre[metric].median()), median_post=float(post[metric].median()),
        diff_post_minus_pre=float(post[metric].median() - pre[metric].median()),
        ci_low=lo, ci_high=hi,
        ci_excludes_zero=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
    )


def cell_medians(df, metric):
    """Circuit x era medians, only cells with enough events."""
    g = df.dropna(subset=[metric]).groupby(["event", "session"])[metric]
    t = pd.DataFrame({"median": g.median(), "n": g.size()}).reset_index()
    t = t[t["n"] >= config.MIN_EVENTS_PER_CELL]
    piv = t.pivot(index="event", columns="session", values="median")
    if PRE not in piv.columns or POST not in piv.columns:
        return pd.DataFrame()
    return piv.dropna(subset=[PRE, POST])


def paired_summary(df, metric):
    piv = cell_medians(df, metric)
    if piv.empty:
        return dict(metric=metric, n_circuits=0), piv
    diff = piv[POST] - piv[PRE]
    res = dict(metric=metric, n_circuits=len(piv),
               median_diff_post_minus_pre=float(diff.median()),
               circuits_lower_post=int((diff < 0).sum()),
               circuits_higher_post=int((diff > 0).sum()))
    if len(piv) >= 6 and (diff != 0).any():
        try:
            res["wilcoxon_p"] = float(stats.wilcoxon(piv[POST], piv[PRE]).pvalue)
        except Exception:
            res["wilcoxon_p"] = np.nan
    return res, piv


def penalty_table(S, C, rng):
    rows = []
    for era in (PRE, POST):
        s, c = S[S["session"] == era], C[C["session"] == era]
        sc = _clusters(s, "drag_reduction_pct")
        if not sc or c.empty:
            continue
        sb = _boot_medians(sc, rng, config.BOOTSTRAP_N)
        drag_med = float(s["drag_reduction_pct"].median())
        for D in config.DOWNFORCE_D250_VALUES:
            col = f"df_loss_pct_D{D:g}"
            cc = _clusters(c, col)
            if not cc:
                continue
            cb = _boot_medians(cc, rng, config.BOOTSTRAP_N)
            df_med = float(c[col].median())
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_b = cb / sb
                eff_b = (1 - cb / 100.0) / (1 - sb / 100.0)
            lo, hi = _ci(ratio_b)
            elo, ehi = _ci(eff_b)
            rows.append(dict(
                era=era, D250=D, median_drag_reduction_pct=drag_med, median_df_loss_pct=df_med,
                penalty_ratio=df_med / drag_med if drag_med else np.nan,
                penalty_ci_low=lo, penalty_ci_high=hi,
                efficiency_retained=(1 - df_med / 100.0) / (1 - drag_med / 100.0),
                eff_ci_low=elo, eff_ci_high=ehi,
                n_straight=int(s["drag_reduction_pct"].notna().sum()), n_corner=int(len(c)),
            ))
    return pd.DataFrame(rows)


def _label(m):
    return {"drag_reduction_pct": "Drag reduction (%)",
            "apex_speed_loss_pct": "Apex speed loss (%)",
            CENTRAL: f"Modelled downforce loss (%), D250={config.DOWNFORCE_D250_CENTRAL:g}"}.get(m, m)


def make_figures(S, C, pen, out_dir):
    metrics = [("drag_reduction_pct", S), ("apex_speed_loss_pct", C), (CENTRAL, C)]
    colors = {PRE: "#D85A30", POST: "#378ADD"}

    # Fig 1: paired circuit medians
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (m, df) in zip(axes, metrics):
        piv = cell_medians(df, m) if len(df) else pd.DataFrame()
        if piv.empty:
            ax.set_title(_label(m) + "\n(not enough data)")
            continue
        for ev, row in piv.iterrows():
            ax.plot([0, 1], [row[PRE], row[POST]], marker="o", alpha=0.7)
            ax.annotate(ev.replace(" Grand Prix", ""), (1.02, row[POST]), fontsize=7, va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2021\n(pre-2022)", "2022\n(post-2022)"])
        ax.set_xlim(-0.2, 1.6)
        ax.set_title(_label(m), fontsize=10)
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-circuit medians, 2021 vs 2022 (only cells with enough events)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_paired_circuits.png"), dpi=200)
    plt.close(fig)

    # Fig 2: pooled distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (m, df) in zip(axes, metrics):
        data = [df[df["session"] == e][m].dropna().values if len(df) else np.array([]) for e in (PRE, POST)]
        if all(len(d) == 0 for d in data):
            ax.set_title(_label(m) + "\n(no data)")
            continue
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], showmeans=True, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"2021\nn={len(data[0])}", f"2022\nn={len(data[1])}"])
        ax.set_title(_label(m), fontsize=10)
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Pooled events (events from one driver-weekend are not independent)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pooled_distributions.png"), dpi=200)
    plt.close(fig)

    # Fig 3: penalty ratio vs assumed D250
    if len(pen):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, (era, name) in enumerate([(PRE, "2021"), (POST, "2022")]):
            p = pen[pen["era"] == era]
            if p.empty:
                continue
            x = np.arange(len(p)) + (k - 0.5) * 0.18
            yerr = np.vstack([p["penalty_ratio"] - p["penalty_ci_low"],
                              p["penalty_ci_high"] - p["penalty_ratio"]])
            ax.errorbar(x, p["penalty_ratio"], yerr=np.nan_to_num(yerr), fmt="o", capsize=4,
                        color=colors[era], label=name)
            ax.set_xticks(np.arange(len(p)))
            ax.set_xticklabels([f"D250={d:g}" for d in p["D250"]])
        ax.axhline(config.LITERATURE_PENALTY_RATIO, color="k", ls="--", lw=1,
                   label="Literature (old regs): ~60% / ~20%")
        ax.set_ylabel("Downforce loss % / drag reduction %")
        ax.set_title("Slipstream penalty ratio (95% cluster-bootstrap CI)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_penalty_ratio.png"), dpi=200)
        plt.close(fig)


def run_all(S, C, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)
    C = add_downforce_columns(C)

    diffs, paired = [], []
    for m in STRAIGHT_METRICS:
        if m in S.columns and len(S):
            diffs.append(era_difference(S, m, rng))
            paired.append(paired_summary(S, m)[0])
    for m in CORNER_METRICS:
        if m in C.columns and len(C):
            diffs.append(era_difference(C, m, rng))
            paired.append(paired_summary(C, m)[0])
    diffs, paired = pd.DataFrame(diffs), pd.DataFrame(paired)
    pen = penalty_table(S, C, rng) if len(S) and len(C) else pd.DataFrame()

    diffs.to_csv(os.path.join(out_dir, "era_difference_pooled.csv"), index=False)
    paired.to_csv(os.path.join(out_dir, "era_difference_paired_circuits.csv"), index=False)
    pen.to_csv(os.path.join(out_dir, "penalty_ratio.csv"), index=False)
    make_figures(S, C, pen, out_dir)
    return diffs, paired, pen


# ----------------------------------------------------------------------------
# EXTRA ANALYSES (v3.1): matched corners and following-gap check
# ----------------------------------------------------------------------------
MATCH_REL_TOL = getattr(config, "MATCH_REL_TOL", 0.012)      # lap-fraction tolerance (~60-85 m)
MATCH_MIN_EVENTS = getattr(config, "MATCH_MIN_EVENTS", 2)    # events needed per corner per year


def match_corners(C):
    """
    Pair each 2022 corner with the nearest 2021 corner at the SAME circuit
    (apex within MATCH_REL_TOL of the same lap fraction, one-to-one).
    Removes the circuit/corner-mix confound in the pooled comparison.
    """
    cells = (C.groupby(["event", "session", "corner_id"])
               .agg(rel=("corner_rel", "mean")).reset_index())
    pairs = []
    for ev, d in cells.groupby("event"):
        pre, post = d[d["session"] == PRE], d[d["session"] == POST]
        cand = []
        for p in post.itertuples():
            for q in pre.itertuples():
                dist = abs(p.rel - q.rel)
                if dist <= MATCH_REL_TOL:
                    cand.append((dist, q.corner_id, p.corner_id, (p.rel + q.rel) / 2.0))
        cand.sort()
        used_pre, used_post = set(), set()
        for dist, qc, pc, rel in cand:
            if qc in used_pre or pc in used_post:
                continue
            used_pre.add(qc)
            used_post.add(pc)
            pairs.append(dict(event=ev, rel=rel, corner_pre=qc, corner_post=pc))
    return pd.DataFrame(pairs)


def _by_driver(df, metric):
    d = df.dropna(subset=[metric])
    return [g[metric].values for _, g in d.groupby("driver")]


def matched_corner_analysis(C, metric, rng):
    """Within-corner 2022-minus-2021 differences, with a cluster bootstrap CI."""
    pairs = match_corners(C)
    rows, cells = [], []
    for r in pairs.itertuples():
        a = C[(C["event"] == r.event) & (C["session"] == PRE) & (C["corner_id"] == r.corner_pre)]
        b = C[(C["event"] == r.event) & (C["session"] == POST) & (C["corner_id"] == r.corner_post)]
        va, vb = a[metric].dropna().values, b[metric].dropna().values
        if len(va) < MATCH_MIN_EVENTS or len(vb) < MATCH_MIN_EVENTS:
            continue
        rows.append(dict(event=r.event, lap_fraction=round(r.rel, 3), n_2021=len(va), n_2022=len(vb),
                         median_2021=float(np.median(va)), median_2022=float(np.median(vb)),
                         diff_2022_minus_2021=float(np.median(vb) - np.median(va))))
        cells.append((_by_driver(a, metric), _by_driver(b, metric)))
    table = pd.DataFrame(rows)
    if table.empty:
        return table, dict(metric=metric, n_corners=0)
    d = table["diff_2022_minus_2021"].values
    k = len(d)
    boots = np.empty(config.BOOTSTRAP_N)
    for i in range(config.BOOTSTRAP_N):
        diffs = []
        for j in rng.integers(0, k, k):
            ca, cb = cells[j]
            ra = np.concatenate([ca[t] for t in rng.integers(0, len(ca), len(ca))])
            rb = np.concatenate([cb[t] for t in rng.integers(0, len(cb), len(cb))])
            diffs.append(np.median(rb) - np.median(ra))
        boots[i] = np.mean(diffs)
    lo, hi = _ci(boots)
    n_lower, n_higher = int((d < 0).sum()), int((d > 0).sum())
    summ = dict(metric=metric, n_corners=k, n_circuits=int(table["event"].nunique()),
                n_lower_in_2022=n_lower, n_higher_in_2022=n_higher,
                mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                ci_low=lo, ci_high=hi, ci_excludes_zero=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                sign_test_p=float(stats.binomtest(n_lower, n_lower + n_higher, 0.5).pvalue)
                if (n_lower + n_higher) > 0 else np.nan)
    try:
        summ["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if k >= 6 else np.nan
    except Exception:
        summ["wilcoxon_p"] = np.nan
    return table, summ


def gap_analysis(S):
    """Does the straight-line drag reduction depend on following gap, and does gap explain the era difference?"""
    S = S.dropna(subset=["gap_m", "drag_reduction_pct"]).copy()
    S["gap_band"] = pd.cut(S["gap_m"], [0, 6, 10, 14, 20], labels=["<6 m", "6-10 m", "10-14 m", "14-20 m"])
    bands = (S.groupby(["gap_band", "session"], observed=True)["drag_reduction_pct"]
               .agg(n="size", median="median").reset_index())
    X = np.column_stack([np.ones(len(S)), S["gap_m"].values, (S["session"] == POST).astype(float).values])
    y = S["drag_reduction_pct"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    adj = dict(median_gap_2021_m=float(S[S["session"] == PRE]["gap_m"].median()),
               median_gap_2022_m=float(S[S["session"] == POST]["gap_m"].median()),
               corr_gap_vs_drag_reduction=float(np.corrcoef(S["gap_m"], y)[0, 1]),
               gap_slope_pts_per_m=float(beta[1]), gap_slope_se=float(se[1]),
               era_effect_adjusted_for_gap=float(beta[2]), era_effect_se=float(se[2]),
               share_negative_2021=float((S[S["session"] == PRE]["drag_reduction_pct"] < 0).mean()),
               share_negative_2022=float((S[S["session"] == POST]["drag_reduction_pct"] < 0).mean()))
    return bands, adj


def make_matched_figure(table, out_dir):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in table.itertuples():
        ax.plot([0, 1], [r.median_2021, r.median_2022], marker="o", alpha=0.75)
        ax.annotate(f"{r.event.replace(' Grand Prix', '')} ({r.lap_fraction:.2f})",
                    (1.02, r.median_2022), fontsize=7, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2021", "2022"])
    ax.set_xlim(-0.2, 1.75)
    ax.set_ylabel("Median apex speed loss when following (%)")
    ax.set_title("Same corner, 2021 vs 2022\n(each line = one corner; label = lap fraction of apex)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_matched_corners.png"), dpi=200)
    plt.close(fig)


def run_extras(S, C, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    out = {}
    C = add_downforce_columns(C) if len(C) else C
    if len(C):
        summaries = []
        for m in ["apex_speed_loss_pct", CENTRAL, "corner_time_delta_s"]:
            if m not in C.columns:
                continue
            table, summ = matched_corner_analysis(C, m, rng)
            summaries.append(summ)
            if m == "apex_speed_loss_pct":
                table.to_csv(os.path.join(out_dir, "matched_corners.csv"), index=False)
                make_matched_figure(table, out_dir)
                out["matched_table"] = table
        out["matched_summary"] = pd.DataFrame(summaries)
        out["matched_summary"].to_csv(os.path.join(out_dir, "matched_corners_summary.csv"), index=False)
    if len(S):
        bands, adj = gap_analysis(S)
        bands.to_csv(os.path.join(out_dir, "straight_gap_bands.csv"), index=False)
        pd.DataFrame([adj]).to_csv(os.path.join(out_dir, "straight_gap_adjusted.csv"), index=False)
        out["gap_bands"], out["gap_adjusted"] = bands, adj
    return out


def main():
    S = pd.read_csv(config.STRAIGHT_CSV) if os.path.exists(config.STRAIGHT_CSV) else pd.DataFrame()
    C = pd.read_csv(config.CORNER_CSV) if os.path.exists(config.CORNER_CSV) else pd.DataFrame()
    if S.empty and C.empty:
        print("No events found. Run run_analysis.py first.")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    diffs, paired, pen = run_all(S, C, config.OUTPUT_DIR)
    print("\n=== POOLED ERA DIFFERENCE (post - pre), cluster-bootstrap 95% CI ===")
    print(diffs.round(3).to_string(index=False))
    print("\n=== PAIRED BY CIRCUIT (same circuit, 2021 vs 2022) ===")
    print(paired.round(3).to_string(index=False))
    print("\n=== SLIPSTREAM PENALTY RATIO (literature, old regs: ~3.0) ===")
    print(pen.round(3).to_string(index=False) if len(pen) else "not enough data")
    ex = run_extras(S, C, config.OUTPUT_DIR)
    if "matched_summary" in ex:
        print("\n=== MATCHED CORNERS (same circuit, same lap position; 2022 minus 2021) ===")
        print(ex["matched_table"].round(2).to_string(index=False))
        print(ex["matched_summary"].round(3).to_string(index=False))
    if "gap_adjusted" in ex:
        print("\n=== STRAIGHTS: FOLLOWING-GAP CHECK ===")
        print(ex["gap_bands"].round(2).to_string(index=False))
        print({k: round(v, 3) for k, v in ex["gap_adjusted"].items()})
    print(f"\nTables and figures saved in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()