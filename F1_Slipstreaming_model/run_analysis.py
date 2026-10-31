"""
Run the straight-line and corner analyses over all configured sessions.

USAGE (from inside src/):
    python run_analysis.py                  # all circuits in config.py
    python run_analysis.py Italian British  # only circuits whose name contains
                                            # any of these words (quick test)

Outputs:
    ../output/straight_events.csv
    ../output/corner_events.csv

Sessions that fail to load are skipped with a '!! Skipped' message.
First run downloads and caches each session (slow, a lot of disk space);
later runs reuse ../cache.
"""

import os
import sys
import traceback
from dataclasses import asdict

import pandas as pd
import fastf1

import config
from telemetry import load_session, build_traces
from straights import detect_straight_events
from corners import detect_corner_events


def main():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(config.CACHE_DIR)

    filters = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    straight_rows, corner_rows, loaded = [], [], []

    for era, sessions in config.SESSIONS.items():
        for (year, event, code) in sessions:
            if filters and not any(f in event.lower() for f in filters):
                continue
            print(f"\n=== {year} {event} ({era}) ===")
            try:
                session = load_session(year, event, code)
                traces, lap_len = build_traces(session)
                if len(traces) < 5:
                    raise RuntimeError(f"only {len(traces)} drivers with usable telemetry")
                print(f"  Drivers: {len(traces)}, lap length ~{lap_len:.0f} m")
                straight_rows += detect_straight_events(traces, year, event, era)
                corner_rows += detect_corner_events(traces, lap_len, year, event, era)
                loaded.append((era, year, event))
            except Exception as e:
                print(f"  !! Skipped {year} {event}: {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()

    S = pd.DataFrame([asdict(e) for e in straight_rows])
    C = pd.DataFrame([asdict(e) for e in corner_rows])
    S.to_csv(config.STRAIGHT_CSV, index=False)
    C.to_csv(config.CORNER_CSV, index=False)

    print("\n\n===== SUMMARY =====")
    print(f"Sessions loaded: {len(loaded)}")
    for era, year, event in loaded:
        ns = int(((S["event"] == event) & (S["year"] == year)).sum()) if len(S) else 0
        nc = int(((C["event"] == event) & (C["year"] == year)).sum()) if len(C) else 0
        print(f"  {era:9s} {year} {event:28s} straights={ns:4d} corners={nc:4d}")
    print(f"\nSaved {len(S)} straight and {len(C)} corner events to ../output/")
    print("Next: python analyse_eras.py")


if __name__ == "__main__":
    main()
