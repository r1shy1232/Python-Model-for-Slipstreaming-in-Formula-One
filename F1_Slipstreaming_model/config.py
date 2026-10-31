"""
Configuration (v3): drag + downforce slipstream model, 2021 (old regs) vs
2022 (new regs).

Only 2021 and 2022 sessions have been loading from FastF1's data source, so
the era comparison is 2021 vs 2022 at the SAME circuits (paired by circuit).
Circuits are chosen to be dry in both years with long straights and/or
high-speed corners. Names are FastF1 event names; any session that fails
to load is skipped.
"""

CIRCUITS = [
    "Italian Grand Prix",         # Monza
    "British Grand Prix",         # Silverstone (Copse, Maggotts/Becketts)
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",   # Jeddah
    "Azerbaijan Grand Prix",      # Baku
    "Austrian Grand Prix",
    "Spanish Grand Prix",         # Barcelona
    "Mexico City Grand Prix",
]
# Optional extras if you want more circuits:
#   "Abu Dhabi Grand Prix", "Sao Paulo Grand Prix"

SESSIONS = {
    "pre2022":  [(2021, name, "R") for name in CIRCUITS],
    "post2022": [(2022, name, "R") for name in CIRCUITS],
}

# --------------------------------------------------------------------------
# Straight-line events (drag effect)
# --------------------------------------------------------------------------
FOLLOWING_DISTANCE_THRESHOLD_M = 20.0   # gap to the car AHEAD
LATERAL_OFFSET_MAX_M = 5.0              # car ahead must be roughly in line
MIN_FOLLOWING_DURATION_S = 2.0
MIN_THROTTLE_PCT = 98.0
MIN_SPEED_KPH = 250.0                   # plateau gating (cube-law validity)
MAX_ACCEL_MS2 = 2.0

# Self-baseline: same driver, clean air, same place on track, same gear
CLEAN_AIR_GAP_S = 3.0
BASELINE_LAP_WINDOW = 8
BASELINE_RELDIST_TOL = 0.002
MIN_BASELINE_SAMPLES = 3
MAX_PLAUSIBLE_SPEED_DELTA_PCT = 10.0

# --------------------------------------------------------------------------
# Corner events (apparent downforce / grip effect)
# --------------------------------------------------------------------------
CORNER_MIN_APEX_KPH = 150.0             # aero-relevant corners only
CORNER_MIN_PROMINENCE_KPH = 15.0        # speed dip needed to count as a corner
CORNER_APPROACH_M = 150.0               # window start before apex
CORNER_EXIT_M = 100.0                   # window end after apex
CORNER_APEX_HALFWIDTH_M = 30.0          # apex speed = min speed within +/- this
CORNER_FOLLOW_GAP_S = 0.4               # time gap to car ahead (start & apex)
CORNER_DRS_LOOKBACK_M = 400.0           # DRS must be closed this far before apex
MIN_CLEAN_LAPS_CORNER = 3
MAX_PLAUSIBLE_APEX_DELTA_PCT = 15.0

# --------------------------------------------------------------------------
# Downforce model: assumed downforce-to-weight ratio at 250 km/h. It scales
# with v^2 at other speeds. These are ASSUMPTIONS shown as a sensitivity
# range; justify/cite a source in the paper.
# --------------------------------------------------------------------------
DOWNFORCE_D250_VALUES = [1.2, 2.0, 3.0]
DOWNFORCE_D250_CENTRAL = 2.0
LITERATURE_PENALTY_RATIO = 3.0          # ~60 % downforce loss / ~20 % drag red.

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
BOOTSTRAP_N = 2000
MIN_EVENTS_PER_CELL = 5                 # events needed per circuit-year cell
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Paths (run scripts from inside src/)
# --------------------------------------------------------------------------
CACHE_DIR = "../cache"
OUTPUT_DIR = "../output"
STRAIGHT_CSV = "../output/straight_events.csv"
CORNER_CSV = "../output/corner_events.csv"
