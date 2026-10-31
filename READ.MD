# Slipstream model v3: drag + downforce, 2021 (old regs) vs 2022 (new regs)

## Pipeline
1. `run_analysis.py` loads race sessions via FastF1 and writes
   - `output/straight_events.csv`: speed gain, effective C_D ratio, drag reduction, time gained
   - `output/corner_events.csv`: apex speed loss, entry speed, braking-point shift, corner time lost
2. `analyse_eras.py` adds a modelled downforce loss (assumption range), the
   slipstream penalty ratio, paired-by-circuit comparisons, cluster-bootstrap
   confidence intervals and figures (`output/`).
3. `test_synthetic.py` self-tests everything offline with synthetic cars.

## Run (from inside src/)
    python test_synthetic.py                 # optional, offline
    python run_analysis.py Italian British   # quick test on two circuits
    python run_analysis.py                   # all circuits in config.py (slow, several GB cache)
    python analyse_eras.py

Add `--debug` to run_analysis.py to print full tracebacks for skipped sessions.

## What is compared
Always the SAME driver against their OWN clean-air laps (same part of the lap,
within +/- 8 laps, valid green-flag laps only). Straights: following < 20 m,
full throttle, DRS closed, >= 250 km/h, near-constant speed, same gear.
Corners: nearest car ahead within 0.4 s at approach start and apex, DRS closed
for 400 m before the apex.

## Downforce is modelled, not measured
Telemetry has no downforce channel. Apex speed loss is the empirical result.
Downforce loss uses a grip model with an ASSUMED downforce-to-weight ratio at
250 km/h (config.DOWNFORCE_D250_VALUES) shown as a sensitivity range.

## Known limitations (for the methodology section)
- Equal engine power between compared laps is assumed, not verified (gear matching is a partial control).
- C_D ratio is an effective ratio (parasitic + induced drag).
- Drag is measured at >= 250 km/h on straights, downforce at 150+ km/h corners,
  so the penalty ratio compares effects at different speeds.
- Following cars take different lines and drivers back off in dirty air; "apparent
  grip loss" includes driver and tyre effects, not only aerodynamics.
- Excluding DRS-open running biases the sample toward the wider part of the gap band.
- One race per circuit per year: weather, strategy and incidents are not controlled.
- 2022 cars differ in mass, tyres and fuel, and several showed porpoising.
- Only 2021 and 2022 have been loading from FastF1's data source.
