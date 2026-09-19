"""
Profile tsauditor's scan() on YOUR actual dataframe, to find out what's
actually slow for you -- the two fixes just delivered got a synthetic
3279x27 dataframe from 12.1s down to ~2.4s, but that's nowhere near your
reported 12 minutes, so something in your real data or environment is
different and needs to be identified rather than guessed at.

Usage:
    python profile_your_scan.py path/to/your_data.csv

Adjust the pd.read_csv / index-setting / scan() call below to match how you
normally load and call it (target=, group_col=, time_col=, whatever you pass).
"""

import sys
import cProfile
import pstats
import pandas as pd

import tsauditor as tsa

path = sys.argv[1] if len(sys.argv) > 1 else "your_data.csv"

df = pd.read_csv(path, index_col=0, parse_dates=True)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

profiler = cProfile.Profile()
profiler.enable()

report = tsa.scan(df)  # <-- add your usual arguments here (target=, etc.)

profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
print("\n=== Top 30 by cumulative time ===")
stats.print_stats(30)

print("\n=== Top 20 by internal (tottime) time (finds hot loops, not just callers) ===")
stats.sort_stats("tottime")
stats.print_stats(20)
