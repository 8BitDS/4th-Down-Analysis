"""Test small feature changes to the replicated nflverse WP model.

Rule: a variant is only kept if it beats the replica on out-of-sample log loss AND the improvement's
90% interval (resampling whole games) stays above zero. Same seasons, same plays, same settings.
"""

import json

from wp_common import *

w = load_wp_data()
print(f"Missing team-strength values: {w['strength_time'].is_null().mean():.1%} of plays (mostly 1999, no prior season); "
      f"missing total_line: {w['total_line'].is_null().mean():.1%}")

base = pl.read_parquet(MODELS / "wp_replica_oof.parquet").select("game_id", "play_id", pl.col("ours_vegas_wp").alias("replica"))

VARIANTS = {
    "+ team strength (season-to-date EPA)": (FEATS_SPREAD + ["strength_time"], MONO_SPREAD + [0]),
    "+ over/under total": (FEATS_SPREAD + ["total_line"], MONO_SPREAD + [0]),
    "+ both": (FEATS_SPREAD + ["strength_time", "total_line"], MONO_SPREAD + [0, 0]),
}

ev = w.filter(pl.col("season") >= 2015).join(base, on=["game_id", "play_id"], how="left")
for i, (label, (feats, mono)) in enumerate(VARIANTS.items()):
    ev = ev.join(xgb_loso(w, feats, mono, f"v{i}"), on=["game_id", "play_id"], how="left")
    print(f"  fitted {label}")

out = {}
print("\nLog loss (lower = better) and improvement vs replica, with 90% interval from resampling games:")
for sub, sel in SUBSETS.items():
    f = sel(ev)
    r0 = score(f, "replica")
    print(f"\n{sub} (n={r0['n']:,}): replica log loss {r0['logloss']:.4f}, calibration {r0['cal']:.4f}")
    for i, label in enumerate(VARIANTS):
        r = score(f, f"v{i}")
        g, lo, hi = paired_gain(f, "replica", f"v{i}")
        verdict = "BETTER" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
        out[f"{sub}|{label}"] = dict(logloss=r["logloss"], cal=r["cal"], gain=g, lo=lo, hi=hi, verdict=verdict)
        print(f"  {label:38s} {r['logloss']:.4f} (cal {r['cal']:.4f}) | gain {g*1000:+.2f} [{lo*1000:+.2f}, {hi*1000:+.2f}] x1e-3 -> {verdict}")
(RES / "wp_variants_metrics.json").write_text(json.dumps(out, indent=2))
ev.select("game_id", "play_id", "replica", *[f"v{i}" for i in range(len(VARIANTS))]).write_parquet(MODELS / "wp_variants_oof.parquet")
