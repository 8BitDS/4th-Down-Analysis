"""Q2. What are the odds on 4th-and-X? Short / medium / long, run vs pass."""

import statsmodels.api as sm

from common import *

c = load("conv_plays").filter(pl.col("season").is_between(*ERA)).with_columns(bucket())
c4 = c.filter(pl.col("down") == 4)
c3 = c.filter(pl.col("down") == 3)


def rates(frame, by):
    t = frame.group_by(by).agg(n=pl.len(), k=pl.col("converted").sum()).sort(by)
    lo, hi = wilson(t["k"].to_numpy(), t["n"].to_numpy())
    return t.with_columns(rate=pl.col("k") / pl.col("n"), lo=pl.Series(lo), hi=pl.Series(hi))


print(f"4th-down go attempts {ERA[0]}-{ERA[1]}: {c4.height:,}")
b4 = rates(c4, ["bucket"])
b3 = rates(c3, ["bucket"])
print("\nConversion by bucket (4th down, with 90% range) vs 3rd down:")
for r4, r3 in zip(b4.iter_rows(named=True), b3.iter_rows(named=True)):
    print(f"  {r4['bucket']:13s}: 4th {pct(r4['rate'],1)} ({pct(r4['lo'])}-{pct(r4['hi'])}, n={r4['n']:,}) | 3rd {pct(r3['rate'],1)} (n={r3['n']:,})")

rp = rates(c4.with_columns(kind=pl.when((pl.col("is_pass") == 1) | (pl.col("qb_scramble") == 1)).then(pl.lit("Pass")).otherwise(pl.lit("Run"))), ["bucket", "kind"])
print("\nRun vs pass on 4th down:")
print(rp.select("bucket", "kind", "n", pl.col("rate").round(3), pl.col("lo").round(3), pl.col("hi").round(3)))
print("\nShare of attempts that are passes, by bucket:")
print(c4.group_by("bucket").agg(((pl.col("is_pass") == 1) | (pl.col("qb_scramble") == 1)).mean().round(3)).sort("bucket"))

# Per-yard detail + the S-curve
yd = rates(c4.filter(pl.col("ydstogo") <= 15), ["ydstogo"])
print("\nPer yard (4th down):")
print(yd.select("ydstogo", "n", pl.col("rate").round(3)))

# Logistic regression: log-odds = a + b * log(yards_to_go)  (fitted on 4th-and-1..15)
# log(distance) beats plain distance (AIC ~21 lower): the drop from 1 to 3 yards is steeper than 10 to 12.
fit_df = c4.filter(pl.col("ydstogo") <= 15)
y_ = fit_df["converted"].to_numpy()
x_ = fit_df["ydstogo"].to_numpy().astype(float)
m_lin = sm.Logit(y_, sm.add_constant(x_)).fit(disp=0)
m = sm.Logit(y_, sm.add_constant(np.log(x_))).fit(disp=0)
print(f"\nModel comparison (lower AIC = better): yards AIC {m_lin.aic:.1f} | log(yards) AIC {m.aic:.1f}")
a, b = m.params
odds_mult = 2 ** b  # effect of doubling the distance
lo_b, hi_b = b - 1.645 * m.bse[1], b + 1.645 * m.bse[1]
print(f"Logistic fit: log-odds = {a:.3f} + ({b:.3f}) x log(yards) -> doubling the distance multiplies the odds by "
      f"{odds_mult:.3f} ({pct(1-odds_mult,1)} lower). 90% range: {2**lo_b:.3f}-{2**hi_b:.3f}")
print(f"  (simpler per-yard version: each extra yard multiplies odds by {np.exp(m_lin.params[1]):.3f})")
for yards in (1, 2, 4, 8, 16):
    lo_ = a + b * np.log(yards)
    print(f"  {yards:2d} yards: log-odds {lo_:+.2f} -> odds {np.exp(lo_):.2f} -> p = {1/(1+np.exp(-lo_)):.3f}")

# Field position effect within short yardage (goal-line squeeze)
print("\nShort yardage by field position (4th down):")
print(rates(c4.filter(pl.col("bucket") == BUCKETS[0]).with_columns(
    zone=pl.when(pl.col("yardline_100") <= 5).then(pl.lit("a) inside 5"))
    .when(pl.col("yardline_100") <= 20).then(pl.lit("b) red zone 6-20"))
    .when(pl.col("yardline_100") <= 50).then(pl.lit("c) opp 21-50"))
    .otherwise(pl.lit("d) own half"))), ["zone"]).select("zone", "n", pl.col("rate").round(3)))

# ---- chart A: per-yard rates with the logistic S-curve, buckets shaded
fig, ax = new_fig()
spans = [(0.5, 3.5), (3.5, 9.5), (9.5, 15.5)]
for (x0, x1), bname in zip(spans, BUCKETS):
    ax.axvspan(x0, x1, color=BUCKET_COLORS[bname], alpha=0.07, lw=0)
    r = b4.filter(pl.col("bucket") == bname).row(0, named=True)
    ax.text((x0 + x1) / 2, 0.97, f"{bname.split()[0].upper()}\n{pct(r['rate'])} convert", ha="center", va="top",
            fontsize=11, color=C["ink"], fontweight="bold")
xs = np.linspace(1, 15, 200)
ax.plot(xs, 1 / (1 + np.exp(-(a + b * np.log(xs)))), color=C["ink2"], lw=1.5, label="Logistic fit on log(distance)")
for r in yd.iter_rows(named=True):
    bname = BUCKETS[0] if r["ydstogo"] <= 3 else BUCKETS[1] if r["ydstogo"] <= 9 else BUCKETS[2]
    col = BUCKET_COLORS[bname]
    ax.plot([r["ydstogo"]] * 2, [r["lo"], r["hi"]], color=col, lw=2, alpha=0.5)
    ax.scatter(r["ydstogo"], r["rate"], s=60, color=col, edgecolor=C["surface"], lw=2, zorder=3)
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0, 1)
ax.set_xticks(range(1, 16))
ax.set_xlabel("Yards to go on 4th down")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Conversion rate")
ax.legend(loc="lower left", fontsize=10.5)
finish(fig, ax, "q2_conversion_by_distance",
       f"Double the distance, and the odds of converting fall by ~{pct(1-odds_mult)}",
       f"4th-down conversion rate by distance, {ERA[0]}–{ERA[1]} (n = {c4.height:,} attempts). "
       "Dots = actual rate, bars = 90% margin of error.")


# QB sneaks: designed runs where the ball carrier is a QB who threw a pass for that team in that game
plays = load("plays").filter(pl.col("season").is_between(*ERA))
qbs = (plays.filter(pl.col("passer_player_name").is_not_null())
       .select("game_id", "posteam", pl.col("passer_player_name").alias("rusher_player_name")).unique()
       .with_columns(qb_run=pl.lit(1)))
KINDS = ["QB sneak", "Other run", "Pass"]
short = (c4.filter(pl.col("ydstogo") <= 3)
         .join(qbs, on=["game_id", "posteam", "rusher_player_name"], how="left")
         .with_columns(kind=pl.when((pl.col("is_pass") == 1) | (pl.col("qb_scramble") == 1)).then(pl.lit("Pass"))
                       .when(pl.col("qb_run") == 1).then(pl.lit("QB sneak")).otherwise(pl.lit("Other run"))))
sr = rates(short, ["ydstogo", "kind"])
print("\nShort yardage by play type (QB sneak = designed QB run):")
print(sr.select("ydstogo", "kind", "n", pl.col("rate").round(3), pl.col("lo").round(3), pl.col("hi").round(3)))
print("QB-sneak share of 4th-and-1 runs by season:")
print(short.filter(pl.col("ydstogo") == 1, pl.col("kind") != "Pass").group_by("season")
      .agg((pl.col("kind") == "QB sneak").mean().round(3).alias("qb_share")).sort("season"))

# ---- chart B: short yardage by play type (medium/long are ~98% passes, no fair comparison)
fig, ax = new_fig()
w = 0.26
kind_col = {"QB sneak": C["aqua"], "Other run": C["blue"], "Pass": C["orange"]}
for i, kind in enumerate(KINDS):
    sub = sr.filter(pl.col("kind") == kind, pl.col("n") >= 30, pl.col("ydstogo") <= 2).sort("ydstogo")
    xs_ = sub["ydstogo"].to_numpy() - 1 + (i - 1) * (w + 0.02)
    vals, ns = sub["rate"].to_numpy(), sub["n"].to_numpy()
    los, his = sub["lo"].to_numpy(), sub["hi"].to_numpy()
    ax.bar(xs_, vals, width=w, color=kind_col[kind], label=kind)
    ax.errorbar(xs_, vals, yerr=[vals - los, his - vals], fmt="none", ecolor=C["ink2"], lw=1.2, capsize=0)
    for x_, v, h_, n_ in zip(xs_, vals, his, ns):
        ax.text(x_, h_ + 0.02, pct(v), ha="center", fontsize=12.5, fontweight="bold")
        ax.text(x_, 0.02, f"n={n_:,}", ha="center", fontsize=8.5, color="white" if kind != "QB sneak" else C["ink"])
ax.set_xticks(range(2), ["4th-and-1", "4th-and-2"])
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Conversion rate")
ax.legend(loc="upper right", ncols=3, fontsize=11)
g = lambda yd, k: sr.filter(pl.col("ydstogo") == yd, pl.col("kind") == k)["rate"][0]  # noqa: E731
finish(fig, ax, "q2_run_vs_pass",
       f"4th-and-1: the QB sneak converts {pct(g(1,'QB sneak'))} of the time",
       f"4th-down conversion rate on short yardage, {ERA[0]}–{ERA[1]}. Lines = 90% margin of error. "
       "Bars with < 30 plays hidden.\nOn 4th-and-4+ teams pass ~98% of the time, so there's no fair run vs pass comparison.")
