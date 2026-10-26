"""Q3. How long is too long for a kicker? FG% by distance (then vs now) + shrinkage for small samples."""

import statsmodels.api as sm

from common import *

f = load("fg_attempts")
ERAS = [(1999, 2005), (2006, 2014), (2015, 2025)]
ERA_COL = [C["aqua"], C["orange"], C["blue"]]


def logit_fit(frame):
    x = frame["kick_distance"].to_numpy().astype(float)
    return sm.Logit(frame["made"].to_numpy(), sm.add_constant(x)).fit(disp=0)


def p_at(m, d):
    return 1 / (1 + np.exp(-(m.params[0] + m.params[1] * np.asarray(d, float))))


fits = {}
print("FG make rate by distance band and era:")
for e in ERAS:
    sub = f.filter(pl.col("season").is_between(*e))
    m = fits[e] = logit_fit(sub)
    d50 = -m.params[0] / m.params[1]
    bands = sub.group_by("dist_band").agg(rate=pl.col("made").mean().round(3), n=pl.len()).sort("dist_band")
    print(f"  {e[0]}-{e[1]}: n={sub.height:,}; 50/50 distance = {d50:.1f} yds; "
          f"model p(50 yds)={p_at(m,50):.3f}, p(55)={p_at(m,55):.3f}; odds x{np.exp(m.params[1]):.3f} per extra yard")
    print("    ", dict(zip(bands["dist_band"], zip(bands["rate"], bands["n"]))))

now = fits[ERAS[-1]]
a, b = now.params
print(f"\nMaths corner ({ERAS[-1][0]}-{ERAS[-1][1]}): log-odds = {a:.3f} + ({b:.4f}) x distance")
for d in (40, 50, 60):
    lo = a + b * d
    print(f"  {d} yds: log-odds {lo:+.2f} -> p = 1/(1+e^-({lo:.2f})) = {1/(1+np.exp(-lo)):.3f}")
# Field position -> kick distance rule of thumb
chk = f.filter(pl.col("season").is_between(2015, 2025)).select((pl.col("kick_distance") - pl.col("yardline_100")).alias("d"))
print(f"\nKick distance = yard line + {chk['d'].median():.0f} (median; 10 end zone + ~7 snap)")

# ---- chart A: S-curves by era + dots for modern era
fig, ax = new_fig()
xs = np.linspace(18, 66, 200)
for e, col in zip(ERAS, ERA_COL):
    ax.plot(xs, p_at(fits[e], xs), color=col, label=f"{e[0]}–{e[1]}")
mod = (f.filter(pl.col("season").is_between(*ERAS[-1]), pl.col("kick_distance").is_between(18, 66))
       .group_by("kick_distance").agg(k=pl.col("made").sum(), n=pl.len()).filter(pl.col("n") >= 15).sort("kick_distance"))
ax.scatter(mod["kick_distance"], mod["k"] / mod["n"], s=mod["n"] / 6 + 10, color=ERA_COL[-1], alpha=0.35,
           edgecolor=C["surface"], lw=1, zorder=3, label="2015–2025 actual (dot size = attempts)")
for e, col, dy in zip(ERAS, ERA_COL, (-0.05, 0.0, 0.05)):
    d50 = -fits[e].params[0] / fits[e].params[1]
    ax.scatter([d50], [0.5], s=70, color=col, edgecolor=C["surface"], lw=2, zorder=4)
ax.axhline(0.5, color=C["axis"], lw=1)
d_old = -fits[ERAS[0]].params[0] / fits[ERAS[0]].params[1]
d_new = -fits[ERAS[-1]].params[0] / fits[ERAS[-1]].params[1]
ax.text((d_old + d_new) / 2, 0.44, f"50/50 distance:\n{d_old:.0f} yds → {d_new:.0f} yds", ha="center", va="top",
        fontsize=11, color=C["ink2"])
ax.set_xlim(17, 67)
ax.set_ylim(0, 1.04)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Field goal distance (yards)")
ax.set_ylabel("Make rate")
ax.legend(loc="lower left", fontsize=10.5)
finish(fig, ax, "q3_fg_by_distance",
       f"A 55-yarder used to be a long shot. Now it goes in {pct(p_at(now,55))} of the time",
       "Field goal make rate by distance, logistic fit per era. Rule of thumb: kick distance ≈ yard line + 18.")

# ---- shrinkage: how good is a kicker REALLY after a handful of kicks?
# Expected make prob for every kick from the league curve of its era.
f2 = f.filter(pl.col("season") >= 2015).with_columns(exp=pl.Series(p_at(now, f.filter(pl.col("season") >= 2015)["kick_distance"])))
ks = f2.filter(pl.col("season").is_between(2015, 2025)).group_by("kicker_player_id", "season").agg(
    n=pl.len(), made=pl.col("made").sum(), exp=pl.col("exp").sum()).filter(pl.col("n") >= 10)
# Between-kicker skill variance = observed variance of (actual - expected) rate minus pure luck (binomial noise)
diff = (ks["made"] - ks["exp"]) / ks["n"]
noise = (ks["exp"] / ks["n"] * (1 - ks["exp"] / ks["n"]) / ks["n"]).mean()
tau2 = max(diff.var() - noise, 1e-5)
pbar = (ks["exp"] / ks["n"]).mean()
m_prior = pbar * (1 - pbar) / tau2 - 1
print(f"\nShrinkage (kicker-seasons {2015}-{2025}, n>=10): spread of true skill sd={np.sqrt(tau2):.3f}, "
      f"luck variance per kick-season={noise:.4f} -> prior worth m = {m_prior:.0f} league-average kicks")

cur = f2.filter(pl.col("season") == CUR).group_by("kicker_player_id", "kicker_player_name").agg(
    team=pl.col("posteam").last(), n=pl.len(), made=pl.col("made").sum(), exp=pl.col("exp").sum()
).filter(pl.col("n") >= 5).with_columns(
    raw=pl.col("made") / pl.col("n"),
    exp_rate=pl.col("exp") / pl.col("n"),
    shrunk=(pl.col("made") + m_prior * pl.col("exp") / pl.col("n")) / (pl.col("n") + m_prior),
).sort("shrunk")
print(f"\n{CUR} kickers (>=5 attempts): raw vs shrunk FG% (shrunk toward league rate for the same kick distances)")
print(cur.select("kicker_player_name", "team", "n", "made", pl.col("raw").round(3), pl.col("exp_rate").round(3), pl.col("shrunk").round(3)).sort("raw", descending=True))
for t in ("DET", "GB"):
    r = cur.filter(pl.col("team") == t).row(0, named=True)
    print(f"  {t} {r['kicker_player_name']}: {r['made']}/{r['n']} = {pct(r['raw'])} raw -> "
          f"({r['made']} + {m_prior:.0f} x {r['exp_rate']:.3f}) / ({r['n']} + {m_prior:.0f}) = {pct(r['shrunk'],1)}")

# ---- chart B: raw (hollow) -> shrunk (filled) for every 2026 kicker; GB and DET highlighted
cur = cur.sort("shrunk")
fig, ax = plt.subplots(figsize=(10, 11))
ax.tick_params(length=0)
ax.grid(axis="y", visible=False)
ys = np.arange(cur.height)
for y, r in zip(ys, cur.iter_rows(named=True)):
    hl = r["team"] in TEAM_COLORS
    col = TEAM_COLORS.get(r["team"], C["muted"])
    ax.plot([r["raw"], r["shrunk"]], [y, y], color=col, lw=2 if hl else 1.2, alpha=1 if hl else 0.6)
    ax.scatter(r["raw"], y, s=60, facecolor=C["surface"], edgecolor=col, lw=2, zorder=3)
    ax.scatter(r["shrunk"], y, s=70, color=col, edgecolor=C["surface"], lw=2, zorder=4)
ax.axvline(cur["exp_rate"].mean(), color=C["axis"], lw=1)
labels = [f"{r['kicker_player_name']} ({r['team']})  {r['made']}/{r['n']}" for r in cur.iter_rows(named=True)]
ax.set_yticks(ys, labels, fontsize=10)
for tick, r in zip(ax.get_yticklabels(), cur.iter_rows(named=True)):
    if r["team"] in TEAM_COLORS:
        tick.set_fontweight("bold"); tick.set_color(C["ink"]); tick.set_fontsize(11.5)
ax.set_xlim(0.53, 1.02)
ax.set_ylim(-0.8, len(ys) - 0.2)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Field goal %   (hollow = raw this season, filled = shrunk estimate)")
fig.subplots_adjust(top=0.88, bottom=0.07, left=0.27, right=0.97)
fig.text(0.02, 0.965, "Is an 8/8 kicker really a 100% kicker? No.", fontsize=19, fontweight="bold", va="top")
fig.text(0.02, 0.928, f"{CUR} kickers so far. Shrinkage adds {m_prior:.0f} 'imaginary' league-average kicks at the same distances,\n"
         "so small samples get pulled toward the league rate. More kicks = less pull.", fontsize=12, color=C["ink2"], va="top")
fig.text(0.02, 0.015, "Data: nflverse play-by-play", fontsize=9.5, color=C["muted"])
fig.savefig(FIG / "q3_kicker_shrinkage.png")
plt.close(fig)
print("  -> figures/q3_kicker_shrinkage.png")
