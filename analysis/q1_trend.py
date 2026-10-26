"""Q1. Teams go for it far more than they used to. How much, and since when?"""

from common import *

d = load("fourth_decisions").filter(pl.col("decision") != "fake_or_botched").with_columns(bucket())

# "Normal" situations: game still in the balance, not end-of-half desperation.
normal = (
    pl.col("wp").is_between(0.10, 0.90)
    & ~((pl.col("qtr") >= 4) & (pl.col("game_seconds_remaining") <= 300))
    & ~((pl.col("qtr") == 2) & (pl.col("half_seconds_remaining") <= 120))
)
dn = d.filter(normal)

# ---- overall go rate by season (all vs normal)
by_season = (
    d.group_by("season").agg(n_all=pl.len(), go_all=(pl.col("decision") == "go").sum())
    .join(dn.group_by("season").agg(n=pl.len(), go=(pl.col("decision") == "go").sum()), on="season")
    .sort("season")
    .with_columns(rate_all=pl.col("go_all") / pl.col("n_all"), rate=pl.col("go") / pl.col("n"))
)
print("\nGo rate by season (all 4th downs | normal situations):")
print(by_season.select("season", "n_all", pl.col("rate_all").round(3), "n", pl.col("rate").round(3)))

# ---- by bucket, normal situations
bb = (
    dn.group_by("season", "bucket").agg(n=pl.len(), go=(pl.col("decision") == "go").sum())
    .with_columns(rate=pl.col("go") / pl.col("n")).sort("season")
)
print("\nGo rate by bucket, normal situations (selected seasons):")
print(bb.filter(pl.col("season").is_in([1999, 2005, 2010, 2015, 2017, 2018, 2020, 2022, 2024, 2025, 2026]))
      .pivot(on="bucket", index="season", values="rate").sort("season"))

# ---- the headline comparison with a margin of error
def window(s0, s1, frame):
    w = frame.filter(pl.col("season").is_between(s0, s1))
    return w.height, (w["decision"] == "go").sum()

n_old, k_old = window(1999, 2017, dn)
n_new, k_new = window(2025, 2025, dn)
p_old, p_new = k_old / n_old, k_new / n_new
se = np.sqrt(p_old * (1 - p_old) / n_old + p_new * (1 - p_new) / n_new)
lo, hi = wilson(k_new, n_new)
print(f"\nNormal situations: 1999-2017 go rate {pct(p_old,1)} (n={n_old:,}) -> 2025 {pct(p_new,1)} (n={n_new:,})")
print(f"  2025 margin of error (90%): {pct(lo,1)} to {pct(hi,1)}; ratio {p_new/p_old:.2f}x; "
      f"difference {pct(p_new-p_old,1)} +/- {pct(1.645*se,1)}")

for b in BUCKETS:
    o = dn.filter(pl.col("bucket") == b)
    n0, k0 = window(1999, 2017, o)
    n1, k1 = window(2025, 2025, o)
    print(f"  {b:13s}: {pct(k0/n0,1)} -> {pct(k1/n1,1)} ({(k1/n1)/(k0/n0):.1f}x), 2025 n={n1}")

# Maths corner worked example: margin of error for one season's short-yardage rate
s25 = bb.filter(pl.col("season") == 2025, pl.col("bucket") == BUCKETS[0]).row(0, named=True)
p, n = s25["rate"], s25["n"]
moe = 1.645 * np.sqrt(p * (1 - p) / n)
print(f"\nMaths corner: 2025 short go rate p={p:.3f}, n={n}; SE = sqrt(p(1-p)/n) = {np.sqrt(p*(1-p)/n):.4f}; "
      f"90% margin = 1.645*SE = +/-{pct(moe,1)}")

# ---- chart: go rate by season, one line per bucket (normal situations)
fig, ax = new_fig()
for b in BUCKETS:
    s = bb.filter(pl.col("bucket") == b).sort("season")
    x, y = s["season"].to_numpy(), s["rate"].to_numpy()
    lo, hi = wilson(s["go"].to_numpy(), s["n"].to_numpy())
    col = BUCKET_COLORS[b]
    ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
    ax.plot(x[:-1], y[:-1], color=col, label=b)
    ax.plot(x[-2:], y[-2:], color=col, ls=(0, (2, 2)))  # 2026 is a partial season
    ax.scatter(x[-1], y[-1], s=55, facecolor=C["surface"], edgecolor=col, lw=2, zorder=3)
    label_y = {BUCKETS[0]: y[-1], BUCKETS[1]: 0.085, BUCKETS[2]: 0.025}[b]
    ax.text(2026.6, label_y, f"{b}: {pct(y[-2])} in 2025", va="center", fontsize=10.5, color=C["ink2"])
ax.axvline(2017.5, color=C["axis"], lw=1)
ax.text(2017.7, 0.62, "2018: the analytics era\nreally takes off", fontsize=10.5, color=C["ink2"], va="top")
ax.set_xlim(1998.5, 2033)
ax.set_ylim(0, 0.65)
ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Share of 4th downs where teams went for it")
ax.legend(loc="upper left", ncols=3, fontsize=10.5)
finish(fig, ax, "q1_go_rate_trend",
       f"Short-yardage go-for-it rate: {pct(window(1999,2017,dn.filter(pl.col('bucket')==BUCKETS[0]))[1]/window(1999,2017,dn.filter(pl.col('bucket')==BUCKETS[0]))[0])} → "
       f"{pct(bb.filter(pl.col('season')==2025, pl.col('bucket')==BUCKETS[0])['rate'][0])}",
       "4th downs in normal game situations (win prob. 10–90%, excl. last 5 min & end of 1st half).\n"
       "Shaded band = 90% margin of error. Hollow dot = 2026 season so far.")
