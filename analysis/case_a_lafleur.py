"""Case A caveat: LaFleur calls GB's plays. How good is GB's 4th-down offence under him, and
how much should that move P(convert) for the 4th-and-4 at the DET 7?

Method (same idea as the kicker shrinkage in Q3):
  1. League model gives each attempt an expected conversion prob (distance + goal-line squeeze).
  2. A team's skill = actual conversions - expected, per attempt ("conversions over expected", CoE).
  3. Shrink CoE toward 0 with m imaginary league-average attempts, where m comes from how much
     real team-to-team spread there is vs pure luck.
"""

import statsmodels.api as sm

from common import *

PLAY = dict(ydstogo=4, yardline_100=7)  # Case A
c = load("conv_plays")
c4 = c.filter(pl.col("down") == 4)


def design(frame):
    y = frame["yardline_100"].to_numpy().astype(float)
    return np.column_stack([np.ones(frame.height), np.log(frame["ydstogo"].to_numpy().astype(float)),
                            (y <= 5).astype(float), (y <= 10).astype(float)])


train = c4.filter(pl.col("season").is_between(2015, 2025))
m = sm.Logit(train["converted"].to_numpy(), design(train)).fit(disp=0)
p_league = float(m.predict(design(pl.DataFrame({"ydstogo": [PLAY["ydstogo"]], "yardline_100": [PLAY["yardline_100"]]})))[0])
print(f"League P(convert) for 4th-and-4 at the 7: {pct(p_league,1)}")

c4 = c4.filter(pl.col("season") >= 2015).with_columns(exp=pl.Series(m.predict(design(c4.filter(pl.col("season") >= 2015)))))
c4 = c4.with_columns(coach=pl.when(pl.col("posteam_type") == "home").then(pl.col("home_coach")).otherwise(pl.col("away_coach")))

# How much do offences REALLY differ? team-season spread of CoE minus luck
ts = c4.filter(pl.col("season").is_between(2015, 2025)).group_by("posteam", "season").agg(
    n=pl.len(), k=pl.col("converted").sum(), e=pl.col("exp").sum(), v=(pl.col("exp") * (1 - pl.col("exp"))).sum())
coe = (ts["k"] - ts["e"]) / ts["n"]
luck = (ts["v"] / ts["n"] ** 2).mean()
tau2 = max(coe.var() - luck, 1e-6)
pbar = (ts["e"] / ts["n"]).mean()
m_prior = pbar * (1 - pbar) / tau2 - 1
print(f"Team-season spread in 4th-down skill: sd = {np.sqrt(tau2)*100:.1f} pts (observed {coe.std()*100:.1f}, "
      f"luck alone {np.sqrt(luck)*100:.1f}) -> prior worth {m_prior:.0f} league-average attempts")
print("  i.e. most of the year-to-year difference between offences on 4th down is luck.")

# Year-to-year stability check: does a team's CoE this year predict next year?
yy = ts.with_columns(coe=(pl.col("k") - pl.col("e")) / pl.col("n")).sort("posteam", "season")
yy = yy.with_columns(nxt=pl.col("coe").shift(-1).over("posteam")).drop_nulls("nxt")
print(f"Year-to-year correlation of a team's 4th-down CoE: r = {np.corrcoef(yy['coe'], yy['nxt'])[0,1]:.2f}")

gb = c4.filter(pl.col("coach") == "Matt LaFleur")


def summary(frame, label):
    n, k, e = frame.height, int(frame["converted"].sum()), float(frame["exp"].sum())
    lo, hi = wilson(k, n)
    co = (k - e) / n
    shrunk = (k - e) / (n + m_prior)
    print(f"  {label:34s} {k:3d}/{n:<3d} = {pct(k/n):>4s} (90% range {pct(lo)}-{pct(hi)}) | expected {pct(e/n):>4s} | "
          f"over exp {co*100:+5.1f} pts -> shrunk {shrunk*100:+5.1f} pts")
    return dict(label=label, n=n, k=k, exp=e / n, rate=k / n, lo=lo, hi=hi, coe=co, shrunk=shrunk)


print(f"\nGreen Bay 4th-down attempts under Matt LaFleur ({gb['season'].min()}-{gb['season'].max()}):")
rows = [
    summary(gb, "All attempts"),
    summary(gb.filter(pl.col("season") <= 2022), "Rodgers era (2019-22)"),
    summary(gb.filter(pl.col("season") >= 2023), "Love era (2023-26)"),
]
for bname in BUCKETS:
    rows.append(summary(gb.with_columns(bucket()).filter(pl.col("bucket") == bname), bname))
rows.append(summary(gb.filter(pl.col("yardline_100") <= 10), "Inside the 10"))
rows.append(summary(gb.filter(pl.col("ydstogo").is_between(3, 5)), "4th-and-3 to 5"))

# 3rd downs: same play-caller, ~10x the sample, similar distances (3-5 yds)
c3 = (c.filter(pl.col("down") == 3, pl.col("season") >= 2019, pl.col("ydstogo").is_between(3, 5))
      .with_columns(coach=pl.when(pl.col("posteam_type") == "home").then(pl.col("home_coach")).otherwise(pl.col("away_coach"))))
lg = c3.filter(pl.col("season") <= 2025)["converted"].mean()
g3 = c3.filter(pl.col("coach") == "Matt LaFleur")
g3l = c3.filter(pl.col("coach") == "Matt LaFleur", pl.col("season") >= 2023)
print(f"\n3rd-and-3 to 5 (bigger sample of his play-calling): LaFleur {pct(g3['converted'].mean(),1)} (n={g3.height}) vs league "
      f"{pct(lg,1)}; Love era {pct(g3l['converted'].mean(),1)} (n={g3l.height})")
rz = c.filter(pl.col("down").is_in([3, 4]), pl.col("season") >= 2019, pl.col("yardline_100") <= 10, pl.col("ydstogo").is_between(3, 5)).with_columns(
    coach=pl.when(pl.col("posteam_type") == "home").then(pl.col("home_coach")).otherwise(pl.col("away_coach")))
print(f"3rd/4th-and-3 to 5 inside the 10: LaFleur {pct(rz.filter(pl.col('coach')=='Matt LaFleur')['converted'].mean(),1)} "
      f"(n={rz.filter(pl.col('coach')=='Matt LaFleur').height}) vs league {pct(rz.filter(pl.col('season')<=2025)['converted'].mean(),1)}")

# Plug into the play
allr = rows[0]
p_gb = p_league + allr["shrunk"]
print(f"\nCase A P(convert): league {pct(p_league,1)} -> GB/LaFleur-adjusted {pct(p_gb,1)} "
      f"(raw record would say {pct(p_league + allr['coe'],1)})")
print("Effect on WP(GO) = change in P(convert) x (WP if convert - WP if fail).")
for spread in (0.25, 0.35, 0.45):
    print(f"  if convert is worth {spread*100:.0f} WP pts more than failing: WP(GO) moves by {allr['shrunk']*spread*100:+.1f} pts")

# ---------------------------------------------------------------------------
# Bigger sample: 3rd AND 4th downs at 1-6 yards (same play-caller, ~10x the plays)
# ---------------------------------------------------------------------------
print("\n=== Using 3rd + 4th downs, 1-6 yards to go ===")
s = c.filter(pl.col("down").is_in([3, 4]), pl.col("ydstogo") <= 6, pl.col("season") >= 2015).with_columns(
    coach=pl.when(pl.col("posteam_type") == "home").then(pl.col("home_coach")).otherwise(pl.col("away_coach")))


def design2(frame):
    y = frame["yardline_100"].to_numpy().astype(float)
    return np.column_stack([np.ones(frame.height), np.log(frame["ydstogo"].to_numpy().astype(float)),
                            (frame["down"].to_numpy() == 4).astype(float), (y <= 5).astype(float), (y <= 10).astype(float)])


tr = s.filter(pl.col("season").is_between(2015, 2025))
m2 = sm.Logit(tr["converted"].to_numpy(), design2(tr)).fit(disp=0)
s = s.with_columns(exp=pl.Series(m2.predict(design2(s))))
ts2 = s.filter(pl.col("season").is_between(2015, 2025)).group_by("posteam", "season").agg(
    n=pl.len(), k=pl.col("converted").sum(), e=pl.col("exp").sum(), v=(pl.col("exp") * (1 - pl.col("exp"))).sum())
coe2 = (ts2["k"] - ts2["e"]) / ts2["n"]
luck2 = (ts2["v"] / ts2["n"] ** 2).mean()
tau2b = max(coe2.var() - luck2, 1e-6)
pbar2 = (ts2["e"] / ts2["n"]).mean()
m2_prior = pbar2 * (1 - pbar2) / tau2b - 1
yy2 = ts2.with_columns(coe=(pl.col("k") - pl.col("e")) / pl.col("n")).sort("posteam", "season").with_columns(
    nxt=pl.col("coe").shift(-1).over("posteam")).drop_nulls("nxt")
print(f"Plays per team-season: ~{ts2['n'].mean():.0f}. Skill spread sd = {np.sqrt(tau2b)*100:.1f} pts "
      f"(observed {coe2.std()*100:.1f}, luck {np.sqrt(luck2)*100:.1f}); year-to-year r = {np.corrcoef(yy2['coe'], yy2['nxt'])[0,1]:.2f}; "
      f"prior worth m = {m2_prior:.0f} attempts")

g = s.filter(pl.col("coach") == "Matt LaFleur")
out = []
for label, f_ in [("LaFleur 2019-26", g), ("Rodgers era 2019-22", g.filter(pl.col("season") <= 2022)),
                  ("Love era 2023-26", g.filter(pl.col("season") >= 2023)), ("2026 so far", g.filter(pl.col("season") == 2026)),
                  ("Inside the 10", g.filter(pl.col("yardline_100") <= 10))]:
    n, k, e = f_.height, int(f_["converted"].sum()), float(f_["exp"].sum())
    co, sh = (k - e) / n, (k - e) / (n + m2_prior)
    out.append((label, n, k, e, co, sh))
    print(f"  {label:22s} {k:4d}/{n:<4d} = {pct(k/n,1):>5s} | expected {pct(e/n,1):>5s} | over exp {co*100:+5.1f} pts "
          f"-> shrunk {sh*100:+5.1f} pts  (weight on his record: {n/(n+m2_prior):.0%})")

love = out[2]
p_gb2 = p_league + love[5]
print(f"\nCase A P(convert) for 4th-and-4 at the 7: league {pct(p_league,1)} -> Love-era GB adjusted {pct(p_gb2,1)}")
for spread in (0.25, 0.35, 0.45):
    print(f"  if converting is worth {spread*100:.0f} WP pts more than failing: WP(GO) moves by {love[5]*spread*100:+.2f} pts")
