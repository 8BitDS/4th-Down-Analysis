"""Q6. Does the scoreboard change the answer? And which coaches are bold vs timid?

Part A: go-for-it rate by score and time remaining.
Part B: coach aggressiveness = actual go rate - expected go rate for the same situations
        (expected from a model of what a league-average coach does in that season).
"""

from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from common import *

d = load("fourth_decisions").filter(pl.col("season") >= ERA[0], pl.col("decision") != "fake_or_botched").with_columns(
    bucket(), go=(pl.col("decision") == "go").cast(pl.Int8),
    coach=pl.when(pl.col("posteam_type") == "home").then(pl.col("home_coach")).otherwise(pl.col("away_coach")))

# ---------------------------------------------------------------- Part A: score x time
SCORE = [(-99, -9, "Down 9+"), (-8, -4, "Down 4-8"), (-3, -1, "Down 1-3"), (0, 0, "Tied"),
         (1, 3, "Up 1-3"), (4, 8, "Up 4-8"), (9, 99, "Up 9+")]
TIME = [("1st half", (pl.col("qtr") <= 2)),
        ("3rd qtr", (pl.col("qtr") == 3)),
        ("4th: >5 min", (pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") > 300)),
        ("4th: last 5 min", (pl.col("qtr") == 4) & (pl.col("game_seconds_remaining") <= 300))]
a = d.filter(pl.col("season").is_between(*ERA), pl.col("qtr") <= 4)
sc = pl.lit(None, dtype=pl.Int32)
for i, (lo, hi, _) in enumerate(SCORE):
    sc = pl.when(pl.col("score_differential").is_between(lo, hi)).then(pl.lit(i)).otherwise(sc)
tm = pl.lit(None, dtype=pl.Int32)
for j, (_, cond) in enumerate(TIME):
    tm = pl.when(cond).then(pl.lit(j)).otherwise(tm)
a = a.with_columns(sc.alias("sc"), tm.alias("tm"))
# Hold the situation roughly fixed: 4th-and-1 to 5, between the 30s... no: use all field positions but report per bucket
grid = a.filter(pl.col("ydstogo") <= 3).group_by("sc", "tm").agg(rate=pl.col("go").mean(), n=pl.len())
M = np.full((len(SCORE), len(TIME)), np.nan); N = np.zeros_like(M)
for r in grid.iter_rows(named=True):
    M[r["sc"], r["tm"]], N[r["sc"], r["tm"]] = r["rate"], r["n"]
print("Go rate on 4th-and-1 to 3 by score (rows) and time (cols):")
print(pl.DataFrame({"score": [s[2] for s in SCORE], **{t[0]: [round(x, 2) for x in M[:, j]] for j, t in enumerate(TIME)}}))

# Same field position & distance, different scoreboard: 4th-and-1-3 at opp 30-45
ex = a.filter(pl.col("ydstogo") <= 3, pl.col("yardline_100").is_between(30, 45))
print("\n4th-and-1-3 at opp 30-45, go rate by score in the 4th quarter (last 5 min) vs 1st half:")
print(ex.group_by("sc", "tm").agg(rate=pl.col("go").mean().round(2), n=pl.len())
      .filter(pl.col("tm").is_in([0, 3])).sort("sc", "tm"))

fig, ax = new_fig(10, 7)
ax.grid(False)
cm = LinearSegmentedColormap.from_list("bl", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
ax.imshow(M, cmap=cm, vmin=0, vmax=0.9, aspect="auto")
for i in range(len(SCORE)):
    for j in range(len(TIME)):
        ax.text(j, i, pct(M[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if M[i, j] > 0.5 else C["ink"])
ax.set_xticks(range(len(TIME)), [t[0] for t in TIME])
ax.set_yticks(range(len(SCORE)), [s[2] for s in SCORE])
ax.xaxis.tick_top()
ax.spines[:].set_visible(False)
finish(fig, ax, "q6_score_time",
       f"Same 4th-and-short: tied early, teams go {pct(M[3,0])}. Down 4-8 late: {pct(M[1,3])}",
       f"Go-for-it rate on 4th-and-short (1-3 yards) by score and time, {ERA[0]}–{ERA[1]}. Same field, same distance:\n"
       "the scoreboard changes the answer, because the goal is winning, not points.", top=0.74, left=0.11)

# ---------------------------------------------------------------- Part B: expected go rate model
FEATS = ["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining", "half_seconds_remaining",
         "qtr", "wp", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "season"]
b = d.filter(pl.col("wp").is_between(0.05, 0.95), pl.col("qtr") <= 4).drop_nulls(FEATS + ["coach"])
X, y, groups = b.select(FEATS).to_numpy(), b["go"].to_numpy(), b["game_id"].to_numpy()
oof = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):  # out-of-fold: a coach's own games never train their prediction
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]
print(f"\nExpected-go model (out-of-fold): log loss {log_loss(y, oof):.3f}, Brier {brier_score_loss(y, oof):.4f}, "
      f"base rate {y.mean():.3f}. Calibration by decile:")
b = b.with_columns(exp=pl.Series(oof))
print(b.with_columns(dec=(pl.col("exp").rank() / b.height * 10).floor().clip(0, 9))
      .group_by("dec").agg(pred=pl.col("exp").mean().round(3), actual=pl.col("go").mean().round(3)).sort("dec"))

# Aggressiveness per coach since 2025 (current coaches), plus GB/DET history
def agg(frame, by):
    return frame.group_by(by).agg(
        n=pl.len(), go=pl.col("go").sum(), exp=pl.col("exp").sum(),
        var=(pl.col("exp") * (1 - pl.col("exp"))).sum(),
        team=pl.col("posteam").mode().first(),
    ).with_columns(
        over=(pl.col("go") - pl.col("exp")) / pl.col("n"),
        moe=1.645 * pl.col("var").sqrt() / pl.col("n"),
        go_rate=pl.col("go") / pl.col("n"),
    )

recent = agg(b.filter(pl.col("season") >= 2025), "coach").filter(pl.col("n") >= 60).sort("over", descending=True)
print(f"\nCoach aggressiveness 2025-{CUR} (go rate above/below expected, percentage points):")
print(recent.select("coach", "team", "n", "go", pl.col("exp").round(1), pl.col("go_rate").round(3),
                    (pl.col("over") * 100).round(1).alias("over_pp"), (pl.col("moe") * 100).round(1).alias("moe_pp")))

team_hist = agg(b.filter(pl.col("posteam").is_in(["GB", "DET"])), ["posteam", "season", "coach"]).sort("posteam", "season")
print("\nGB & DET by season:")
print(team_hist.select("posteam", "season", "coach", "n", "go", pl.col("exp").round(1),
                       (pl.col("over") * 100).round(1).alias("over_pp"), (pl.col("moe") * 100).round(1).alias("moe_pp")))

# Maths corner: a residual in plain numbers
r = recent.filter(pl.col("team") == "DET").row(0, named=True)
print(f"\nMaths corner: {r['coach']} ({r['team']}) faced {r['n']} 4th downs. A league-average coach would have gone "
      f"{r['exp']:.1f} times; he went {r['go']}. Residual = ({r['go']} - {r['exp']:.1f}) / {r['n']} = {r['over']*100:+.1f} pts "
      f"(+/- {r['moe']*100:.1f} from luck alone)")

# ---- chart: coach leaderboard, GB and DET highlighted
rec = recent.sort("over")
fig, ax = plt.subplots(figsize=(10, 11.5))
ax.tick_params(length=0)
ax.grid(axis="y", visible=False)
ys = np.arange(rec.height)
cols = [TEAM_COLORS.get(t, "#c3c2b7") for t in rec["team"]]
ax.barh(ys, rec["over"].to_numpy() * 100, height=0.62, color=cols)
ax.errorbar(rec["over"].to_numpy() * 100, ys, xerr=rec["moe"].to_numpy() * 100, fmt="none", ecolor=C["muted"], lw=1)
ax.axvline(0, color=C["ink2"], lw=1)
labels = [f"{c} ({t})" for c, t in zip(rec["coach"], rec["team"])]
ax.set_yticks(ys, labels, fontsize=10)
for tick, t in zip(ax.get_yticklabels(), rec["team"]):
    if t in TEAM_COLORS:
        tick.set_fontweight("bold"); tick.set_color(C["ink"]); tick.set_fontsize(11.5)
for y_, v, t in zip(ys, rec["over"].to_numpy() * 100, rec["team"]):
    if t in TEAM_COLORS:
        m_ = rec.filter(pl.col("team") == t)["moe"][0] * 100
        ax.text(v + m_ + 0.3 if v >= 0 else v - m_ - 0.3, y_, f"{v:+.1f} pts", va="center", ha="left" if v >= 0 else "right",
                fontsize=11, fontweight="bold")
ax.set_xlabel("Go-for-it rate vs a league-average coach in the same situations (percentage points)")
ax.set_ylim(-0.7, len(ys) - 0.3)
ax.set_xlim(-11, 13)
fig.subplots_adjust(top=0.885, bottom=0.07, left=0.30, right=0.97)
fig.text(0.02, 0.97, "Who's bold on 4th down?", fontsize=19, fontweight="bold", va="top")
fig.text(0.02, 0.935, f"Head coaches, 2025–{CUR} season so far (min. 60 4th downs, win prob. 5–95%). Bar = how much more\n"
         "(or less) often they go for it than expected. Thin line = 90% range from luck alone.",
         fontsize=12, color=C["ink2"], va="top")
fig.text(0.02, 0.015, "Data: nflverse play-by-play. Expected rate: gradient-boosted model of league decisions by season.",
         fontsize=9.5, color=C["muted"])
fig.savefig(FIG / "q6_coach_aggressiveness.png")
plt.close(fig)
print("  -> figures/q6_coach_aggressiveness.png")
