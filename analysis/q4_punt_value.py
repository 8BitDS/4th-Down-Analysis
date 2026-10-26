"""Q4. What is a punt actually worth? Expected points by field position + where punts leave you."""

from common import *

p = load("plays").filter(pl.col("season").is_between(*ERA))

# Expected points of a fresh 1st-and-10 at every yard line (nflverse EP model values, averaged)
ep = (p.filter(pl.col("down") == 1, (pl.col("ydstogo") == 10) | ((pl.col("goal_to_go") == 1) & (pl.col("ydstogo") == pl.col("yardline_100"))), pl.col("qtr").is_in([1, 3]))
      .group_by("yardline_100").agg(ep=pl.col("ep").mean(), n=pl.len()).sort("yardline_100"))
ep_at = dict(zip(ep["yardline_100"], ep["ep"]))
print("Expected points of 1st-and-10 (offence's view):")
for y in (95, 80, 75, 65, 50, 35, 20, 10, 5):
    side = f"own {100-y}" if y > 50 else ("midfield" if y == 50 else f"opp {y}")
    print(f"  {side:9s}: {ep_at[y]:+.2f}")

# Punts: where they're kicked from, where the opponent starts, and the value left behind
va = value_after(p).select("game_id", "play_id", "value_after")
pv = (load("punts").filter(pl.col("season").is_between(*ERA), pl.col("opp_start_yardline_100").is_not_null())
      .join(va, on=["game_id", "play_id"], how="left"))
pv = pv.with_columns(spot=(pl.col("yardline_100") / 5).floor() * 5)
tbl = (pv.group_by("spot").agg(n=pl.len(), net=pl.col("net_yards").median(),
                               opp_start=pl.col("opp_start_yardline_100").median(),
                               value=pl.col("value_after").mean(), ep_before=pl.col("ep").mean(),
                               touchback=pl.col("touchback").mean())
       .filter(pl.col("n") >= 50).sort("spot", descending=True))
print("\nPunts by line of scrimmage (yardline_100 bin): median net, opponent's median start (yds from THEIR goal), value after")
print(tbl.with_columns(pl.col(["value", "ep_before", "touchback"]).round(2)))

# Worked example for the maths corner: punt from own 35 (yardline_100 = 65)
ex = pv.filter(pl.col("yardline_100").is_between(63, 67))
ex_net = ex["net_yards"].median()
ex_start = int(ex["opp_start_yardline_100"].median())
print(f"\nMaths corner: punt from own 35 -> median net {ex_net:.0f} yds -> opponent 1st-and-10 at their own {100-ex_start} "
      f"-> their EP {ep_at[ex_start]:+.2f} -> worth {-ep_at[ex_start]:+.2f} to you. Avg actual value after: {ex['value_after'].mean():+.2f}")
# Compare: turnover on downs at your own 35 gives them the ball at your 35
print(f"  vs failing on 4th down at own 35: opponent at your 35 -> worth {-ep_at[35]:+.2f} to you. "
      f"Cost of failing instead of punting = {(-ep_at[35]) - (-ep_at[ex_start]):+.2f} pts")
print(f"  vs converting: 1st-and-10 around own 40 -> {ep_at[60]:+.2f}")
mid = pv.filter(pl.col("yardline_100").is_between(38, 42))
print(f"\nPunt from opp 40: median net {mid['net_yards'].median():.0f}, touchback rate {mid['touchback'].mean():.2f}, "
      f"value {mid['value_after'].mean():+.2f}")

# ---- chart: EP curve with punt arrows
fig, ax = new_fig()
x = 100 - ep["yardline_100"].to_numpy()          # yards from own goal line
ax.plot(x, ep["ep"].to_numpy(), color=C["blue"], lw=2.5)
ax.axhline(0, color=C["axis"], lw=1)
for y in (80, 50, 20):
    ax.scatter(100 - y, ep_at[y], s=60, color=C["blue"], edgecolor=C["surface"], lw=2, zorder=4)
    lab = {80: "Own 20", 50: "Midfield", 20: "Opp 20"}[y]
    ax.text(100 - y, ep_at[y] + 0.45, f"{lab}\n{ep_at[y]:+.1f} pts", ha="center", fontsize=11, color=C["ink"])
# Punt from own 35: arrow from punter's spot to where the opponent starts (mirrored onto the other team's curve)
y0 = 65
x0, x1 = 100 - y0, 100 - y0 + ex_net
ax.annotate("", xy=(x1, -3.2), xytext=(x0, -3.2),
            arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2, mutation_scale=16))
ax.text((x0 + x1) / 2, -3.0, f"Average punt from your own 35: {ex_net:.0f} net yards", ha="center", fontsize=11, color=C["ink2"])
ax.text((x0 + x1) / 2, -3.85,
        f"They start at their own {100-ex_start}, worth {ep_at[ex_start]:+.1f} to them = {-ep_at[ex_start]:+.1f} to you",
        ha="center", fontsize=11, color=C["ink2"])
ax.set_xlim(0, 100)
ax.set_ylim(-4.3, 6.2)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              ["Own\ngoal", "10", "20", "30", "40", "50", "40", "30", "20", "10", "Opp\ngoal"])
ax.set_ylabel("Expected points of a 1st-and-10 here")
finish(fig, ax, "q4_expected_points",
       "Every yard line has a points value. That's how you price a punt",
       f"Expected points (EP) for a 1st-and-10, {ERA[0]}–{ERA[1]}: the average net points the team with the ball "
       "goes on to score\nbefore the other side scores. Negative = the opponent is more likely to score next.",
       source="Data: nflverse play-by-play & nflverse EP model")
