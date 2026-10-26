"""Q5. When do you only need a ~40% chance to go for it? Break-even conversion rates vs what coaches do.

For each (distance bucket x field-position zone):
    V_conv = expected points after a conversion           (3rd+4th down conversions: same next state)
    V_fail = expected points after failing on 4th down    (turnover on downs: minus the opponent's
             1st-and-10 expected points at that spot - robust where real failed attempts are rare)
    V_kick = best of punt or field goal attempt
    break-even p* = (V_kick - V_fail) / (V_conv - V_fail)
Go for it if P(convert) > p*.
"""

import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import *

plays = value_after(load("plays").filter(pl.col("season").is_between(*ERA)))
not_eoh = ~((pl.col("half_seconds_remaining") <= 120))  # end-of-half plays have their own logic
ZONES = [(91, 99, "Own\n1-9"), (81, 90, "Own\n10-19"), (71, 80, "Own\n20-29"), (61, 70, "Own\n30-39"),
         (51, 60, "Own\n40-49"), (41, 50, "Opp\n41-50"), (31, 40, "Opp\n31-40"), (21, 30, "Opp\n21-30"),
         (11, 20, "Opp\n11-20"), (1, 10, "Opp\n1-10")]
zone = pl.lit(None, dtype=pl.Int32)
for i, (lo, hi, _) in enumerate(ZONES):
    zone = pl.when(pl.col("yardline_100").is_between(lo, hi)).then(pl.lit(i)).otherwise(zone)
normal = pl.col("wp").is_between(0.10, 0.90) & ~((pl.col("qtr") >= 4) & (pl.col("game_seconds_remaining") <= 300))
# 1st-and-10 expected points by yard line (same table as Q4) for the turnover-on-downs value
ep1 = (plays.filter(pl.col("down") == 1, (pl.col("ydstogo") == 10) | ((pl.col("goal_to_go") == 1) & (pl.col("ydstogo") == pl.col("yardline_100"))),
                    pl.col("qtr").is_in([1, 3])).group_by("yardline_100").agg(ep1=pl.col("ep").mean()))
plays = plays.with_columns(zone.alias("zone"), bucket()).filter(not_eoh, normal, pl.col("value_after").is_not_null())

fourth = plays.filter(pl.col("down") == 4)
conv = plays.filter(pl.col("down").is_in([3, 4]), pl.col("play_type").is_in(["run", "pass"]),
                    pl.when(pl.col("down") == 3).then(pl.col("third_down_converted")).otherwise(pl.col("fourth_down_converted")) == 1)

v_conv = conv.group_by("bucket", "zone").agg(V_conv=pl.col("value_after").mean(), n_conv=pl.len())
v_fail = (fourth.with_columns(opp_yl=100 - pl.col("yardline_100")).join(ep1.rename({"yardline_100": "opp_yl"}), on="opp_yl", how="left")
          .group_by("zone").agg(V_fail=(-pl.col("ep1")).mean(), n_fail=pl.len()))
v_punt = fourth.filter(pl.col("play_type") == "punt").group_by("zone").agg(V_punt=pl.col("value_after").mean(), n_punt=pl.len())
v_fg = fourth.filter(pl.col("play_type") == "field_goal").group_by("zone").agg(V_fg=pl.col("value_after").mean(), n_fg=pl.len())

# P(convert) for the situations in each cell: logistic model on 4th-down attempts, averaged over ALL 4th downs in the cell
att = load("conv_plays").filter(pl.col("season").is_between(*ERA), pl.col("down") == 4)


def design(frame):
    y = frame["yardline_100"].to_numpy().astype(float)
    return np.column_stack([np.ones(frame.height), np.log(frame["ydstogo"].to_numpy().astype(float)),
                            (y <= 5).astype(float), (y <= 10).astype(float)])


m = sm.Logit(att["converted"].to_numpy(), design(att)).fit(disp=0)
fourth = fourth.with_columns(p_conv=pl.Series(m.predict(design(fourth))))
cell_p = fourth.group_by("bucket", "zone").agg(p_hat=pl.col("p_conv").mean(), n4=pl.len())

# Coaches' actual go rate in normal situations
dec = load("fourth_decisions").filter(pl.col("season").is_between(*ERA), pl.col("decision") != "fake_or_botched")
dec = dec.with_columns(zone.alias("zone"), bucket()).filter(not_eoh, normal)
go_rate = dec.group_by("bucket", "zone").agg(go=(pl.col("decision") == "go").mean(), n_dec=pl.len())
go_rate_recent = dec.filter(pl.col("season") >= 2023).group_by("bucket", "zone").agg(go_recent=(pl.col("decision") == "go").mean())

cells = (v_conv.join(v_fail, on="zone").join(v_punt, on="zone", how="left").join(v_fg, on="zone", how="left")
         .join(cell_p, on=["bucket", "zone"], how="left").join(go_rate, on=["bucket", "zone"], how="left")
         .join(go_rate_recent, on=["bucket", "zone"], how="left")
         .with_columns(V_punt=pl.when(pl.col("n_punt") >= 50).then(pl.col("V_punt")),
                       V_fg=pl.when(pl.col("n_fg") >= 50).then(pl.col("V_fg")))
         .with_columns(V_kick=pl.max_horizontal("V_punt", "V_fg"),
                       kick=pl.when(pl.col("V_fg").fill_null(-99) > pl.col("V_punt").fill_null(-99)).then(pl.lit("FG")).otherwise(pl.lit("Punt")))
         .with_columns(p_star=((pl.col("V_kick") - pl.col("V_fail")) / (pl.col("V_conv") - pl.col("V_fail"))).clip(0, 1))
         .with_columns(edge=pl.col("p_hat") - pl.col("p_star"))
         .sort("bucket", "zone"))
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
print(cells.select("bucket", "zone", pl.col(["V_conv", "V_fail", "V_punt", "V_fg"]).round(2), "kick",
                   pl.col(["p_star", "p_hat", "edge", "go", "go_recent"]).round(2), "n_dec"))

# Worked example for the maths corner: short yardage, own 30-39
for b, z in [(BUCKETS[0], 3), (BUCKETS[0], 6), (BUCKETS[1], 5)]:
    r = cells.filter(pl.col("bucket") == b, pl.col("zone") == z).row(0, named=True)
    print(f"\nMaths corner [{b}, {ZONES[z][2].replace(chr(10),' ')}]: V_conv {r['V_conv']:+.2f}, V_fail {r['V_fail']:+.2f}, "
          f"best kick ({r['kick']}) {r['V_kick']:+.2f}\n  p* = ({r['V_kick']:+.2f} - {r['V_fail']:+.2f}) / ({r['V_conv']:+.2f} - {r['V_fail']:+.2f}) "
          f"= {r['p_star']:.2f}; typical conversion {r['p_hat']:.2f} -> {'GO' if r['edge'] > 0 else 'KICK'}; coaches go {pct(r['go'])} "
          f"(2023-25: {pct(r['go_recent'])})")

# Missed opportunities: 4th downs (normal situations) where the cell says GO but the team kicked
dd = dec.join(cells.select("bucket", "zone", "edge"), on=["bucket", "zone"], how="left")
per_season = dd.group_by("season").agg(
    go_cells=(pl.col("edge") > 0.05).sum(),
    kicked_in_go_cells=((pl.col("edge") > 0.05) & (pl.col("decision") != "go")).sum()).sort("season")
print("\nClear-GO situations (edge > 5 pts) where teams kicked anyway, per season:")
print(per_season.with_columns(share_kicked=(pl.col("kicked_in_go_cells") / pl.col("go_cells")).round(2)))

# ---- chart: two stacked heatmaps sharing the field-position axis
blue_red = LinearSegmentedColormap.from_list("br", ["#e34948", "#f0efec", "#2a78d6"])
blues = LinearSegmentedColormap.from_list("bl", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
nz, nb = len(ZONES), len(BUCKETS)
E = np.full((nb, nz), np.nan); Pst = E.copy(); Ph = E.copy(); G = E.copy()
for r in cells.iter_rows(named=True):
    i = BUCKETS.index(r["bucket"])
    E[i, r["zone"]], Pst[i, r["zone"]], Ph[i, r["zone"]] = r["edge"], r["p_star"], r["p_hat"]
    G[i, r["zone"]] = r["go_recent"] if r["go_recent"] is not None else np.nan

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8.6), sharex=True)
for ax in (a1, a2):
    ax.grid(False); ax.tick_params(length=0)
a1.imshow(E, cmap=blue_red, norm=TwoSlopeNorm(0, -0.4, 0.4), aspect="auto")
for i in range(nb):
    for j in range(nz):
        if np.isnan(E[i, j]):
            continue
        dark = abs(E[i, j]) > 0.22
        a1.text(j, i - 0.12, f"need {pct(Pst[i, j])}", ha="center", va="center", fontsize=9.5,
                color="white" if dark else C["ink"], fontweight="bold")
        a1.text(j, i + 0.2, f"get {pct(Ph[i, j])}", ha="center", va="center", fontsize=9,
                color="white" if dark else C["ink2"])
a1.set_yticks(range(nb), BUCKETS)
a1.set_title("What the maths says: blue = go for it, red = kick", fontsize=13, loc="left", pad=8)
a2.imshow(G, cmap=blues, vmin=0, vmax=0.8, aspect="auto")
for i in range(nb):
    for j in range(nz):
        if not np.isnan(G[i, j]):
            a2.text(j, i, pct(G[i, j]), ha="center", va="center", fontsize=10.5,
                    color="white" if G[i, j] > 0.45 else C["ink"], fontweight="bold")
a2.set_yticks(range(nb), BUCKETS)
a2.set_xticks(range(nz), [z[2] for z in ZONES], fontsize=10)
a2.set_title("What coaches actually do: go-for-it rate, 2023–2025", fontsize=13, loc="left", pad=8)
a2.set_xlabel("Line of scrimmage  (own goal → opponent's goal)")
fig.subplots_adjust(top=0.81, bottom=0.11, left=0.12, right=0.98, hspace=0.30)
fig.text(0.02, 0.965, "Break-even: how often you need to convert for going for it to pay off", fontsize=18, fontweight="bold", va="top")
fig.text(0.02, 0.918, f"need = break-even conversion rate vs the best kick (punt or FG). get = typical conversion rate there. "
         f"{ERA[0]}–{ERA[1]}.\nNormal game situations; expected-points basis (score & clock ignored, see Q6).",
         fontsize=11.5, color=C["ink2"], va="top")
fig.text(0.02, 0.015, "Data: nflverse play-by-play & EP model", fontsize=9.5, color=C["muted"])
fig.savefig(FIG / "q5_break_even_heatmap.png")
plt.close(fig)
print("  -> figures/q5_break_even_heatmap.png")
