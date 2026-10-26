"""The decision: GO vs KICK for both case plays, in win probability.

WP(GO)   = P(convert) x WP(after a conversion) + (1 - P(convert)) x WP(after a failure)
WP(PUNT) = WP(after a punt)
WP(FG)   = P(make) x WP(after a make) + (1 - P(make)) x WP(after a miss)

"WP after X" = average over REAL outcomes of similar plays 2015-2025 (yards gained, clock used, scores,
returns, penalties...), transplanted into this game's score / clock / timeouts / spread and scored by our
XGBoost WP model (the nflverse replica).
P(convert) = locked conversion model (distance + field position + rushing efficiency, logistic).

Uncertainty: bootstrap B times. Each round resamples GAMES to refit the WP model and the conversion
model, and resamples the outcome pools. Usage: decision.py [B]   (B=0 -> point estimates only, B=-1 -> redraw charts with saved ranges)
"""

import contextlib
import io
import json
import runpy
import sys
import time

import statsmodels.api as sm

from wp_common import *

B = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- the two case plays
GAME = "2026_07_GB_DET"
plays = load("plays")
g = plays.filter(pl.col("game_id") == GAME)
CASES = {
    "DET": dict(row=g.filter(pl.col("posteam") == "DET", pl.col("down") == 4, pl.col("yardline_100") == 76),
                kick="PUNT", title="Detroit: 4th-and-1 at own 24, tied 17–17, 4:53 left"),
    "GB": dict(row=g.filter(pl.col("posteam") == "GB", pl.col("down") == 4, pl.col("yardline_100") == 7),
               kick="FG", title="Green Bay: 4th-and-4 at the DET 7, down 7–10, 4:20 left in Q2"),
}
for k_, cs in CASES.items():
    assert cs["row"].height == 1, k_
    cs["s"] = cs["row"].row(0, named=True)

# ---------------------------------------------------------------- outcome pools (real plays, next-snap states)
hist = plays.filter(pl.col("season").is_between(2015, 2025), pl.col("qtr") <= 4)
snaps = (hist.filter(pl.col("down").is_not_null(), pl.col("posteam").is_not_null()).sort("game_id", "play_id")
         .with_columns(
             n_pos=pl.col("posteam").shift(-1).over("game_id"), n_yl=pl.col("yardline_100").shift(-1).over("game_id"),
             n_down=pl.col("down").shift(-1).over("game_id"), n_togo=pl.col("ydstogo").shift(-1).over("game_id"),
             n_sd=pl.col("score_differential").shift(-1).over("game_id"),
             n_gsr=pl.col("game_seconds_remaining").shift(-1).over("game_id"),
             n_half=pl.col("game_half").shift(-1).over("game_id"))
         .filter(pl.col("n_half") == pl.col("game_half"), pl.col("half_seconds_remaining") > 120)
         .with_columns(same=(pl.col("n_pos") == pl.col("posteam")),
                       dt=pl.col("game_seconds_remaining") - pl.col("n_gsr"))
         .with_columns(dscore=pl.when(pl.col("same")).then(pl.col("n_sd")).otherwise(-pl.col("n_sd")) - pl.col("score_differential")))
fake = pl.col("desc").str.contains(r"(?i)\((punt|field goal) formation\)")
go = pl.col("play_type").is_in(["run", "pass"]) & ~fake & (pl.col("down") == 4)

POOLS = {
    "DET": {
        "convert": snaps.filter(go, pl.col("ydstogo") == 1, pl.col("yardline_100").is_between(60, 90), pl.col("fourth_down_converted") == 1),
        "fail": snaps.filter(go, pl.col("ydstogo") == 1, pl.col("yardline_100").is_between(60, 90), pl.col("fourth_down_failed") == 1),
        "punt": snaps.filter(pl.col("play_type") == "punt", pl.col("yardline_100").is_between(72, 80)),
    },
    "GB": {
        "convert": snaps.filter(go, pl.col("ydstogo").is_between(3, 5), pl.col("yardline_100").is_between(4, 10), pl.col("fourth_down_converted") == 1),
        "fail": snaps.filter(go, pl.col("ydstogo").is_between(3, 5), pl.col("yardline_100").is_between(4, 10), pl.col("fourth_down_failed") == 1),
        "make": snaps.filter(pl.col("play_type") == "field_goal", pl.col("yardline_100").is_between(4, 10), pl.col("field_goal_result") == "made"),
        "miss": snaps.filter(pl.col("play_type") == "field_goal", pl.col("yardline_100").is_between(4, 10), pl.col("field_goal_result") != "made"),
    },
}
for k_, pools in POOLS.items():
    print(k_, {b: f"{p.height} plays" for b, p in pools.items()})
gbc = POOLS["GB"]["convert"]
print(f"  GB pool: {gbc.filter(pl.col('dscore') > 0).height / gbc.height:.0%} of 4th-and-3-5 conversions from the 4-10 scored a TD on that drive's next snap")

# ---------------------------------------------------------------- state -> WP features
def states_from_pool(pool: pl.DataFrame, s: dict) -> pl.DataFrame:
    """Transplant each pool outcome into the case game. Returns rows from the CASE team's point of view."""
    me, opp = s["posteam"], s["defteam"]
    gsr = s["game_seconds_remaining"] - pool["dt"].to_numpy()
    keep = gsr > 0
    pool = pool.filter(pl.Series(keep))
    gsr = gsr[keep]
    same = pool["same"].to_numpy()
    sd_me = s["score_differential"] + pool["dscore"].to_numpy()
    half_end = 1800 if s["qtr"] <= 2 else 0
    me_home = s["posteam_type"] == "home"
    me_spread = s["posteam_spread"]
    me_rec2h = s["posteam_receives_2h_ko"] if s["qtr"] <= 2 else 0
    pos_home = np.where(same, me_home, not me_home).astype(int)
    # Field position: transplant the CHANGE in position, not the historical spot.
    # Measure the next snap in the pool team's frame (distance to the end zone it attacks), shift it by
    # (case spot - pool spot), then convert back. After a score the next snap follows a kickoff, so keep it as is.
    y0 = pool["yardline_100"].to_numpy().astype(float)
    y1 = pool["n_yl"].to_numpy().astype(float)
    pos_me = np.where(same, y1, 100 - y1) + (s["yardline_100"] - y0)
    pos_me = np.clip(pos_me, 1, 99)
    yl = np.where(pool["dscore"].to_numpy() != 0, y1, np.where(same, pos_me, 100 - pos_me))
    togo = np.minimum(pool["n_togo"].to_numpy().astype(float), yl)  # 1st-and-10 at the 4 is 1st-and-goal
    df = pl.DataFrame({
        "same": same,
        "receive_2h_ko": np.where(same, me_rec2h, (1 - me_rec2h) if s["qtr"] <= 2 else 0),
        "home": pos_home,
        "posteam_spread": np.where(same, me_spread, -me_spread),
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": gsr - half_end,
        "score_differential": np.where(same, sd_me, -sd_me),
        "down": pool["n_down"].to_numpy(), "ydstogo": togo, "yardline_100": yl,
        "posteam_timeouts_remaining": np.where(same, s["posteam_timeouts_remaining"], s["defteam_timeouts_remaining"]),
        "defteam_timeouts_remaining": np.where(same, s["defteam_timeouts_remaining"], s["posteam_timeouts_remaining"]),
    })
    el = (3600 - pl.col("game_seconds_remaining")) / 3600
    return df.with_columns(spread_time=pl.col("posteam_spread") * (-4 * el).exp(),
                           Diff_Time_Ratio=pl.col("score_differential") / (-4 * el).exp())


def wp_me(model, st: pl.DataFrame) -> np.ndarray:
    p = model.predict(xgb.DMatrix(st.select(FEATS_SPREAD).to_numpy().astype(float), feature_names=FEATS_SPREAD))
    return np.where(st["same"].to_numpy(), p, 1 - p)


STATES = {k_: {b: states_from_pool(p, CASES[k_]["s"]) for b, p in pools.items()} for k_, pools in POOLS.items()}

# ---------------------------------------------------------------- P(convert): locked conversion model (M1)
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(ROOT / "analysis" / "explosive_runs.py"))
conv, COLS = ns["c"], ns["MODELS_"]["M1 + rushing efficiency (EPA, success)"]
conv_train = conv.filter(pl.col("season").is_between(2000, 2025))
case_conv = {k_: conv.filter(pl.col("game_id") == GAME, pl.col("down") == 4, pl.col("posteam") == k_,
                             pl.col("yardline_100") == CASES[k_]["s"]["yardline_100"]) for k_ in CASES}


def p_convert(train: pl.DataFrame) -> dict:
    m = sm.Logit(train["converted"].to_numpy(), sm.add_constant(train.select(COLS).to_numpy())).fit(disp=0)
    return {k_: float(m.predict(sm.add_constant(r.select(COLS).to_numpy(), has_constant="add"))[0]) for k_, r in case_conv.items()}


# ---------------------------------------------------------------- P(make) for Smack from the 7 (~25 yds)
fg = load("fg_attempts").filter(pl.col("season").is_between(2015, 2025))
mfg = sm.Logit(fg["made"].to_numpy(), sm.add_constant(fg["kick_distance"].to_numpy().astype(float))).fit(disp=0)
lg = lambda d: float(mfg.predict(np.array([[1.0, d]]))[0])  # noqa: E731
dist = CASES["GB"]["s"]["yardline_100"] + 18
sm_k = load("fg_attempts").filter(pl.col("kicker_player_name") == "T.Smack", pl.col("game_id") != GAME)
exp_k = float(sum(lg(d) for d in sm_k["kick_distance"]))
made_k = int(sm_k["made"].sum())
shrunk_shift = (made_k - exp_k) / (sm_k.height + 76)  # Q3 shrinkage: 76 imaginary league kicks
logit = lambda p: np.log(p / (1 - p))  # noqa: E731
p_make_league = lg(dist)
p_make = 1 / (1 + np.exp(-(logit(p_make_league) + logit(np.clip(0.83 + shrunk_shift, .01, .99)) - logit(0.83))))
print(f"\nGB FG from the 7 = {dist} yds: league {pct(p_make_league,1)}; Smack before this game {made_k}/{sm_k.height} "
      f"(expected {exp_k:.1f}) -> shrunk {shrunk_shift*100:+.1f} pts -> P(make) {pct(p_make,1)}")


# ---------------------------------------------------------------- evaluate
def evaluate(model, pc: dict, pools_idx=None) -> dict:
    out = {}
    for k_, st in STATES.items():
        w_ = {}
        for b, df in st.items():
            if pools_idx is not None:
                df = df[pools_idx[k_][b]]
            w_[b] = float(wp_me(model, df).mean())
        p = pc[k_]
        wp_go = p * w_["convert"] + (1 - p) * w_["fail"]
        wp_kick = w_["punt"] if k_ == "DET" else p_make * w_["make"] + (1 - p_make) * w_["miss"]
        p_star = (wp_kick - w_["fail"]) / (w_["convert"] - w_["fail"])
        out[k_] = dict(p_convert=p, **{f"wp_{b}": v for b, v in w_.items()}, wp_go=wp_go, wp_kick=wp_kick,
                       gap=wp_go - wp_kick, break_even=p_star)
    return out


wp_model = xgb.Booster()
wp_model.load_model(MODELS / "wp_replica_vegas_wp.json")
pc0 = p_convert(conv_train)
point = evaluate(wp_model, pc0)

# Before-snap WP (sanity vs nflverse)
for k_, cs in CASES.items():
    s = cs["s"]
    st = states_from_pool(pl.DataFrame({"dt": [0.0], "same": [True], "dscore": [0.0], "n_down": [4.0],
                                        "n_togo": [float(s["ydstogo"])], "n_yl": [float(s["yardline_100"])], "yardline_100": [float(s["yardline_100"])]}), s)
    point[k_]["wp_before"] = float(wp_me(wp_model, st)[0])
    point[k_]["nflverse_vegas_wp"] = s["vegas_wp"]

# ---------------------------------------------------------------- bootstrap
boots = {k_: [] for k_ in CASES}
if B > 0:
    wd = load_wp_data().filter(pl.col("season") <= 2025)
    games = wd["game_id"].unique().to_numpy()
    cgames = conv_train["game_id"].unique().to_numpy()
    params = {**XGB_PARAMS, "monotone_constraints": "(" + ",".join(map(str, MONO_SPREAD)) + ")"}
    t0 = time.time()
    for i in range(B):
        gs = pl.DataFrame({"game_id": rng.choice(games, len(games))})
        bw = wd.join(gs, on="game_id")  # duplicates games drawn twice
        d = xgb.DMatrix(bw.select(FEATS_SPREAD).to_numpy(), label=bw["y"].to_numpy(), feature_names=FEATS_SPREAD)
        mdl = xgb.train({**params, "seed": i}, d, num_boost_round=XGB_ROUNDS)
        cg = pl.DataFrame({"game_id": rng.choice(cgames, len(cgames))})
        pc = p_convert(conv_train.join(cg, on="game_id"))
        idx = {k_: {b: rng.integers(0, df.height, df.height) for b, df in st.items()} for k_, st in STATES.items()}
        r = evaluate(mdl, pc, idx)
        for k_ in CASES:
            boots[k_].append(r[k_])
        if i % 10 == 9:
            print(f"  bootstrap {i+1}/{B} ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- report
summary = {}
for k_, cs in CASES.items():
    r = point[k_]
    kick = cs["kick"]
    print(f"\n=== {cs['title']} ===")
    print(f"  WP before the snap: ours {pct(r['wp_before'],1)} | nflverse vegas_wp {pct(r['nflverse_vegas_wp'],1)}")
    print(f"  P(convert) {pct(r['p_convert'],1)}  ->  WP if convert {pct(r['wp_convert'],1)} | WP if fail {pct(r['wp_fail'],1)}")
    if k_ == "DET":
        print(f"  WP after a punt {pct(r['wp_kick'],1)}")
    else:
        print(f"  P(make) {pct(p_make,1)} -> WP if make {pct(r['wp_make'],1)} | WP if miss {pct(r['wp_miss'],1)} -> WP(FG) {pct(r['wp_kick'],1)}")
    print(f"  WP(GO) {pct(r['wp_go'],1)} vs WP({kick}) {pct(r['wp_kick'],1)}  ->  gap {r['gap']*100:+.1f} WP pts")
    print(f"  Break-even conversion rate: {pct(r['break_even'],1)} (model gives {pct(r['p_convert'],1)})")
    summary[k_] = dict(point=r)
    if B > 0:
        gaps = np.array([b["gap"] for b in boots[k_]])
        bes = np.array([b["break_even"] for b in boots[k_]])
        pcs = np.array([b["p_convert"] for b in boots[k_]])
        lo, hi = np.quantile(gaps, [0.05, 0.95])
        share = (gaps > 0).mean()
        print(f"  90% range for the gap: {lo*100:+.1f} to {hi*100:+.1f} pts; GO better in {share:.0%} of {B} bootstrap runs")
        print(f"  90% range: break-even {pct(np.quantile(bes,.05),1)}-{pct(np.quantile(bes,.95),1)}, "
              f"P(convert) {pct(np.quantile(pcs,.05),1)}-{pct(np.quantile(pcs,.95),1)}")
        summary[k_].update(gap_lo=lo, gap_hi=hi, go_better_share=share,
                           be_lo=float(np.quantile(bes, .05)), be_hi=float(np.quantile(bes, .95)),
                           pc_lo=float(np.quantile(pcs, .05)), pc_hi=float(np.quantile(pcs, .95)))
summary["GB"]["p_make"] = p_make
if B == -1:  # redraw charts only, using the ranges saved by the last bootstrap run
    saved = json.loads((RES / "decision_summary.json").read_text())
    for k_ in CASES:
        summary[k_].update({kk: vv for kk, vv in saved[k_].items() if kk != "point"})
    B = 100
else:
    (RES / "decision_summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------------------------------------------------------- chart: WP(GO) as a line in P(convert) vs flat kick
for k_, cs in CASES.items():
    r = summary[k_]["point"]
    fig, ax = new_fig(10, 6.2)
    xs = np.linspace(0, 1, 101)
    ax.plot(xs, xs * r["wp_convert"] + (1 - xs) * r["wp_fail"], color=C["blue"], lw=2.5, label="Go for it")
    ax.axhline(r["wp_kick"], color=C["orange"], lw=2.5, label="Punt" if k_ == "DET" else "Field goal")
    be, pc = r["break_even"], r["p_convert"]
    if B:
        ax.axvspan(summary[k_]["pc_lo"], summary[k_]["pc_hi"], color=C["blue"], alpha=0.08, lw=0)
    ax.axvline(be, color=C["ink2"], lw=1, ls=(0, (3, 3)))
    ax.text(be - 0.012, r["wp_fail"] + 0.005, f"Break-even\n{pct(be)}", ha="right", va="bottom", fontsize=11, color=C["ink2"])
    ax.scatter([pc], [r["wp_go"]], s=90, color=C["blue"], edgecolor=C["surface"], lw=2, zorder=4)
    ax.annotate(f"Model: {pct(pc)} to convert\n→ {pct(r['wp_go'],1)} win chance",
                xy=(pc, r["wp_go"]), xytext=(pc + 0.025, r["wp_go"] - 0.012), ha="left", va="top",
                fontsize=11, color=C["ink"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Chance of converting the 4th down")
    ax.set_ylabel(f"{k_} win probability")
    ax.legend(loc="upper left", fontsize=11)
    verdict = "GO" if r["gap"] > 0 else ("PUNT" if k_ == "DET" else "KICK")
    rng_txt = (f": plausibly {summary[k_]['gap_lo']*100:+.1f} to {summary[k_]['gap_hi']*100:+.1f} pts "
               f"(go better in {summary[k_]['go_better_share']:.0%} of 100 re-runs)" if B else "")
    kick_word = "punting" if k_ == "DET" else "the field goal"
    if B and summary[k_]["gap_lo"] < 0 < summary[k_]["gap_hi"]:
        headline = f"{cs['title'].split(':')[0]}: a coin flip. Going for it and {kick_word} are within {abs(r['gap'])*100:.1f} pts"
    else:
        headline = f"{cs['title'].split(':')[0]}: the maths says {verdict}, by {abs(r['gap'])*100:.1f} win-prob. pts"
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    finish(fig, ax, f"decision_{k_}",
           headline,
           f"{cs['title'].split(': ')[1]}. Gap{rng_txt}.\nShaded = range for the conversion chance. "
           "Lines cross at break-even: above it, going for it wins more games.",
           source="Data: nflverse play-by-play. Models: our XGBoost WP model + conversion model; outcomes from similar plays 2015–2025.")
