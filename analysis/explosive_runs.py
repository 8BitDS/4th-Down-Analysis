"""Does a team's explosive RUN rate raise its chance of converting on 3rd/4th down?

Theory: a team that breaks big runs forces defences to load the box / commit to the run,
which should make short-yardage conversions easier.

Explosive run = designed run (no scrambles, no kneels) gaining 10+ yards.
Team feature  = season-to-date rate from EARLIER games only, blended with last season early on
                (empirical-Bayes weight), then expressed relative to the league (z-score vs last
                season's spread of team rates, so 1999 and 2025 are on the same scale).
Controls      = season-to-date rushing EPA/play and success rate (is it explosiveness or just a good run game?).

Split: train 1999-2024 | validate 2025 | test 2026 (looked at once, at the end).
Data: 3rd + 4th downs (4th downs alone are too few to see team effects), with a 4th-down flag.
"""

import json

import statsmodels.api as sm

from common import *

p = load("plays")

# ---------------------------------------------------------------- per team-game rushing profile
runs = p.filter(pl.col("play_type") == "run", pl.col("qb_scramble").fill_null(0) == 0, pl.col("epa").is_not_null())


def team_games(side: str) -> pl.DataFrame:
    team = "posteam" if side == "off" else "defteam"
    return runs.group_by("game_id", "season", "game_date", pl.col(team).alias("team")).agg(
        n=pl.len(), xr=(pl.col("yards_gained") >= 10).sum(),
        epa=pl.col("epa").sum(), succ=(pl.col("epa") > 0).sum())


def season_to_date(g: pl.DataFrame, side: str) -> pl.DataFrame:
    g = g.sort("team", "game_date")
    prior = lambda c: pl.col(c).cum_sum().shift(1).over("team", "season").fill_null(0)  # noqa: E731
    g = g.with_columns(n_std=prior("n"), xr_std=prior("xr"), epa_std=prior("epa"), succ_std=prior("succ"))
    full = g.group_by("team", "season").agg(n_full=pl.col("n").sum(), xr_full=pl.col("xr").sum(),
                                             epa_full=pl.col("epa").sum(), succ_full=pl.col("succ").sum())
    prev = full.with_columns(pl.col("season") + 1).rename({c: c.replace("full", "prev") for c in full.columns if "full" in c})
    g = g.join(prev, on=["team", "season"], how="left")

    # League environment per season (for blending + z-scores): from the PREVIOUS season only
    lg = full.group_by("season").agg(
        lg_rate=(pl.col("xr_full").sum() / pl.col("n_full").sum()),
        lg_sd=(pl.col("xr_full") / pl.col("n_full")).std(),
        lg_epa=(pl.col("epa_full").sum() / pl.col("n_full").sum()),
        lg_epa_sd=(pl.col("epa_full") / pl.col("n_full")).std(),
        lg_succ=(pl.col("succ_full").sum() / pl.col("n_full").sum()),
        lg_succ_sd=(pl.col("succ_full") / pl.col("n_full")).std(),
        avg_n=pl.col("n_full").mean(),
    ).with_columns(pl.col("season") + 1)
    g = g.join(lg, on="season", how="left")
    return g, full


def add_features(g: pl.DataFrame, side: str, k: dict) -> pl.DataFrame:
    """Blend: (season-to-date events + k * prior mean) / (season-to-date attempts + k).
    Prior mean = last season's team rate shrunk halfway to the league (teams change between seasons)."""
    def blend(ev, n, prev_ev, prev_n, lg, kk):
        prior_mean = pl.when(prev_n > 0).then(0.5 * prev_ev / prev_n + 0.5 * lg).otherwise(lg)
        return (ev + kk * prior_mean) / (n + kk)

    return g.with_columns(
        xr_rate=blend(pl.col("xr_std"), pl.col("n_std"), pl.col("xr_prev"), pl.col("n_prev"), pl.col("lg_rate"), k["xr"]),
        rush_epa=blend(pl.col("epa_std"), pl.col("n_std"), pl.col("epa_prev"), pl.col("n_prev"), pl.col("lg_epa"), k["epa"]),
        rush_succ=blend(pl.col("succ_std"), pl.col("n_std"), pl.col("succ_prev"), pl.col("n_prev"), pl.col("lg_succ"), k["succ"]),
    ).with_columns(
        **{f"{side}_xr_z": (pl.col("xr_rate") - pl.col("lg_rate")) / pl.col("lg_sd"),
           f"{side}_rush_epa_z": (pl.col("rush_epa") - pl.col("lg_epa")) / pl.col("lg_epa_sd"),
           f"{side}_rush_succ_z": (pl.col("rush_succ") - pl.col("lg_succ")) / pl.col("lg_succ_sd"),
           f"{side}_xr_rate": pl.col("xr_rate")}
    ).select("game_id", "team", pl.col(f"^{side}_.*$"))


def shrink_k(full: pl.DataFrame, ev: str, is_rate: bool = True) -> float:
    """Empirical-Bayes prior weight k = (luck variance x n) / true-skill variance, from team-seasons 1999-2024."""
    f = full.filter(pl.col("season") <= 2024).with_columns(r=pl.col(ev) / pl.col("n_full"))
    var_obs = f["r"].var()
    if is_rate:
        pbar = f[ev].sum() / f["n_full"].sum()
        luck = (pbar * (1 - pbar) / f["n_full"]).mean()
        per_play_var = pbar * (1 - pbar)
    else:  # EPA: per-play variance from the raw plays
        per_play_var = runs.filter(pl.col("season") <= 2024)["epa"].var()
        luck = (per_play_var / f["n_full"]).mean()
    tau2 = max(var_obs - luck, 1e-9)
    return per_play_var / tau2, np.sqrt(tau2), np.sqrt(var_obs), float(np.sqrt(luck))


feats = {}
for side in ("off", "def"):
    g, full = season_to_date(team_games(side), side)
    k = {}
    for name, col, is_rate in [("xr", "xr_full", True), ("succ", "succ_full", True), ("epa", "epa_full", False)]:
        k[name], tau, obs, luck = shrink_k(full, col, is_rate)
        print(f"{side} {name:4s}: real team spread sd={tau:.4f} (observed {obs:.4f}, luck {luck:.4f}) -> prior weight k = {k[name]:.0f} rushes")
    if side == "off":
        yy = full.sort("team", "season").with_columns(r=pl.col("xr_full") / pl.col("n_full")).with_columns(
            nxt=pl.col("r").shift(-1).over("team")).drop_nulls("nxt").filter(pl.col("season") <= 2023)
        print(f"  explosive run rate, team year-to-year correlation r = {np.corrcoef(yy['r'], yy['nxt'])[0,1]:.2f}")
    feats[side] = add_features(g, side, k)

# ---------------------------------------------------------------- conversion data
c = load("conv_plays").filter(pl.col("season") >= 2000)  # 1999 has no prior season
c = (c.join(feats["off"].rename({"team": "posteam"}), on=["game_id", "posteam"], how="left")
      .join(feats["def"].rename({"team": "defteam"}), on=["game_id", "defteam"], how="left"))
c = c.with_columns(
    lyd=pl.col("ydstogo").cast(pl.Float64).log(), d4=(pl.col("down") == 4).cast(pl.Float64),
    gl5=(pl.col("yardline_100") <= 5).cast(pl.Float64), gl10=(pl.col("yardline_100") <= 10).cast(pl.Float64),
    short=(pl.col("ydstogo") <= 3).cast(pl.Float64),
    is_run=((pl.col("is_pass") == 0) & (pl.col("qb_scramble").fill_null(0) == 0)).cast(pl.Float64),
).drop_nulls(["off_xr_z", "def_xr_z", "off_rush_epa_z", "def_rush_epa_z"])
c = c.with_columns(
    off_xr_short=pl.col("off_xr_z") * pl.col("short"), def_xr_short=pl.col("def_xr_z") * pl.col("short"),
)
print(f"\nRows: {c.height:,} 3rd/4th-down plays (4th: {c.filter(pl.col('down')==4).height:,})")

BASE = ["lyd", "d4", "gl5", "gl10"]
CTRL = ["off_rush_epa_z", "off_rush_succ_z", "def_rush_epa_z", "def_rush_succ_z"]
XR = ["off_xr_z", "def_xr_z"]
XR_INT = XR + ["off_xr_short", "def_xr_short"]
MODELS_ = {
    "M0 distance + field position": BASE,
    "M1 + rushing efficiency (EPA, success)": BASE + CTRL,
    "M2 + explosive run rate": BASE + XR,
    "M3 + efficiency + explosive": BASE + CTRL + XR,
    "M4 + efficiency + explosive x short-yardage": BASE + CTRL + XR_INT,
}

train = c.filter(pl.col("season") <= 2024)
val = c.filter(pl.col("season") == 2025)
test = c.filter(pl.col("season") == 2026)


def fit(cols):
    return sm.Logit(train["converted"].to_numpy(), sm.add_constant(train.select(cols).to_numpy())).fit(disp=0)


def pr(m, frame, cols):
    return m.predict(sm.add_constant(frame.select(cols).to_numpy(), has_constant="add"))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_gain(frame, p0, p1, n_boot=2000):
    y = frame["converted"].to_numpy()
    d = ll(y, p0) - ll(y, p1)
    gid = frame["game_id"].to_numpy()
    _, inv = np.unique(gid, return_inverse=True)
    s, n = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return d.mean(), np.quantile(b, 0.05), np.quantile(b, 0.95)


fits = {k_: fit(v) for k_, v in MODELS_.items()}
print("\nCoefficients of interest (train 1999-2024), as odds multipliers per +1 SD vs league:")
for k_, m in fits.items():
    names = ["const"] + MODELS_[k_]
    show = [(n_, b_, se) for n_, b_, se in zip(names, m.params, m.bse) if n_ not in ["const"] + BASE]
    if show:
        print(f"  {k_}")
        for n_, b_, se in show:
            print(f"     {n_:18s} x{np.exp(b_):.3f}  (90% range x{np.exp(b_-1.645*se):.3f}-x{np.exp(b_+1.645*se):.3f})")

res = {}
for split_name, frame in [("VALIDATE 2025", val), ("TEST 2026", test)]:
    for sub_name, sel in [("3rd+4th downs", frame), ("4th downs only", frame.filter(pl.col("down") == 4)),
                          ("short yardage (1-3)", frame.filter(pl.col("ydstogo") <= 3))]:
        y = sel["converted"].to_numpy()
        p0 = pr(fits["M0 distance + field position"], sel, MODELS_["M0 distance + field position"])
        p1 = pr(fits["M1 + rushing efficiency (EPA, success)"], sel, MODELS_["M1 + rushing efficiency (EPA, success)"])
        print(f"\n{split_name} | {sub_name} (n={sel.height:,}): M0 log loss {ll(y,p0).mean():.4f}")
        for k_ in list(MODELS_)[1:]:
            pk = pr(fits[k_], sel, MODELS_[k_])
            g0, lo0, hi0 = boot_gain(sel, p0, pk)
            line = f"   {k_:46s} {ll(y,pk).mean():.4f} | vs M0 {g0*1000:+.2f} [{lo0*1000:+.2f},{hi0*1000:+.2f}]"
            if k_ in ("M3 + efficiency + explosive", "M4 + efficiency + explosive x short-yardage"):
                g1, lo1, hi1 = boot_gain(sel, p1, pk)
                line += f" | vs M1 {g1*1000:+.2f} [{lo1*1000:+.2f},{hi1*1000:+.2f}]"
                res[f"{split_name}|{sub_name}|{k_}"] = dict(gain_vs_M1=g1, lo=lo1, hi=hi1)
            print(line)
print("\n(gains are log-loss improvement x1e-3, 90% interval from resampling games; positive = better)")

# ---------------------------------------------------------------- mechanism checks (descriptive, train years)
print("\nMechanism check (2015-2024): 4th-and-1/2 conversion by offence's explosive-run tier and play type")
t = c.filter(pl.col("season").is_between(2015, 2024), pl.col("down") == 4, pl.col("ydstogo") <= 2).with_columns(
    tier=pl.when(pl.col("off_xr_z") >= 0.75).then(pl.lit("c) top (z>=+0.75)"))
    .when(pl.col("off_xr_z") <= -0.75).then(pl.lit("a) bottom (z<=-0.75)")).otherwise(pl.lit("b) middle")),
    kind=pl.when(pl.col("is_run") == 1).then(pl.lit("run")).otherwise(pl.lit("pass")))
tt = t.group_by("tier", "kind").agg(n=pl.len(), k=pl.col("converted").sum()).sort("tier", "kind")
lo_, hi_ = wilson(tt["k"].to_numpy(), tt["n"].to_numpy())
print(tt.with_columns(rate=(pl.col("k") / pl.col("n")).round(3), lo=pl.Series(lo_).round(3), hi=pl.Series(hi_).round(3)))

# ---------------------------------------------------------------- the two teams going into the game
gm = c.filter(pl.col("game_id") == "2026_07_GB_DET").select("posteam", "off_xr_rate", "off_xr_z", "off_rush_epa_z", "def_xr_z").unique("posteam")
print("\nGoing into GB @ DET (season-to-date, blended):")
print(gm.with_columns(pl.col(pl.Float64).round(3)))
(RES / "explosive_runs_metrics.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- effect on the two case plays
cp = c.filter(pl.col("game_id") == "2026_07_GB_DET", pl.col("down") == 4,
              ((pl.col("posteam") == "GB") & (pl.col("yardline_100") == 7)) | ((pl.col("posteam") == "DET") & (pl.col("yardline_100") == 76)))
print("\nCase plays, P(convert) by model:")
for r_i, r in enumerate(cp.iter_rows(named=True)):
    row = cp.slice(r_i, 1)
    vals = {k_[:2]: float(pr(fits[k_], row, MODELS_[k_])[0]) for k_ in MODELS_}
    print(f"  {r['posteam']} 4th-and-{r['ydstogo']} (yardline_100={r['yardline_100']}): " + " | ".join(f"{k_} {pct(v,1)}" for k_, v in vals.items()))
