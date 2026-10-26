"""Replicate nflverse's win-probability models (no-spread `wp` and spread `vegas_wp`) with XGBoost.

Same features, same hyperparameters, same monotone constraints as nflfastR, same validation idea:
leave-one-season-out (LOSO). For every test season 2015-2025 we train on all OTHER seasons 1999-2025,
predict the held-out season, then score the pooled out-of-sample predictions.
2026 is scored with a model trained on 1999-2025 (fully out of sample, like real use).

Source of settings: nflfastR R/helper_add_ep_wp.R and the 2020 Open Source Football model write-up.
"""

import json
import time

import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from common import *

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ---- data: labelled plays, regulation only, no ties (as nflverse)
w = load("wp_plays").filter(pl.col("qtr") <= 4, pl.col("posteam_win") != 0.5)
elapsed = (3600 - pl.col("game_seconds_remaining")) / 3600
w = w.with_columns(
    receive_2h_ko=pl.when(pl.col("qtr") <= 2).then(pl.col("posteam_receives_2h_ko")).otherwise(0),
    home=(pl.col("posteam_type") == "home").cast(pl.Int8),
    spread_time=pl.col("posteam_spread").fill_null(pl.when(pl.col("posteam_type") == "home").then(1.5).otherwise(-1.5))
    * (-4 * elapsed).exp(),
    Diff_Time_Ratio=pl.col("score_differential") / (-4 * elapsed).exp(),
    y=pl.col("posteam_win").cast(pl.Int8),
).drop_nulls(["score_differential", "yardline_100", "down", "posteam_timeouts_remaining", "defteam_timeouts_remaining"])

FEATS_SPREAD = ["receive_2h_ko", "spread_time", "home", "half_seconds_remaining", "game_seconds_remaining",
                "Diff_Time_Ratio", "score_differential", "down", "ydstogo", "yardline_100",
                "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
FEATS_BASE = [f for f in FEATS_SPREAD if f != "spread_time"]

PARAMS = {
    "wp": dict(params=dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss", eta=0.2, gamma=0,
                           subsample=0.8, colsample_bytree=0.8, max_depth=4, min_child_weight=1, tree_method="hist"),
               rounds=65, feats=FEATS_BASE, nflverse_col="wp"),
    "vegas_wp": dict(params=dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss", eta=0.05,
                                 gamma=0.79012017, subsample=0.9224245, colsample_bytree=5 / 12, max_depth=5,
                                 min_child_weight=7, tree_method="hist",
                                 monotone_constraints="(0,0,0,0,0,1,1,-1,-1,-1,1,-1)"),
                     rounds=534, feats=FEATS_SPREAD, nflverse_col="vegas_wp"),
}
print(f"Training rows 1999-2025: {w.filter(pl.col('season') <= 2025).height:,}")


def fit(frame, cfg, seed=0):
    d = xgb.DMatrix(frame.select(cfg["feats"]).to_numpy(), label=frame["y"].to_numpy(), feature_names=cfg["feats"])
    return xgb.train({**cfg["params"], "seed": seed}, d, num_boost_round=cfg["rounds"])


def predict(model, frame, cfg):
    return model.predict(xgb.DMatrix(frame.select(cfg["feats"]).to_numpy(), feature_names=cfg["feats"]))


def cal_error(frame, col):
    t = (frame.with_columns(b=(pl.col(col) / 0.05).floor())
         .group_by("qtr", "b").agg(p=pl.col(col).mean(), a=pl.col("y").mean(), n=pl.len()))
    return float((t["n"] * (t["p"] - t["a"]).abs()).sum() / t["n"].sum())


def score(frame, col):
    f = frame.drop_nulls(col)
    p = f[col].clip(1e-6, 1 - 1e-6).to_numpy()
    y = f["y"].to_numpy()
    return dict(n=f.height, logloss=log_loss(y, p), brier=brier_score_loss(y, p), acc=float(((p > 0.5) == y).mean()),
                cal=cal_error(f, col))


results = {}
oof_all = []
for name, cfg in PARAMS.items():
    t0 = time.time()
    parts = []
    for s in range(2015, 2026):  # leave-one-season-out
        tr = w.filter(pl.col("season") != s, pl.col("season") <= 2025)
        te = w.filter(pl.col("season") == s)
        parts.append(te.select("game_id", "play_id").with_columns(pl.Series(f"ours_{name}", predict(fit(tr, cfg), te, cfg))))
    final = fit(w.filter(pl.col("season") <= 2025), cfg)
    final.save_model(MODELS / f"wp_replica_{name}.json")
    te26 = w.filter(pl.col("season") == 2026)
    parts.append(te26.select("game_id", "play_id").with_columns(pl.Series(f"ours_{name}", predict(final, te26, cfg))))
    oof_all.append(pl.concat(parts))
    print(f"{name}: 11 LOSO fits + final fit in {time.time()-t0:.0f}s")

ev = w.filter(pl.col("season") >= 2015)
for o in oof_all:
    ev = ev.join(o, on=["game_id", "play_id"], how="left")
ev.select("game_id", "play_id", "season", "qtr", "down", "y", "wp", "vegas_wp", "ours_wp", "ours_vegas_wp").write_parquet(
    MODELS / "wp_replica_oof.parquet")

subsets = {
    "All plays 2015-2025": ev.filter(pl.col("season") <= 2025),
    "4th downs 2015-2025": ev.filter(pl.col("season") <= 2025, pl.col("down") == 4),
    "Q4 last 5 min 2015-25": ev.filter(pl.col("season") <= 2025, pl.col("qtr") == 4, pl.col("game_seconds_remaining") <= 300),
    "2026 (true holdout)": ev.filter(pl.col("season") == 2026),
}
print("\n                          | nflverse                         | our replica (out-of-sample)")
print("                          | logloss  Brier  acc    cal       | logloss  Brier  acc    cal")
for name in PARAMS:
    print(f"--- {name} ---")
    for label, f in subsets.items():
        a, b = score(f, name), score(f, f"ours_{name}")
        results[f"{name}|{label}"] = dict(nflverse=a, ours=b)
        print(f"  {label:23s} | {a['logloss']:.3f}   {a['brier']:.3f}  {a['acc']:.1%}  {a['cal']:.4f}    "
              f"| {b['logloss']:.3f}   {b['brier']:.3f}  {b['acc']:.1%}  {b['cal']:.4f}")

# Agreement between the two on the same plays
for name in PARAMS:
    f = ev.drop_nulls([name, f"ours_{name}"])
    diff = (f[name] - f[f"ours_{name}"]).abs()
    print(f"\n{name}: ours vs nflverse on the same plays - correlation {np.corrcoef(f[name], f[f'ours_{name}'])[0,1]:.4f}, "
          f"median gap {diff.median()*100:.1f} pts, 95% of plays within {diff.quantile(0.95)*100:.1f} pts")

(RES / "wp_replica_metrics.json").write_text(json.dumps(results, indent=2))

# ---- the two case-study plays, before the snap
cases = w.filter(pl.col("game_id") == "2026_07_GB_DET", pl.col("down") == 4,
                 ((pl.col("posteam") == "GB") & (pl.col("yardline_100") == 7)) | ((pl.col("posteam") == "DET") & (pl.col("yardline_100") == 76)))
print("\nCase plays, WP before the snap:")
for name, cfg in PARAMS.items():
    mdl = xgb.Booster(); mdl.load_model(MODELS / f"wp_replica_{name}.json")
    p = predict(mdl, cases, cfg)
    for r, pp in zip(cases.iter_rows(named=True), p):
        print(f"  {name:9s} {r['posteam']} 4th-and-{r['ydstogo']} at yardline_100={r['yardline_100']}: nflverse {pct(r[name],1)} | ours {pct(float(pp),1)}")

# ---- chart: calibration by quarter, ours (vegas) out-of-sample vs perfect
f = ev.filter(pl.col("season") <= 2025).drop_nulls("ours_vegas_wp")
fig, axes = plt.subplots(1, 4, figsize=(13, 4.8), sharey=True)
for q, ax in zip((1, 2, 3, 4), axes):
    ax.tick_params(length=0)
    t = (f.filter(pl.col("qtr") == q).with_columns(b=(pl.col("ours_vegas_wp") / 0.05).floor())
         .group_by("b").agg(p=pl.col("ours_vegas_wp").mean(), a=pl.col("y").mean(), n=pl.len()).sort("b"))
    t2 = (f.filter(pl.col("qtr") == q).with_columns(b=(pl.col("vegas_wp") / 0.05).floor())
          .group_by("b").agg(p=pl.col("vegas_wp").mean(), a=pl.col("y").mean(), n=pl.len()).sort("b"))
    ax.plot([0, 1], [0, 1], color=C["axis"], lw=1)
    ax.plot(t2["p"], t2["a"], color=C["orange"], lw=2, label="nflverse vegas_wp")
    ax.plot(t["p"], t["a"], color=C["blue"], lw=2, label="Our replica (out-of-sample)")
    ax.scatter(t["p"], t["a"], s=18, color=C["blue"], edgecolor=C["surface"], lw=1, zorder=3)
    ax.set_title(f"Q{q}", fontsize=13, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0.25, 0.5, 0.75]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
axes[0].set_ylabel("Actual win rate")
fig.supxlabel("Predicted win probability", fontsize=11, color=C["ink2"], y=0.06)
axes[0].legend(loc="upper left", fontsize=9.5)
fig.subplots_adjust(top=0.74, bottom=0.2, left=0.07, right=0.98, wspace=0.12)
s_ours = results["vegas_wp|All plays 2015-2025"]["ours"]
fig.text(0.02, 0.96, "Our win-probability model matches nflverse's", fontsize=18, fontweight="bold", va="top")
fig.text(0.02, 0.88, f"Calibration by quarter, 2015–2025, each season predicted by a model that never saw it. On the diagonal = "
         f"'says 30%, wins 30%'.\nLog loss {s_ours['logloss']:.3f}, calibration error {s_ours['cal']:.4f}.",
         fontsize=11, color=C["ink2"], va="top")
fig.text(0.02, 0.015, "Data: nflverse play-by-play. Model: XGBoost with nflfastR's features & settings.", fontsize=9, color=C["muted"])
fig.savefig(FIG / "wp_replica_calibration.png")
plt.close(fig)
print("  -> figures/wp_replica_calibration.png")
