"""Shared pieces for the win-probability models: data prep, features, LOSO loop, scoring."""

import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from common import *

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
TEST_SEASONS = list(range(2015, 2026))

FEATS_SPREAD = ["receive_2h_ko", "spread_time", "home", "half_seconds_remaining", "game_seconds_remaining",
                "Diff_Time_Ratio", "score_differential", "down", "ydstogo", "yardline_100",
                "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
MONO_SPREAD = [0, 0, 0, 0, 0, 1, 1, -1, -1, -1, 1, -1]
XGB_PARAMS = dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss", eta=0.05,
                  gamma=0.79012017, subsample=0.9224245, colsample_bytree=5 / 12, max_depth=5,
                  min_child_weight=7, tree_method="hist")
XGB_ROUNDS = 534


def load_wp_data() -> pl.DataFrame:
    """Labelled regulation plays (no ties) with nflverse features + candidate extra features."""
    w = load("wp_plays").filter(pl.col("qtr") <= 4, pl.col("posteam_win") != 0.5)
    elapsed = (3600 - pl.col("game_seconds_remaining")) / 3600
    decay = (-4 * elapsed).exp()

    # Season-to-date team strength for BOTH sides (offence EPA/play minus defence EPA/play allowed),
    # games before this one only. Week 1 falls back to last season's full-year value.
    ts = load("team_game_strength")
    net = ts.select(
        "game_id", "team",
        net=pl.coalesce(pl.col("off_epa_std"), pl.col("off_epa_prev_season"))
        - pl.coalesce(pl.col("def_epa_std"), pl.col("def_epa_prev_season")),
    )
    w = (w.join(net.rename({"team": "posteam", "net": "pos_net"}), on=["game_id", "posteam"], how="left")
         .join(net.rename({"team": "defteam", "net": "opp_net"}), on=["game_id", "defteam"], how="left"))

    return w.with_columns(
        receive_2h_ko=pl.when(pl.col("qtr") <= 2).then(pl.col("posteam_receives_2h_ko")).otherwise(0),
        home=(pl.col("posteam_type") == "home").cast(pl.Int8),
        spread_time=pl.col("posteam_spread").fill_null(pl.when(pl.col("posteam_type") == "home").then(1.5).otherwise(-1.5)) * decay,
        Diff_Time_Ratio=pl.col("score_differential") / decay,
        strength_time=(pl.col("pos_net") - pl.col("opp_net")) * decay,
        total_line=pl.col("total_line"),
        elapsed_share=elapsed,
        y=pl.col("posteam_win").cast(pl.Int8),
    ).drop_nulls(["score_differential", "yardline_100", "down", "posteam_timeouts_remaining", "defteam_timeouts_remaining"])


# --------------------------------------------------------------------------- scoring


def cal_error(frame: pl.DataFrame, col: str) -> float:
    t = (frame.with_columns(b=(pl.col(col) / 0.05).floor())
         .group_by("qtr", "b").agg(p=pl.col(col).mean(), a=pl.col("y").mean(), n=pl.len()))
    return float((t["n"] * (t["p"] - t["a"]).abs()).sum() / t["n"].sum())


def score(frame: pl.DataFrame, col: str) -> dict:
    f = frame.drop_nulls(col)
    p = f[col].clip(1e-6, 1 - 1e-6).to_numpy()
    y = f["y"].to_numpy()
    return dict(n=f.height, logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                acc=float(((p > 0.5) == y).mean()), cal=cal_error(f, col))


def paired_gain(frame: pl.DataFrame, base: str, new: str, n_boot: int = 1000, seed: int = 0):
    """Log-loss improvement of `new` over `base` on the same plays, with a 90% interval from resampling
    whole GAMES (plays in one game aren't independent). Positive = new is better."""
    f = frame.drop_nulls([base, new]).with_columns(
        d=(-(pl.col("y") * pl.col(base).clip(1e-6, 1 - 1e-6).log() + (1 - pl.col("y")) * (1 - pl.col(base)).clip(1e-6, 1).log()))
        - (-(pl.col("y") * pl.col(new).clip(1e-6, 1 - 1e-6).log() + (1 - pl.col("y")) * (1 - pl.col(new)).clip(1e-6, 1).log())))
    g = f.group_by("game_id").agg(s=pl.col("d").sum(), n=pl.len())
    s, n = g["s"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


# --------------------------------------------------------------------------- XGBoost LOSO


def xgb_loso(w: pl.DataFrame, feats: list[str], mono: list[int], name: str, save: bool = False) -> pl.DataFrame:
    """Leave-one-season-out predictions for 2015-2025 + 2026 from a model trained on 1999-2025."""
    params = {**XGB_PARAMS, "monotone_constraints": "(" + ",".join(map(str, mono)) + ")", "seed": 0}

    def fit(frame):
        d = xgb.DMatrix(frame.select(feats).to_numpy(), label=frame["y"].to_numpy(), feature_names=feats)
        return xgb.train(params, d, num_boost_round=XGB_ROUNDS)

    def pred(model, frame):
        return model.predict(xgb.DMatrix(frame.select(feats).to_numpy(), feature_names=feats))

    parts = []
    for s in TEST_SEASONS:
        te = w.filter(pl.col("season") == s)
        parts.append(te.select("game_id", "play_id").with_columns(pl.Series(name, pred(fit(w.filter(pl.col("season") != s, pl.col("season") <= 2025)), te))))
    final = fit(w.filter(pl.col("season") <= 2025))
    te = w.filter(pl.col("season") == 2026)
    parts.append(te.select("game_id", "play_id").with_columns(pl.Series(name, pred(final, te))))
    if save:
        final.save_model(MODELS / f"{name}.json")
    return pl.concat(parts)


SUBSETS = {
    "All plays 2015-25": lambda f: f.filter(pl.col("season") <= 2025),
    "4th downs 2015-25": lambda f: f.filter(pl.col("season") <= 2025, pl.col("down") == 4),
    "Q4 close (<=8 pts)": lambda f: f.filter(pl.col("season") <= 2025, pl.col("qtr") == 4, pl.col("score_differential").abs() <= 8),
    "2026 holdout": lambda f: f.filter(pl.col("season") == 2026),
}
