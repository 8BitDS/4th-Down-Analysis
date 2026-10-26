"""Shared helpers for the background-question analyses: loading, buckets, maths, chart style."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# "Modern era" window for background stats: complete seasons only.
ERA = (2015, 2025)
CUR = 2026

BUCKETS = ["Short (1-3)", "Medium (4-9)", "Long (10+)"]


def bucket(col: str = "ydstogo") -> pl.Expr:
    return (
        pl.when(pl.col(col) <= 3).then(pl.lit(BUCKETS[0]))
        .when(pl.col(col) <= 9).then(pl.lit(BUCKETS[1]))
        .otherwise(pl.lit(BUCKETS[2]))
        .alias("bucket")
    )


def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(CLEAN / f"{name}.parquet")


# --------------------------------------------------------------------------- maths


def wilson(k, n, z: float = 1.645):
    """Wilson interval for a proportion (90% by default). Works on arrays."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def value_after(plays: pl.DataFrame) -> pl.DataFrame:
    """Expected points AFTER each snap, from the offence's point of view.

    value_after = points the offence gained on the scoreboard before the next snap
                + expected points of the next snap (flipped if the other team has the ball)

    Using the next snap (not the scoring play itself) means touchdowns include the
    extra point and every score 'pays back' the kickoff, so punts, field goals,
    turnovers on downs and conversions are all measured on the same scale.
    """
    snaps = (
        plays.filter(pl.col("down").is_not_null(), pl.col("posteam").is_not_null(), pl.col("ep").is_not_null())
        .select("game_id", "play_id", "posteam", "posteam_type", "game_half", "score_differential", "ep", "result")
        .sort("game_id", "play_id")
        .with_columns(
            n_pos=pl.col("posteam").shift(-1).over("game_id"),
            n_half=pl.col("game_half").shift(-1).over("game_id"),
            n_diff=pl.col("score_differential").shift(-1).over("game_id"),
            n_ep=pl.col("ep").shift(-1).over("game_id"),
        )
    )
    same = pl.col("n_pos") == pl.col("posteam")
    final_diff = pl.when(pl.col("posteam_type") == "home").then(pl.col("result")).otherwise(-pl.col("result"))
    next_diff_my_view = pl.when(same).then(pl.col("n_diff")).otherwise(-pl.col("n_diff"))
    next_ep_my_view = (
        pl.when(pl.col("n_half") != pl.col("game_half")).then(0.0)
        .when(same).then(pl.col("n_ep")).otherwise(-pl.col("n_ep"))
    )
    out = snaps.with_columns(
        value_after=pl.when(pl.col("n_pos").is_null())  # last snap of the game
        .then((final_diff - pl.col("score_differential")).cast(pl.Float64))
        .otherwise((next_diff_my_view - pl.col("score_differential")) + next_ep_my_view)
    )
    return plays.join(out.select("game_id", "play_id", "value_after"), on=["game_id", "play_id"], how="left")


# --------------------------------------------------------------------------- chart style

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#f0efec",
}
BUCKET_COLORS = dict(zip(BUCKETS, [C["blue"], C["orange"], C["aqua"]]))
DECISION_COLORS = {"Go for it": C["blue"], "Field goal": C["orange"], "Punt": C["aqua"]}
TEAM_COLORS = {"GB": C["aqua"], "DET": C["blue"]}

mpl.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": C["ink"], "axes.labelcolor": C["ink2"], "axes.edgecolor": C["axis"],
    "xtick.color": C["muted"], "ytick.color": C["muted"], "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titlesize": 16, "axes.titleweight": "bold",
    "figure.dpi": 100, "savefig.dpi": 200,
})


def new_fig(w: float = 10, h: float = 6.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    return fig, ax


def finish(fig, ax, name: str, title: str, subtitle: str, source: str = "Data: nflverse play-by-play",
           top: float = 0.80, left: float = 0.08):
    """Title + subtitle top-left, source bottom-left; save PNG sized for X (16:10)."""
    fig.subplots_adjust(top=top, bottom=0.14, left=left, right=0.97)
    fig.text(0.02, 0.955, title, fontsize=19, fontweight="bold", color=C["ink"], va="top")
    fig.text(0.02, 0.885, subtitle, fontsize=12.5, color=C["ink2"], va="top")
    fig.text(0.02, 0.025, source, fontsize=9.5, color=C["muted"])
    ax.set_title("")
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


