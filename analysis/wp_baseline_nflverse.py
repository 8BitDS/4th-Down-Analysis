"""Benchmark: how accurate is nflverse's own win-probability model on our data?

Metrics (lower = better for all three):
  log loss          - penalises confident wrong calls heavily (the metric nflverse trains on)
  Brier score       - mean (p - outcome)^2; 0.25 = always guessing 50%
  calibration error - nflverse's definition: bin predictions in 5% steps within each quarter,
                      |mean predicted - actual win rate| weighted by plays
"""

from sklearn.metrics import brier_score_loss, log_loss

from common import *

w = load("wp_plays").filter(pl.col("posteam_win") != 0.5, pl.col("qtr") <= 4, pl.col("season") >= 2015)
w = w.with_columns(y=pl.col("posteam_win").cast(pl.Int8))


def cal_error(frame, col):
    t = (frame.with_columns(b=(pl.col(col) / 0.05).floor())
         .group_by("qtr", "b").agg(p=pl.col(col).mean(), a=pl.col("y").mean(), n=pl.len()))
    return float((t["n"] * (t["p"] - t["a"]).abs()).sum() / t["n"].sum())


def metrics(frame, col):
    f = frame.drop_nulls(col)
    p = f[col].clip(1e-6, 1 - 1e-6).to_numpy()
    y = f["y"].to_numpy()
    return dict(n=f.height, logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                acc=((p > 0.5) == y).mean(), cal=cal_error(f, col))


def show(label, frame):
    a, b = metrics(frame, "wp"), metrics(frame, "vegas_wp")
    print(f"  {label:32s} n={a['n']:>8,} | wp: logloss {a['logloss']:.3f} Brier {a['brier']:.3f} acc {a['acc']:.1%} cal {a['cal']:.4f}"
          f" | vegas_wp: logloss {b['logloss']:.3f} Brier {b['brier']:.3f} acc {b['acc']:.1%} cal {b['cal']:.4f}")


print("nflverse WP accuracy on our data (ties and overtime excluded):")
show("All plays 2015-2025", w.filter(pl.col("season") <= 2025))
show("2026 season so far", w.filter(pl.col("season") == 2026))
show("4th-down plays 2015-2025", w.filter(pl.col("season") <= 2025, pl.col("down") == 4))
show("Q4, last 5 min, 2015-2025", w.filter(pl.col("season") <= 2025, pl.col("qtr") == 4, pl.col("game_seconds_remaining") <= 300))
show("Close games (|score| <= 8), Q4", w.filter(pl.col("season") <= 2025, pl.col("qtr") == 4, pl.col("score_differential").abs() <= 8))

print("\nBy season (vegas_wp log loss / calibration error):")
for s in range(2015, 2027):
    m = metrics(w.filter(pl.col("season") == s), "vegas_wp")
    print(f"  {s}: logloss {m['logloss']:.3f}  Brier {m['brier']:.3f}  cal {m['cal']:.4f}  (n={m['n']:,})")

print("\nBy quarter (all 2015-2025): how much is knowable as the game goes on")
for q in (1, 2, 3, 4):
    m = metrics(w.filter(pl.col("season") <= 2025, pl.col("qtr") == q), "vegas_wp")
    print(f"  Q{q}: logloss {m['logloss']:.3f}  Brier {m['brier']:.3f}  accuracy {m['acc']:.1%}")
