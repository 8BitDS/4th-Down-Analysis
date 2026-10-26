"""Logistic regression baseline for win probability, on nflverse's 12 features.

Same test as the XGBoost replica: leave-one-season-out for 2015-2025, 2026 from a 1999-2025 fit.
Logistic regression = one weight per feature, added up on the log-odds scale:
    log-odds(win) = b0 + b1*x1 + b2*x2 + ...      p = 1 / (1 + e^-(log-odds))
It can't bend or interact on its own, which is exactly what we want to see: what do the trees add?
"""

import json

import statsmodels.api as sm

from wp_common import *

w = load_wp_data()
FEATS = FEATS_SPREAD
oof = pl.read_parquet(MODELS / "wp_replica_oof.parquet").select("game_id", "play_id", "ours_vegas_wp").rename({"ours_vegas_wp": "xgb"})


def fit(frame):
    X = sm.add_constant(frame.select(FEATS).to_numpy().astype(float))
    return sm.Logit(frame["y"].to_numpy(), X).fit(disp=0, maxiter=200)


def pred(m, frame):
    return m.predict(sm.add_constant(frame.select(FEATS).to_numpy().astype(float), has_constant="add"))


parts = []
for s in TEST_SEASONS:
    te = w.filter(pl.col("season") == s)
    parts.append(te.select("game_id", "play_id").with_columns(pl.Series("logit", pred(fit(w.filter(pl.col("season") != s, pl.col("season") <= 2025)), te))))
final = fit(w.filter(pl.col("season") <= 2025))
te = w.filter(pl.col("season") == 2026)
parts.append(te.select("game_id", "play_id").with_columns(pl.Series("logit", pred(final, te))))
ev = w.filter(pl.col("season") >= 2015).join(pl.concat(parts), on=["game_id", "play_id"]).join(oof, on=["game_id", "play_id"], how="left")
ev.select("game_id", "play_id", "logit").write_parquet(MODELS / "wp_logit_oof.parquet")

print("                     | Logistic regression              | XGBoost replica")
print("                     | logloss  Brier  acc    cal       | logloss  Brier  acc    cal")
res = {}
for label, sel in SUBSETS.items():
    f = sel(ev)
    a, b = score(f, "logit"), score(f, "xgb")
    res[label] = dict(logit=a, xgb=b)
    print(f"  {label:19s}| {a['logloss']:.3f}   {a['brier']:.3f}  {a['acc']:.1%}  {a['cal']:.4f}    | "
          f"{b['logloss']:.3f}   {b['brier']:.3f}  {b['acc']:.1%}  {b['cal']:.4f}")
(RES / "wp_logistic_metrics.json").write_text(json.dumps(res, indent=2))

# Where does logistic fall short? Calibration error by quarter
print("\nCalibration error by quarter (2015-25):")
for q in (1, 2, 3, 4):
    f = ev.filter(pl.col("season") <= 2025, pl.col("qtr") == q)
    print(f"  Q{q}: logistic {cal_error(f, 'logit'):.4f} | XGBoost {cal_error(f, 'xgb'):.4f}")

# Coefficients for the maths corner (fit on 1999-2025), as odds multipliers
print("\nLogistic coefficients (1999-2025 fit):")
names = ["const"] + FEATS
for n_, b_, se in zip(names, final.params, final.bse):
    print(f"  {n_:28s} {b_:+.5f}  (se {se:.5f})  odds x{np.exp(b_):.4f} per unit")
b = dict(zip(names, final.params))
print("\nPlain-English reads:")
print(f"  +1 point of spread at kickoff (decays over the game): odds x{np.exp(b['spread_time']):.3f} at kickoff")
print(f"  +1 point of lead: odds x{np.exp(b['score_differential'] + b['Diff_Time_Ratio']):.3f} at kickoff, "
      f"x{np.exp(b['score_differential'] + b['Diff_Time_Ratio'] / np.exp(-4 * 0.5)):.3f} at halftime, "
      f"x{np.exp(b['score_differential'] + b['Diff_Time_Ratio'] / np.exp(-4 * (55/60))):.3f} with 5 min left")
print(f"  +10 yards closer to the end zone: odds x{np.exp(-10 * b['yardline_100']):.3f}")
print(f"  +1 timeout (yours): odds x{np.exp(b['posteam_timeouts_remaining']):.3f}")

# Case plays
cases = w.filter(pl.col("game_id") == "2026_07_GB_DET", pl.col("down") == 4,
                 ((pl.col("posteam") == "GB") & (pl.col("yardline_100") == 7)) | ((pl.col("posteam") == "DET") & (pl.col("yardline_100") == 76)))
for r, p in zip(cases.iter_rows(named=True), pred(final, cases)):
    print(f"Case {r['posteam']}: logistic {pct(float(p),1)} | nflverse vegas_wp {pct(r['vegas_wp'],1)}")

# ---- chart: Q4 calibration, logistic vs XGBoost (where they differ most)
fig, ax = new_fig(10, 6.4)
f = ev.filter(pl.col("season") <= 2025, pl.col("qtr") == 4)
ax.plot([0, 1], [0, 1], color=C["axis"], lw=1)
for col, lab, colr in [("xgb", "XGBoost (nflverse replica)", C["blue"]), ("logit", "Logistic regression", C["orange"])]:
    t = f.with_columns(b=(pl.col(col) / 0.05).floor()).group_by("b").agg(p=pl.col(col).mean(), a=pl.col("y").mean()).sort("b")
    ax.plot(t["p"], t["a"], color=colr, lw=2, label=f"{lab}: calibration error {cal_error(f, col):.3f}")
    ax.scatter(t["p"], t["a"], s=22, color=colr, edgecolor=C["surface"], lw=1, zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Predicted win probability (4th quarter)")
ax.set_ylabel("Actual win rate")
ax.legend(loc="upper left", fontsize=10.5)
r_all = res["All plays 2015-25"]
finish(fig, ax, "wp_logistic_vs_xgb",
       "A straight-line model gets close. Trees are about twice as well calibrated",
       f"Out-of-sample, 2015–2025. Log loss: logistic {r_all['logit']['logloss']:.3f} vs XGBoost {r_all['xgb']['logloss']:.3f}. "
       "4th quarter shown; the gap is similar in every quarter.", top=0.80)
