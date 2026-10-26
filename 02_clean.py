"""Build clean modelling tables from the raw nflverse parquet files.

Inputs:  data/raw/pbp_*.parquet, data/raw/schedules.parquet  (from 01_pull.py)
Outputs (data/clean/):
    plays.parquet             every real snap (no timeouts / quarter ends / pre-snap fouls) + features
    conv_plays.parquet        3rd & 4th down runs/passes with a `converted` label   -> model A
    fg_attempts.parquet       field goal attempts with kicker history + conditions  -> model B
    wp_plays.parquet          plays with a final-result label for win probability   -> model C
    fourth_decisions.parquet  every 4th-down decision: go / fg / punt / fake        -> coach tendencies
    punts.parquet             punts with where the opponent took over               -> PUNT branch
    team_game_strength.parquet  season-to-date team strength (pre-game only)

No future information: every "to date" feature only uses games BEFORE the current one.
"""

from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

KEEP = [
    # ids / time
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "posteam_type", "drive", "qtr", "game_half",
    # game state
    "down", "ydstogo", "yardline_100", "goal_to_go", "game_seconds_remaining",
    "half_seconds_remaining", "quarter_seconds_remaining", "score_differential",
    "posteam_score", "defteam_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "home_opening_kickoff",
    # play
    "play_type", "desc", "yards_gained", "first_down", "touchdown", "shotgun", "no_huddle",
    "qb_dropback", "qb_scramble", "sack", "interception", "fumble_lost", "penalty", "penalty_type",
    "aborted_play", "passer_player_name", "rusher_player_name",
    "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    # kicking
    "field_goal_result", "kick_distance", "kicker_player_id", "kicker_player_name",
    "punter_player_id", "punter_player_name", "punt_blocked", "touchback", "return_yards",
    # conditions / context
    "roof", "surface", "temp", "wind", "weather", "spread_line", "total_line", "div_game",
    "home_coach", "away_coach", "stadium",
    # nflverse model outputs (benchmarks)
    "ep", "epa", "wp", "def_wp", "wpa", "vegas_wp",
    # final score
    "result",
]

PRE_SNAP_FOULS = [
    "False Start", "Delay of Game", "Defensive Offside", "Neutral Zone Infraction",
    "Encroachment", "Defensive 12 On-field", "Offensive 12 On-field", "Illegal Shift",
    "Illegal Substitution", "Defensive Delay of Game",
]


# --------------------------------------------------------------------------- base


def load_base() -> pl.DataFrame:
    # Column types drift between seasons (e.g. Int32 vs Float64), so read each file and relax.
    files = sorted(RAW.glob("pbp_*.parquet"))
    pbp = pl.concat([pl.read_parquet(f, columns=KEEP) for f in files], how="vertical_relaxed")
    print(f"raw pbp: {pbp.height:,} rows")
    int_cols = ["season", "week", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go", "drive",
                "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home_opening_kickoff",
                "first_down", "touchdown", "penalty", "aborted_play", "third_down_converted",
                "third_down_failed", "fourth_down_converted", "fourth_down_failed", "kick_distance",
                "punt_blocked", "touchback", "div_game"]
    pbp = pbp.with_columns(pl.col(int_cols).cast(pl.Int32))

    is_presnap_foul = (pl.col("play_type") == "no_play") & (
        pl.col("penalty_type").is_in(PRE_SNAP_FOULS)
        # desc like "(5:00) (Punt formation) PENALTY on ..." = foul before any action
        | pl.col("desc").str.contains(r"^\(\d+:\d+\)\s*(\([^)]*\)\s*)?PENALTY")
    )

    df = pbp.filter(
        pl.col("play_type").is_not_null(),
        pl.col("play_type") != "",
        pl.col("down").is_not_null() | pl.col("play_type").is_in(["kickoff", "extra_point"]),
        ~is_presnap_foul,
    )
    print(f"after removing non-plays and pre-snap fouls: {df.height:,} rows")

    indoor = pl.col("roof").is_in(["dome", "closed"])
    wind_from_text = (
        pl.col("weather").str.extract(r"(?i)wind[:\s]+[A-Z]*\s*(\d+)\s*mph", 1).cast(pl.Float64, strict=False)
    )
    calm = pl.col("weather").str.contains(r"(?i)wind[:\s]+calm")

    df = df.with_columns(
        pl.col("game_date").str.to_date(),
        surface=pl.when(pl.col("surface").str.strip_chars().is_in(["grass", "dessograss"]))
        .then(pl.lit("grass"))
        .when(pl.col("surface").str.strip_chars() == "")
        .then(None)
        .otherwise(pl.lit("turf")),
        indoor=indoor.cast(pl.Int8),
        # Wind: 0 indoors; outdoors use the column, else parse the weather text.
        wind_filled=pl.when(indoor).then(0.0)
        .otherwise(pl.coalesce(pl.col("wind"), wind_from_text, pl.when(calm).then(0.0))),
        temp_filled=pl.when(indoor).then(70.0).otherwise(pl.col("temp")),
        # Spread from the offence's point of view (positive = offence favoured)
        posteam_spread=pl.when(pl.col("posteam_type") == "home")
        .then(pl.col("spread_line"))
        .otherwise(-pl.col("spread_line")),
        # Rule eras that change kicking / WP maths
        xp_era=(pl.col("season") >= 2015).cast(pl.Int8),  # PAT moved back to 33 yds
        kickoff_rule=pl.when(pl.col("season") >= 2024).then(pl.lit("dynamic")).otherwise(pl.lit("traditional")),
        ot_rule=pl.when(pl.col("season") < 2012).then(pl.lit("sudden_death"))
        .when(pl.col("season") < 2017).then(pl.lit("modified_15min"))
        .when(pl.col("season") < 2025).then(pl.lit("modified_10min"))
        .otherwise(pl.lit("both_possess_10min")),
        posteam_receives_2h_ko=pl.when(pl.col("posteam_type") == "home")
        .then(1 - pl.col("home_opening_kickoff"))
        .otherwise(pl.col("home_opening_kickoff")),
        # Final outcome from the offence's point of view (tie = 0.5)
        posteam_win=pl.when(pl.col("result") == 0).then(0.5)
        .when((pl.col("posteam_type") == "home") == (pl.col("result") > 0)).then(1.0)
        .otherwise(0.0),
        low_leverage=((pl.col("wp") < 0.05) | (pl.col("wp") > 0.95)).cast(pl.Int8),
    )
    return df.sort("game_id", "play_id")


def add_next_possession(df: pl.DataFrame) -> pl.DataFrame:
    """Where the next snap happened (needed for punt / missed FG outcomes)."""
    snaps = (
        df.filter(pl.col("down").is_not_null())
        .select("game_id", "play_id", "posteam", "yardline_100")
        .sort("game_id", "play_id")
        .with_columns(
            next_posteam=pl.col("posteam").shift(-1).over("game_id"),
            next_yardline_100=pl.col("yardline_100").shift(-1).over("game_id"),
        )
        .select("game_id", "play_id", "next_posteam", "next_yardline_100")
    )
    return df.join(snaps, on=["game_id", "play_id"], how="left")


# --------------------------------------------------------------------------- team strength


def team_strength(df: pl.DataFrame) -> pl.DataFrame:
    """Per team-game strength, using only games BEFORE that game (no leakage)."""
    scrimmage = df.filter(pl.col("play_type").is_in(["run", "pass"]), pl.col("epa").is_not_null())
    short = scrimmage.filter(pl.col("down").is_in([3, 4]), pl.col("ydstogo") <= 2).with_columns(
        conv=pl.coalesce(
            pl.when(pl.col("down") == 3).then(pl.col("third_down_converted")).otherwise(pl.col("fourth_down_converted"))
        )
    )

    def per_game(side: str) -> pl.DataFrame:
        team = "posteam" if side == "off" else "defteam"
        g = scrimmage.group_by("game_id", "season", "week", "game_date", pl.col(team).alias("team")).agg(
            epa_sum=pl.col("epa").sum(), n=pl.len()
        )
        s = short.group_by("game_id", pl.col(team).alias("team")).agg(sy_conv=pl.col("conv").sum(), sy_n=pl.len())
        g = g.join(s, on=["game_id", "team"], how="left").fill_null(0).sort("team", "game_date")
        prior = lambda c: pl.col(c).cum_sum().shift(1).over("team", "season")  # noqa: E731
        g = g.with_columns(
            **{
                f"{side}_epa_std": prior("epa_sum") / prior("n"),
                f"{side}_plays_std": prior("n"),
                f"{side}_sy_rate_std": prior("sy_conv") / prior("sy_n"),
                f"{side}_sy_n_std": prior("sy_n"),
                f"{side}_games_std": pl.col("n").cum_count().shift(1).over("team", "season"),
            }
        )
        # Full previous-season value: fallback for early weeks
        prev = g.group_by("team", "season").agg(
            (pl.col("epa_sum").sum() / pl.col("n").sum()).alias(f"{side}_epa_prev_season")
        ).with_columns(pl.col("season") + 1)
        return g.join(prev, on=["team", "season"], how="left").select(
            "game_id", "team", pl.col("^" + side + "_.*$")
        )

    out = per_game("off").join(per_game("def"), on=["game_id", "team"], how="full", coalesce=True)
    return out


def attach_strength(df: pl.DataFrame, ts: pl.DataFrame) -> pl.DataFrame:
    off = ts.select("game_id", "team", "off_epa_std", "off_epa_prev_season", "off_sy_rate_std", "off_sy_n_std", "off_games_std")
    dfn = ts.select("game_id", "team", "def_epa_std", "def_epa_prev_season", "def_sy_rate_std", "def_sy_n_std")
    return df.join(off, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="left").join(
        dfn, left_on=["game_id", "defteam"], right_on=["game_id", "team"], how="left"
    )


# --------------------------------------------------------------------------- output tables


def fourth_decisions(df: pl.DataFrame) -> pl.DataFrame:
    fake = pl.col("desc").str.contains(r"(?i)\((punt|field goal) formation\)") & pl.col("play_type").is_in(["run", "pass", "no_play"]) & ~pl.col("desc").str.contains(r"(?i)\bpunts\b|field goal is")
    d = df.filter(pl.col("down") == 4, pl.col("play_type").is_in(["run", "pass", "punt", "field_goal", "no_play"]))
    d = d.with_columns(
        nullified_by_penalty=(pl.col("play_type") == "no_play").cast(pl.Int8),
        decision=pl.when(fake).then(pl.lit("fake_or_botched"))
        .when(pl.col("play_type") == "punt").then(pl.lit("punt"))
        .when(pl.col("play_type") == "field_goal").then(pl.lit("fg"))
        .when(pl.col("play_type").is_in(["run", "pass"])).then(pl.lit("go"))
        # nullified plays: infer the decision from the text
        .when(pl.col("desc").str.contains(r"(?i)\bpunts\b|punt is blocked")).then(pl.lit("punt"))
        .when(pl.col("desc").str.contains(r"(?i)field goal")).then(pl.lit("fg"))
        .otherwise(pl.lit("go")),
    )
    return d


def conv_plays(df: pl.DataFrame) -> pl.DataFrame:
    fake = pl.col("desc").str.contains(r"(?i)\((punt|field goal) formation\)")
    c = df.filter(
        pl.col("down").is_in([3, 4]),
        pl.col("play_type").is_in(["run", "pass"]),
        ~fake,
        pl.col("aborted_play").fill_null(0) == 0,
    ).with_columns(
        converted=pl.when(pl.col("down") == 3).then(pl.col("third_down_converted")).otherwise(pl.col("fourth_down_converted")),
        failed=pl.when(pl.col("down") == 3).then(pl.col("third_down_failed")).otherwise(pl.col("fourth_down_failed")),
        is_pass=(pl.col("play_type") == "pass").cast(pl.Int8),
    )
    # Drop the handful of plays with no outcome (offsetting penalties etc.)
    c = c.filter((pl.col("converted") + pl.col("failed")) == 1).drop("failed")
    return c.with_columns(pl.col("converted").cast(pl.Int8))


def fg_attempts(df: pl.DataFrame) -> pl.DataFrame:
    f = df.filter(pl.col("play_type") == "field_goal", pl.col("kick_distance").is_not_null()).sort("game_date", "game_id", "play_id")
    f = f.with_columns(
        made=(pl.col("field_goal_result") == "made").cast(pl.Int8),
        blocked=(pl.col("field_goal_result") == "blocked").cast(pl.Int8),
        dist_band=pl.when(pl.col("kick_distance") < 30).then(pl.lit("<30"))
        .when(pl.col("kick_distance") < 40).then(pl.lit("30-39"))
        .when(pl.col("kick_distance") < 50).then(pl.lit("40-49"))
        .when(pl.col("kick_distance") < 55).then(pl.lit("50-54"))
        .otherwise(pl.lit("55+")),
    )
    # Kicker career history BEFORE this kick (for shrinkage / priors)
    prior = lambda e: e.cum_sum().shift(1, fill_value=0).over("kicker_player_id")  # noqa: E731
    return f.with_columns(
        kicker_prior_att=prior(pl.col("made").is_not_null().cast(pl.Int32)),
        kicker_prior_made=prior(pl.col("made").cast(pl.Int32)),
        kicker_prior_att_40plus=prior((pl.col("kick_distance") >= 40).cast(pl.Int32)),
        kicker_prior_made_40plus=prior(((pl.col("kick_distance") >= 40) & (pl.col("made") == 1)).cast(pl.Int32)),
    )


def punts(df: pl.DataFrame) -> pl.DataFrame:
    p = df.filter(pl.col("play_type") == "punt")
    return p.with_columns(
        # Opponent's first snap: yards from the end zone THEY attack (net_yards flips it to the punter's view)
        opp_start_yardline_100=pl.when(pl.col("next_posteam") == pl.col("defteam")).then(pl.col("next_yardline_100")),
    ).with_columns(
        net_yards=pl.col("yardline_100") - (100 - pl.col("opp_start_yardline_100")),
    )


def wp_plays(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("down").is_not_null(),
        pl.col("play_type").is_in(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"]),
        pl.col("posteam").is_not_null(),
        pl.col("result").is_not_null(),
    )


# --------------------------------------------------------------------------- main


def report(name: str, d: pl.DataFrame, key_cols: list[str]) -> None:
    print(f"\n{name}: {d.height:,} rows, seasons {d['season'].min()}-{d['season'].max()}")
    nulls = {c: round(d[c].null_count() / max(d.height, 1) * 100, 1) for c in key_cols if c in d.columns}
    print("  % missing:", {k: v for k, v in nulls.items()})


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    df = add_next_possession(load_base())

    ts = team_strength(df)
    ts.write_parquet(CLEAN / "team_game_strength.parquet")
    df = attach_strength(df, ts)
    df.write_parquet(CLEAN / "plays.parquet")

    tables = {
        "conv_plays": conv_plays(df),
        "fg_attempts": fg_attempts(df),
        "wp_plays": wp_plays(df),
        "fourth_decisions": fourth_decisions(df),
        "punts": punts(df),
    }
    for name, t in tables.items():
        t.write_parquet(CLEAN / f"{name}.parquet")

    # ---- sanity report
    core = ["down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining",
            "posteam_timeouts_remaining", "posteam_spread", "wp", "ep", "wind_filled", "temp_filled",
            "off_epa_std", "def_epa_std", "off_epa_prev_season"]
    report("plays", df, core)
    report("conv_plays", tables["conv_plays"], core + ["converted"])
    report("fg_attempts", tables["fg_attempts"], ["kick_distance", "kicker_player_id", "wind_filled", "made"])
    report("wp_plays", tables["wp_plays"], core + ["posteam_win"])
    report("fourth_decisions", tables["fourth_decisions"], core)
    report("punts", tables["punts"], ["opp_start_yardline_100", "net_yards"])

    pl.Config.set_tbl_rows(30)
    c = tables["conv_plays"]
    print("\nConversion rate by yards to go (2015+):")
    print(
        c.filter(pl.col("season") >= 2015, pl.col("ydstogo") <= 10)
        .group_by("down", "ydstogo").agg(rate=pl.col("converted").mean().round(3), n=pl.len())
        .pivot(on="down", index="ydstogo", values=["rate", "n"]).sort("ydstogo")
    )
    print("\n4th-down decisions by season (share):")
    print(
        tables["fourth_decisions"].group_by("season").agg(
            n=pl.len(),
            go=(pl.col("decision") == "go").mean().round(3),
            fg=(pl.col("decision") == "fg").mean().round(3),
            punt=(pl.col("decision") == "punt").mean().round(3),
            fake=(pl.col("decision") == "fake_or_botched").mean().round(4),
        ).sort("season")
    )
    f = tables["fg_attempts"]
    print("\nFG make rate by distance band (2015+):")
    print(f.filter(pl.col("season") >= 2015).group_by("dist_band").agg(rate=pl.col("made").mean().round(3), n=pl.len()).sort("dist_band"))
    print("\nPunt net yards (2015+):", tables["punts"].filter(pl.col("season") >= 2015)["net_yards"].describe())


if __name__ == "__main__":
    main()
