"""Polars price preparation, historical eligibility and monthly IC adapters."""

from pathlib import Path

import polars as pl

from factor_test.analyzer import daily_information_coefficient
from factor_test.diagnostics import quantize_factor
from example.bias_factor import DEFAULT_BIAS_WINDOW, BiasFactor


def tradable_price() -> pl.Expr:
    return (
        pl.col("open").is_finite()
        & (pl.col("open") > 0)
        & pl.col("close").is_finite()
        & (pl.col("close") > 0)
        & (pl.col("volume") > 0)
    ).fill_null(False)


def load_panel(cache: Path, data_start: str, end: str):
    """Scan cached long-format quotes without building Pandas price matrices."""
    market = pl.read_parquet(cache / f"csi500_{data_start}_{end}.parquet").sort("date")
    first, last = market["date"].min(), market["date"].max()
    stocks = (
        pl.read_parquet(cache / "universe.parquet")
        .filter(
            ~pl.col("symbol").str.starts_with("689")
            & (pl.col("listing_date") <= last)
            & (pl.col("delisting_date").is_null() | (pl.col("delisting_date") >= first))
        )
        .sort("symbol")
    )
    quotes = pl.scan_parquet(str(cache / f"hfq_{data_start}_{end}" / "*.parquet"))
    if "source" not in quotes.collect_schema():
        quotes = quotes.with_columns(pl.lit("unknown").alias("source"))
    quotes = quotes.join(stocks.lazy().select("symbol"), on="symbol", how="inner")
    in_calendar = pl.col("date").is_in(market["date"].implode())
    good_price = (
        pl.col("open").is_finite()
        & (pl.col("open") > 0)
        & pl.col("close").is_finite()
        & (pl.col("close") > 0)
    ).fill_null(False)
    diagnostics = quotes.group_by("symbol").agg(
        pl.len().alias("rows"),
        pl.col("source").first(),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
        (in_calendar & ~good_price).sum().alias("invalid_price_rows"),
        (in_calendar & (pl.col("volume") <= 0)).sum().alias("zero_volume_rows"),
    )
    quality = (
        stocks.lazy()
        .join(diagnostics, on="symbol", how="left")
        .with_columns(
            pl.col("rows").is_not_null().alias("downloaded"),
            pl.col("rows", "invalid_price_rows", "zero_volume_rows").fill_null(0),
        )
        .sort("symbol")
        .collect()
    )
    return (
        quotes.select("date", "symbol", "open", "close", "volume").filter(in_calendar),
        stocks,
        market,
        quality,
    )


def build_signals(
    quotes: pl.LazyFrame,
    stocks: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    lookback: int = DEFAULT_BIAS_WINDOW,
    min_age: int = 60,
    signal_dates=None,
):
    """Align exchange sessions, carry past closes, then call the BaseFactor subclass."""
    if type(min_age) is not int or min_age < 1:
        raise ValueError("min_age must be a positive integer")
    factor = BiasFactor(window=lookback)
    dates = calendar.select("date").unique().sort("date").with_row_index("_session")
    if signal_dates is None:
        signal_dates = (
            dates.group_by(pl.col("date").dt.truncate("1mo"))
            .agg(pl.col("date").max().alias("_month_end"))["_month_end"]
            .sort()
            .to_list()
        )
    dates = dates.with_columns(
        pl.col("date").shift(lookback - 1).alias("_history_start")
    )
    stocks = stocks.with_columns(
        dates["date"]
        .search_sorted(stocks["listing_date"])
        .cast(pl.Int64)
        .alias("_listing_session")
    )
    prepared = (
        dates.lazy()
        .join(stocks.lazy(), how="cross")
        .join(quotes, on=["date", "symbol"], how="left")
        .with_columns(tradable_price().alias("_tradable"))
        .with_columns(pl.when(pl.col("_tradable")).then(pl.col("close")).alias("close"))
        .sort(["symbol", "date"])
        .with_columns(pl.col("close").forward_fill().over("symbol"))
    )
    listed = (
        pl.col("listing_date").is_not_null()
        & (pl.col("listing_date") <= pl.col("date"))
        & (
            pl.col("delisting_date").is_null()
            | (pl.col("date") < pl.col("delisting_date"))
        )
    )
    snapshot = (
        factor.compute(prepared)
        .filter(pl.col("date").is_in(signal_dates))
        .with_columns(
            listed.alias("_listed"),
            (pl.col("_session").cast(pl.Int64) - pl.col("_listing_session") + 1).alias(
                "listing_age"
            ),
        )
        .with_columns(
            (pl.col("_listed") & (pl.col("listing_age") >= min_age)).alias("_old")
        )
    )
    snapshot = (
        snapshot.with_columns(
            (
                pl.col("_old")
                & (pl.col("listing_date") <= pl.col("_history_start"))
                & pl.col("_tradable")
                & pl.col("factor").is_finite()
            )
            .fill_null(False)
            .alias("_usable")
        )
        .select("date", "symbol", "factor", "listing_age", "_listed", "_old", "_usable")
        .collect()
    )
    coverage = (
        snapshot.group_by("date")
        .agg(
            pl.col("_listed").sum().alias("listed"),
            (pl.col("_listed") & ~pl.col("_old")).sum().alias("excluded_under_60"),
            pl.col("_old").sum().alias("eligible_age"),
            pl.col("_usable").sum().alias("factor_stocks"),
        )
        .sort("date")
    )
    signals = snapshot.filter(pl.col("_usable")).select(
        "date", "symbol", "factor", "listing_age"
    )
    # Preserve the original per-month calls: scalar vs batched division in the
    # quantizer can round exact quintile boundaries differently by one label.
    if signals.is_empty():
        signals = signals.with_columns(pl.lit(None, dtype=pl.Int64).alias("quantile"))
    else:
        labels = pl.concat(
            [
                quantize_factor(
                    month.select("date", "symbol", "factor").lazy()
                ).collect()
                for month in signals.partition_by("date")
            ]
        )
        signals = signals.join(
            labels.select(
                "date", "symbol", pl.col("factor_quantile").alias("quantile")
            ),
            on=["date", "symbol"],
            how="left",
        )
    return signals.sort(["date", "symbol"]), coverage


def monthly_ic(
    quotes: pl.LazyFrame,
    signals: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    start,
    end,
    signal_dates=None,
):
    """Map consecutive monthly signals to next opens; let src compute both ICs."""
    dates = (
        calendar.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("entry_date"))
    )
    signal_days = (
        signals.select("date").unique().sort("date")
        if signal_dates is None
        else pl.DataFrame({"date": signal_dates}, schema={"date": pl.Date}).sort("date")
    )
    mapping = (
        signal_days.with_columns(pl.col("date").shift(-1).alias("_next_signal"))
        .join(dates, on="date", how="left")
        .join(
            dates.rename({"date": "_next_signal", "entry_date": "exit_date"}),
            on="_next_signal",
            how="left",
        )
        .filter((pl.col("entry_date") >= start) & (pl.col("exit_date") <= end))
    )
    prices = quotes.select(
        "date", "symbol", pl.when(tradable_price()).then(pl.col("open")).alias("_open")
    )
    pairs = (
        signals.lazy()
        .join(mapping.lazy(), on="date", how="inner")
        .join(
            prices.rename({"date": "entry_date", "_open": "_entry_open"}),
            on=["entry_date", "symbol"],
            how="left",
        )
        .join(
            prices.rename({"date": "exit_date", "_open": "_exit_open"}),
            on=["exit_date", "symbol"],
            how="left",
        )
        .with_columns((pl.col("_exit_open") / pl.col("_entry_open") - 1).alias("ret_1"))
    )
    stats = (
        daily_information_coefficient(pairs, horizons=(1,)).collect().drop("horizon")
    )
    return (
        mapping.join(stats, on="date", how="left")
        .with_columns(
            pl.col("n_total", "n_assets").fill_null(0),
            pl.col("coverage").fill_null(0.0),
        )
        .rename({"date": "signal_date"})
        .select(
            "signal_date",
            "entry_date",
            "exit_date",
            "n_total",
            "n_assets",
            "coverage",
            "ic",
            "rank_ic",
        )
        .sort("signal_date")
    )
