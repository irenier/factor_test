"""Polars BIAS20 research: BaseFactor signals and src account execution."""

import argparse
from datetime import UTC, date, datetime, timedelta
import gzip
import json
from pathlib import Path
import sys

import polars as pl

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example.bias_factor import DEFAULT_BIAS_WINDOW
from example.framework_account import run_accounts
from example.momentum_inputs import (
    build_signals,
    load_panel,
    monthly_ic,
    tradable_price,
)

BIAS_LOOKBACK = DEFAULT_BIAS_WINDOW


def prepare_results(daily, metrics, benchmark):
    """Join benchmark NAV and aggregate returns entirely in Polars."""
    daily = daily.join(benchmark, on="date", how="left").with_columns(
        (pl.col("nav") / pl.col("benchmark_nav")).alias("relative_nav"),
        (pl.col("nav") - pl.col("benchmark_nav")).alias("cumulative_return_difference"),
    )
    nav = (
        daily.pivot(on="portfolio", index="date", values="nav")
        .join(benchmark.rename({"benchmark_nav": "CSI500"}), on="date")
        .sort("date")
    )
    labels = ["Momentum_LS", "Momentum_LongOnly", "CSI500"]
    relative = (
        nav.unpivot(
            on=labels, index="date", variable_name="portfolio", value_name="_nav"
        )
        .join(benchmark, on="date")
        .sort(["portfolio", "date"])
        .with_columns((pl.col("_nav") / pl.col("benchmark_nav")).alias("_relative"))
    )
    drawdown = (
        relative.with_columns(
            (
                pl.col("_relative")
                / pl.col("_relative").cum_max().over("portfolio").clip(lower_bound=1.0)
                - 1
            ).alias("relative_max_drawdown")
        )
        .group_by("portfolio")
        .agg(pl.col("relative_max_drawdown").min())
    )
    metrics = (
        metrics.rename({"annual_tracking_error": "tracking_error"})
        .with_columns(
            ((1 + pl.col("relative_return")).pow(252 / nav.height) - 1).alias(
                "annual_relative_return"
            ),
            (pl.col("total_return") - (benchmark["benchmark_nav"][-1] - 1)).alias(
                "return_difference"
            ),
        )
        .join(drawdown, on="portfolio", how="left")
    )
    annual = (
        nav.with_columns(pl.col(labels) / pl.col(labels).shift(1).fill_null(1.0))
        .group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col(labels).product() - 1)
        .sort("year")
    )
    for label, prefix in [("Momentum_LS", "LS"), ("Momentum_LongOnly", "LO")]:
        annual = annual.with_columns(
            (pl.col(label) - pl.col("CSI500")).alias(f"{prefix}_minus_CSI500"),
            ((1 + pl.col(label)) / (1 + pl.col("CSI500")) - 1).alias(
                f"{prefix}_relative"
            ),
        )
    return daily, metrics, nav, annual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-start", default="20140101")
    parser.add_argument("--cache", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-tax-rate", type=float, default=0.001)
    parser.add_argument("--initial-value", type=float, default=100_000_000.0)
    parser.add_argument("--quantile-pct", type=float, default=0.2)
    parser.add_argument("--short-borrow-rate", type=float, default=0.0)
    parser.add_argument(
        "--offline", action="store_true", help="Only use local cached data"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--provider", choices=["sina", "em", "tx"], default="sina")
    parser.add_argument(
        "--direct", action="store_true", help="Use direct network access for downloads"
    )
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start > end:
        parser.error("start must be <= end")
    end_key = end.strftime("%Y%m%d")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        from example.akshare_data import (
            benchmark,
            configure_requests,
            download_prices,
            universe,
        )

        configure_requests(direct=args.direct)
        args.cache.mkdir(parents=True, exist_ok=True)
        stocks = universe(args.cache).filter(
            (pl.col("listing_date") <= end)
            & (
                pl.col("delisting_date").is_null()
                | (
                    pl.col("delisting_date")
                    >= datetime.strptime(args.data_start, "%Y%m%d").date()
                )
            )
        )
        benchmark(args.cache, args.data_start, end_key)
        download_prices(
            stocks, args.cache, args.data_start, end_key, args.workers, args.provider
        )

    print("Scanning cached quotes with Polars...", flush=True)
    quotes, stocks, market, quality = load_panel(args.cache, args.data_start, end_key)
    if market["date"].max().year < end.year or market["date"].min() > start - timedelta(
        days=370
    ):
        raise RuntimeError(
            "Benchmark history does not cover the requested period and warm-up"
        )
    calendar = market.select("date")
    print("Computing BiasFactor on the common trading calendar...", flush=True)
    signals, coverage = build_signals(quotes, stocks, calendar, lookback=BIAS_LOOKBACK)
    anchor = market.filter(pl.col("date") < start)["date"][-1]
    signals = signals.filter(pl.col("date").is_between(anchor, end, closed="left"))
    coverage = coverage.filter(pl.col("date").is_between(anchor, end, closed="left"))
    prices = quotes.filter(pl.col("date").is_between(anchor, end)).select(
        "date", "symbol", "open", (~tradable_price()).alias("suspended")
    )
    benchmark_nav = market.filter(pl.col("date").is_between(start, end)).select(
        "date", (pl.col("open") / pl.col("open").first()).alias("benchmark_nav")
    )
    daily, metrics, trades, unfilled, rebalances = run_accounts(
        signals,
        prices,
        benchmark_nav,
        start=start,
        initial_value=args.initial_value,
        commission_rate=args.commission_rate,
        sell_tax_rate=args.sell_tax_rate,
        quantile_pct=args.quantile_pct,
        short_borrow_rate=args.short_borrow_rate,
    )
    daily, metrics, nav, annual = prepare_results(daily, metrics, benchmark_nav)
    ic = monthly_ic(
        quotes,
        signals,
        calendar,
        start=start,
        end=end,
        signal_dates=coverage["date"].to_list(),
    ).with_columns(
        (-pl.col("ic")).alias("direction_adjusted_ic"),
        (-pl.col("rank_ic")).alias("direction_adjusted_rank_ic"),
    )
    tables = dict(
        performance=metrics,
        daily_nav=daily,
        annual_returns=annual,
        monthly_ic=ic,
        universe_coverage=coverage,
        data_quality=quality,
        rebalance_summary=rebalances,
    )
    for name, table in tables.items():
        table.write_csv(args.output / f"{name}.csv", include_bom=True)
    factor_table = signals.rename({"date": "signal_date"}).with_columns(
        (-pl.col("factor")).alias("strategy_factor")
    )
    for name, table in [
        ("trades", trades),
        ("unfilled_orders", unfilled),
        ("factor_signals", factor_table),
    ]:
        with gzip.open(args.output / f"{name}.csv.gz", "wb") as destination:
            table.write_csv(destination, include_bom=True)
    metadata = dict(
        start=str(start),
        end=str(end),
        first_session=str(nav["date"].min()),
        last_session=str(nav["date"].max()),
        trading_days=nav.height,
        factor_name="BIAS",
        factor_class="example.bias_factor.BiasFactor",
        data_engine="polars",
        lookback=BIAS_LOOKBACK,
        factor_formula=f"close / SMA(close, {BIAS_LOOKBACK}) - 1",
        min_listing_sessions=60,
        commission_rate=args.commission_rate,
        sell_tax_rate=args.sell_tax_rate,
        slippage_rate=0.0,
        initial_value=args.initial_value,
        engine="factor_test.analyzer.portfolio_returns -> backtest_portfolio",
        factor_direction=-1,
        weight_method="quantile",
        strategies=[
            dict(
                portfolio="Momentum_LS",
                long_short=True,
                target_long_weight=0.5,
                target_short_weight=-0.5,
            ),
            dict(
                portfolio="Momentum_LongOnly",
                long_short=False,
                target_long_weight=1.0,
                target_short_weight=0.0,
            ),
        ],
        quantile_pct=args.quantile_pct,
        short_borrow_rate=args.short_borrow_rate,
        valuation="open-to-open",
        rebalance_every="monthly",
        entry_lag=1,
        calendar="CSI500 observed trading dates",
        signal_months=coverage.height,
        turnover_definition="sum(abs(trade_notional)) / previous_equity / 2",
        generated_at_utc=datetime.now(UTC).isoformat(),
        price_adjustment="hfq",
        benchmark="000905 price index",
        requested_stocks=stocks.height,
        downloaded_stocks=int(quality["downloaded"].sum()),
        price_sources=dict(
            quality.group_by("source").len().select("source", "len").iter_rows()
        ),
        price_rows=int(quality["rows"].sum()),
    )
    import akshare as ak

    metadata["akshare_version"] = ak.__version__
    metadata["universe_sources"] = json.loads(
        (args.cache / "universe_sources.json").read_text(encoding="utf-8")
    )
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from example.momentum_report import write_polars_report

    write_polars_report(
        args.output, nav, metrics, annual, ic, coverage, quality, daily, metadata
    )
    print(
        metrics.select("portfolio", "total_return", "annual_return", "mean_turnover"),
        flush=True,
    )
    print(f'Report: {(args.output / "report.md").resolve()}', flush=True)


if __name__ == "__main__":
    main()
