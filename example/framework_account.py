"""Compare factor quantile long/short and long-only accounts in src/factor_test."""

from dataclasses import replace

import polars as pl

from factor_test.analyzer import portfolio_returns
from factor_test.portfolio import BacktestResult, portfolio_metrics


def run_accounts(
    signals,
    prices,
    benchmark_nav,
    *,
    start,
    initial_value=100_000_000.0,
    commission_rate=0.0003,
    sell_tax_rate=0.001,
    quantile_pct=0.2,
    short_borrow_rate=0.0,
    factor_direction=-1,
    portfolio_prefix="Momentum",
):
    """Use the framework's integer units, costs, frozen positions and daily NAV.

    Prices must include the last session before start so that its month-end
    signal can execute on the first session of the requested backtest period.
    The shared engine values/trades at open; benchmark_nav must use the same
    opening-price endpoints. No example-specific account simulation is used.
    All inputs/results are Polars frames; benchmark_nav has date/benchmark_nav.
    factor_direction defaults to the existing negative-factor convention;
    use +1 for a high-factor long leg and -1 for a low-factor long leg.
    """
    if factor_direction not in (-1, 1):
        raise ValueError("factor_direction must be -1 or +1")
    benchmark_nav = benchmark_nav.sort("date").with_columns(
        (
            pl.col("benchmark_nav") / pl.col("benchmark_nav").shift(1).fill_null(1.0)
            - 1
        ).alias("benchmark_return")
    )
    benchmark = benchmark_nav.select("date", "benchmark_return").lazy()
    factor_data = signals.with_columns(
        (factor_direction * pl.col("factor")).alias("factor")
    ).lazy()
    metric_rows, daily, trades, unfilled, rebalances = [], [], [], [], []
    for label, long_short in (
        (f"{portfolio_prefix}_LS", True),
        (f"{portfolio_prefix}_LongOnly", False),
    ):
        print(f"Framework account: {label}, signal rows={signals.height:,}", flush=True)
        result = portfolio_returns(
            factor_data,
            prices,
            rebalance_every="monthly",
            entry_lag=1,
            weight_method="quantile",
            long_short=long_short,
            quantile_pct=quantile_pct,
            initial_value=initial_value,
            commission_rate=commission_rate,
            sell_tax_rate=sell_tax_rate,
            slippage_rate=0.0,
            short_borrow_rate=short_borrow_rate if long_short else 0.0,
            suspended_col="suspended",
            missing_price="carry",
            record_positions=False,
        )
        result = replace(result, daily=result.daily.filter(pl.col("date") >= start))
        stats = portfolio_metrics(result, benchmark=benchmark).row(0, named=True)
        metric_rows.append(dict(stats, portfolio=label))
        for table, destination in (
            (result.daily, daily),
            (result.trades, trades),
            (result.unfilled_orders, unfilled),
        ):
            destination.append(table.with_columns(pl.lit(label).alias("portfolio")))
        rebalances.append(
            result.daily.filter(
                (pl.col("traded_notional") > 0) | (pl.col("unfilled_orders") > 0)
            ).with_columns(pl.lit(label).alias("portfolio"))
        )
    # The index is a frictionless reference, measured at matching open endpoints.
    benchmark_daily = benchmark_nav.select(
        "date",
        pl.col("benchmark_nav").alias("nav"),
        pl.col("benchmark_return").alias("net_return"),
        (
            pl.col("benchmark_nav")
            / pl.col("benchmark_nav").cum_max().clip(lower_bound=1.0)
            - 1
        ).alias("drawdown"),
        pl.lit(0.0).alias("cost"),
        pl.lit(0.0).alias("turnover"),
        pl.lit(0.0).alias("traded_notional"),
    )
    benchmark_result = BacktestResult(
        benchmark_daily, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), 1.0
    )
    stats = portfolio_metrics(benchmark_result, benchmark=benchmark).row(0, named=True)
    metric_rows.append(dict(stats, portfolio="CSI500"))
    return (
        pl.concat(daily),
        pl.DataFrame(metric_rows, infer_schema_length=None),
        pl.concat(trades),
        pl.concat(unfilled),
        pl.concat(rebalances),
    )
