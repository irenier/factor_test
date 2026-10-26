"""Factor quantiles, return diagnostics and signal persistence."""

import polars as pl

from ._utils import calendar_for, validate_horizons, validate_keys


def _validate_quantiles(quantiles: int) -> None:
    if type(quantiles) is not int or quantiles < 2:
        raise ValueError("quantiles 必须为不小于 2 的整数")


def quantize_factor(factor_data: pl.LazyFrame, quantiles: int = 5) -> pl.LazyFrame:
    """每日平均排名分箱；并列值留在同组，允许空组。

    无效因子或全截面相同值的标签为 null；保留输入行。
    不用股票代码打破并列，也不利用未来收益选组。
    """
    _validate_quantiles(quantiles)
    validate_keys(factor_data, ["date", "symbol"])
    valid = pl.col("factor").cast(pl.Float64).is_finite().fill_null(False)
    values = factor_data.filter(valid).select("date", "symbol", "factor")
    labels = values.with_columns(
        pl.col("factor").rank(method="average").over("date").alias("_rank"),
        pl.len().over("date").alias("_n"),
        pl.col("factor").n_unique().over("date").alias("_unique"),
    ).select(
        "date",
        "symbol",
        pl.when(pl.col("_unique") > 1)
        .then(((pl.col("_rank") - 1) * quantiles / pl.col("_n")).floor() + 1)
        .cast(pl.Int64)
        .alias("factor_quantile"),
    )
    return (
        factor_data.drop("factor_quantile", strict=False)
        .join(labels, on=["date", "symbol"], how="left", validate="1:1")
        .sort(["date", "symbol"])
    )


def mean_return_by_quantile(
    factor_data: pl.LazyFrame,
    quantiles: int = 5,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    *,
    demeaned: bool = False,
) -> pl.LazyFrame:
    """每日分组均值、横截面标准误和样本覆盖率。

    仅按有效因子-收益对统计；分组在观察未来收益之前确定。
    std_error 是横截面描述统计，不是时间序列显著性检验。
    空组和全部缺失收益的组为 null 收益，绝不填零。
    """
    validate_horizons(horizons)
    data = quantize_factor(factor_data, quantiles)
    grid = (
        data.select("date")
        .unique()
        .join(
            pl.DataFrame(
                {"factor_quantile": range(1, quantiles + 1)},
                schema={"factor_quantile": pl.Int64},
            ).lazy(),
            how="cross",
        )
    )
    assigned_factors = data.filter(pl.col("factor_quantile").is_not_null())
    results = []
    for horizon in horizons:
        forward_return = pl.col(f"ret_{horizon}")
        valid = forward_return.is_finite().fill_null(False)
        sample = assigned_factors.with_columns(
            pl.when(valid).then(forward_return).alias("_ret")
        )
        if demeaned:
            sample = sample.with_columns(
                (pl.col("_ret") - pl.col("_ret").mean().over("date")).alias("_ret")
            )
        stats = sample.group_by(["date", "factor_quantile"]).agg(
            pl.len().alias("n_total"),
            pl.col("_ret").count().alias("n_assets"),
            pl.col("_ret").mean().alias("mean_return"),
            pl.col("_ret").std(ddof=1).alias("std_return"),
        )
        daily_stats = (
            grid.join(stats, on=["date", "factor_quantile"], how="left")
            .with_columns(
                pl.lit(horizon, dtype=pl.Int64).alias("horizon"),
                pl.col("n_total").fill_null(0),
                pl.col("n_assets").fill_null(0),
            )
            .with_columns(
                pl.when(pl.col("n_total") > 0)
                .then(pl.col("n_assets") / pl.col("n_total"))
                .otherwise(0.0)
                .alias("coverage"),
                (pl.col("std_return") / pl.col("n_assets").sqrt()).alias("std_error"),
            )
        )
        results.append(daily_stats)
    return pl.concat(results).sort(["date", "horizon", "factor_quantile"])


def quantile_return_summary(
    quantile_returns: pl.LazyFrame,
) -> pl.LazyFrame:
    """按日期等权汇总分组收益、时间标准差和平均覆盖率。

    mean_return 仍为对应 horizon 的持有期收益，不是日化收益。
    """
    return (
        quantile_returns.group_by(["horizon", "factor_quantile"])
        .agg(
            pl.col("mean_return").count().alias("n_dates"),
            pl.col("mean_return").mean().alias("mean_return"),
            pl.col("mean_return").std(ddof=1).alias("std_return"),
            pl.col("coverage").mean().alias("mean_coverage"),
        )
        .sort(["horizon", "factor_quantile"])
    )


def quantile_spread(
    quantile_returns: pl.LazyFrame,
    upper_quantile: int = 5,
    lower_quantile: int = 1,
) -> pl.LazyFrame:
    """首尾收益差；spread 为多空各 50%（总绝对权重 1）的收益。

    std_error 假设两组横截面均值误差独立，不代表时间序列显著性。
    """
    if lower_quantile >= upper_quantile:
        raise ValueError("upper_quantile 必须大于 lower_quantile")
    top = quantile_returns.filter(pl.col("factor_quantile") == upper_quantile).select(
        "date",
        "horizon",
        pl.col("mean_return").alias("upper_return"),
        pl.col("std_error").alias("upper_std_error"),
    )
    bottom = quantile_returns.filter(
        pl.col("factor_quantile") == lower_quantile
    ).select(
        "date",
        "horizon",
        pl.col("mean_return").alias("lower_return"),
        pl.col("std_error").alias("lower_std_error"),
    )
    return_spread = pl.col("upper_return") - pl.col("lower_return")
    return (
        top.join(bottom, on=["date", "horizon"], how="full", coalesce=True)
        .with_columns(
            return_spread.alias("raw_spread"),
            (return_spread / 2).alias("spread"),
            (
                (pl.col("upper_std_error") ** 2 + pl.col("lower_std_error") ** 2).sqrt()
                / 2
            ).alias("std_error"),
        )
        .sort(["date", "horizon"])
    )


def quantile_monotonicity(quantile_returns: pl.LazyFrame) -> pl.LazyFrame:
    """每日分组编号与分组均值的 Spearman 相关，至少需要三个有效组。"""
    valid = pl.col("mean_return").is_finite().fill_null(False)
    return (
        quantile_returns.group_by(["date", "horizon"])
        .agg(
            valid.sum().alias("n_quantiles"),
            pl.when(valid.sum() >= 3)
            .then(
                pl.corr(
                    pl.col("factor_quantile").filter(valid),
                    pl.col("mean_return").filter(valid),
                    method="spearman",
                ).fill_nan(None)
            )
            .alias("monotonicity"),
        )
        .sort(["date", "horizon"])
    )


def spread_return_summary(
    spreads: pl.LazyFrame,
) -> pl.LazyFrame:
    """首尾组多空收益的时间均值、标准差和覆盖率。"""
    series = spreads.select(
        "date",
        "horizon",
        pl.lit(0, dtype=pl.Int64).alias("factor_quantile"),
        pl.col("spread").alias("mean_return"),
        pl.col("spread")
        .is_finite()
        .fill_null(False)
        .cast(pl.Float64)
        .alias("coverage"),
    )
    return (
        quantile_return_summary(series)
        .drop("factor_quantile")
        .rename({"mean_return": "mean_spread", "std_return": "std_spread"})
    )


def quantile_turnover(
    factor_data: pl.LazyFrame,
    quantiles: int = 5,
    period: int = 1,
) -> pl.LazyFrame:
    """各分位组新进入股票数 / 当前组股票数，按输入中已有日期滞后比较。

    首期、空组、当前或比较日无可分组信号时为 null；这不是交易金额换手。
    """
    if type(period) is not int or period < 1:
        raise ValueError("period 必须为正整数")
    data = quantize_factor(factor_data, quantiles)
    dates = (
        calendar_for(data)
        .with_row_index("_idx")
        .with_columns(pl.col("_idx").cast(pl.Int64))
    )
    groups = pl.DataFrame(
        {"factor_quantile": range(1, quantiles + 1)},
        schema={"factor_quantile": pl.Int64},
    ).lazy()
    names = (
        data.filter(pl.col("factor_quantile").is_not_null())
        .group_by(["date", "factor_quantile"])
        .agg(pl.col("symbol").alias("_names"))
    )
    observed = data.group_by("date").agg(
        pl.col("factor_quantile").is_not_null().any().alias("_observed")
    )
    symbol_type = data.collect_schema()["symbol"]
    panel = (
        dates.join(groups, how="cross")
        .join(names, on=["date", "factor_quantile"], how="left")
        .join(observed, on="date", how="left")
        .with_columns(
            pl.col("_names").fill_null(pl.lit([], dtype=pl.List(symbol_type))),
            pl.col("_observed").fill_null(False),
        )
    )
    previous = panel.select(
        (pl.col("_idx") + period).alias("_idx"),
        "factor_quantile",
        pl.col("_names").alias("_previous"),
        pl.col("_observed").alias("_previous_observed"),
    )
    return (
        panel.join(previous, on=["_idx", "factor_quantile"], how="left")
        .with_columns(
            pl.col("_names").list.len().alias("n_assets"),
            pl.when(pl.col("_observed") & pl.col("_previous_observed"))
            .then(pl.col("_names").list.set_difference(pl.col("_previous")).list.len())
            .alias("n_new"),
        )
        .with_columns(
            pl.when(pl.col("n_assets") > 0)
            .then(pl.col("n_new") / pl.col("n_assets"))
            .alias("turnover"),
        )
        .select("date", "factor_quantile", "n_assets", "n_new", "turnover")
        .sort(["date", "factor_quantile"])
    )


def factor_rank_autocorrelation(
    factor_data: pl.LazyFrame,
    period: int = 1,
    *,
    min_assets: int = 3,
) -> pl.LazyFrame:
    """两日共同股票的截面平均排名相关，按输入中已有日期比较。"""
    if (
        type(period) is not int
        or period < 1
        or type(min_assets) is not int
        or min_assets < 2
    ):
        raise ValueError("period 必须为正整数，min_assets 必须为不小于 2 的整数")
    validate_keys(factor_data, ["date", "symbol"])
    dates = (
        calendar_for(factor_data)
        .with_row_index("_idx")
        .with_columns(pl.col("_idx").cast(pl.Int64))
    )
    ranks = (
        factor_data.filter(pl.col("factor").is_finite().fill_null(False))
        .select(
            "date",
            "symbol",
            pl.col("factor").rank(method="average").over("date").alias("_rank"),
        )
        .join(dates, on="date", how="left")
    )
    previous = ranks.select(
        (pl.col("_idx") + period).alias("_idx"),
        "symbol",
        pl.col("_rank").alias("_previous"),
    )
    pairs = (
        ranks.join(previous, on=["_idx", "symbol"], how="inner")
        .group_by("date")
        .agg(
            pl.len().alias("n_assets"),
            pl.when(pl.len() >= min_assets)
            .then(pl.corr("_rank", "_previous").fill_nan(None))
            .alias("rank_autocorrelation"),
        )
    )
    return (
        dates.select("date")
        .join(pairs, on="date", how="left")
        .with_columns(pl.col("n_assets").fill_null(0))
        .sort("date")
    )
