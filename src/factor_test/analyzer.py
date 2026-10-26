import polars as pl
import polars.selectors as cs
from scipy.stats import t as student_t

from ._utils import (
    calendar_for as _calendar_for,
    validate_horizons as _validate_horizons,
    validate_keys as _validate_keys,
)
from .diagnostics import (
    factor_rank_autocorrelation,
    mean_return_by_quantile,
    quantile_monotonicity,
    quantile_return_summary,
    quantile_spread,
    quantile_turnover,
    quantize_factor,
    spread_return_summary,
)
from .portfolio import (
    BacktestResult,
    backtest_portfolio,
    portfolio_metrics,
)

__all__ = [
    "compute_returns",
    "daily_information_coefficient",
    "information_coefficient",
    "factor_weights",
    "portfolio_returns",
    "quantize_factor",
    "mean_return_by_quantile",
    "quantile_return_summary",
    "quantile_spread",
    "spread_return_summary",
    "quantile_monotonicity",
    "quantile_turnover",
    "factor_rank_autocorrelation",
    "backtest_portfolio",
    "portfolio_metrics",
    "BacktestResult",
]


def compute_returns(
    factor_data: pl.LazyFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    price_col: str = "open",
    entry_lag: int = 1,
    *,
    prices: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """按行情已有日期计算 P[t + entry_lag + h] / P[t + entry_lag] - 1。

    prices 省略时使用输入表行情。个股缺价不跳日，整体缺失日期不补充。
    返回 entry_date、exit_date_{h} 和 ret_{h}，报价无效时收益为 null。
    """

    _validate_horizons(horizons)
    if type(entry_lag) is not int or entry_lag < 0:
        raise ValueError("entry_lag 必须是非负整数")
    _validate_keys(factor_data, ["date", "symbol"])
    price_data = factor_data if prices is None else prices
    if prices is not None:
        _validate_keys(price_data, ["date", "symbol"])
    dates = _calendar_for(price_data)
    unknown_dates = (
        factor_data.select("date")
        .unique()
        .join(dates, on="date", how="anti")
        .limit(1)
        .collect()
    )
    if not unknown_dates.is_empty():
        raise ValueError("信号日期必须包含在行情 date 中")

    price = pl.col(price_col).cast(pl.Float64)
    quotes = price_data.select(
        "date",
        "symbol",
        pl.when(price.is_finite() & (price > 0)).then(price).alias("_price"),
    )
    mapping = dates.with_columns(
        pl.col("date").shift(-entry_lag).alias("entry_date"),
        *[
            pl.col("date").shift(-(entry_lag + horizon)).alias(f"exit_date_{horizon}")
            for horizon in horizons
        ],
    )
    generated_columns = ["entry_date"] + [
        name
        for horizon in horizons
        for name in (f"ret_{horizon}", f"exit_date_{horizon}")
    ]
    data = (
        factor_data.drop(generated_columns, strict=False)
        .join(mapping, on="date", how="left")
        .join(
            quotes.rename({"date": "entry_date", "_price": "_entry_price"}),
            on=["entry_date", "symbol"],
            how="left",
        )
    )
    for horizon in horizons:
        data = (
            data.join(
                quotes.rename(
                    {"date": f"exit_date_{horizon}", "_price": "_exit_price"}
                ),
                on=[f"exit_date_{horizon}", "symbol"],
                how="left",
            )
            .with_columns(
                (pl.col("_exit_price") / pl.col("_entry_price") - 1).alias(
                    f"ret_{horizon}"
                )
            )
            .drop("_exit_price")
        )
    return data.drop("_entry_price").sort(["date", "symbol"])


def daily_information_coefficient(
    factor_data: pl.LazyFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    *,
    min_assets: int = 3,
) -> pl.LazyFrame:
    """
    计算每日横截面 Pearson IC 和 Spearman Rank IC。

    参数
    ----
    factor_data: 包含 date, symbol, factor, ret_{h} 的 LazyFrame。
    horizons: 未来收益持有周期，例如 (1, 5, 10, 20)。

    返回
    ----
    date, horizon, n_total, n_assets, coverage, ic, rank_ic。
    n_assets 为有效因子-收益对数量；低于 min_assets 或常数截面时相关性为 null。
    """

    _validate_horizons(horizons)
    if type(min_assets) is not int or min_assets < 2:
        raise ValueError("min_assets 必须为不小于 2 的整数")
    _validate_keys(factor_data, ["date", "symbol"])

    results = []
    for h in horizons:
        valid = (
            pl.col("factor").is_finite() & pl.col(f"ret_{h}").is_finite()
        ).fill_null(False)
        factor = pl.col("factor").filter(valid)
        returns = pl.col(f"ret_{h}").filter(valid)
        valid_count = valid.sum()
        has_enough_assets = valid_count >= min_assets
        results.append(
            factor_data.group_by("date")
            .agg(
                pl.len().alias("n_total"),
                valid_count.alias("n_assets"),
                pl.when(has_enough_assets)
                .then(pl.corr(factor, returns, method="pearson").fill_nan(None))
                .alias("ic"),
                pl.when(has_enough_assets)
                .then(pl.corr(factor, returns, method="spearman").fill_nan(None))
                .alias("rank_ic"),
            )
            .with_columns(
                pl.lit(h, dtype=pl.Int64).alias("horizon"),
                (pl.col("n_assets") / pl.col("n_total")).alias("coverage"),
            )
        )
    return (
        pl.concat(results)
        .select("date", "horizon", "n_total", "n_assets", "coverage", "ic", "rank_ic")
        .sort(["date", "horizon"])
    )


def _two_sided_t_p_value(row) -> float | None:
    """使用 Student t 分布的 survival function 计算双侧 p 值。"""
    t_stat = row["t_stat"]
    if t_stat is None:
        return None
    return float(2 * student_t.sf(abs(t_stat), df=row["n"] - 1))


def information_coefficient(
    ic_data: pl.LazyFrame,
    periods_per_year: int = 252,
) -> pl.LazyFrame:
    """
    统计每日 IC / Rank IC 的时间序列特征。

    统计指标:
        n, mean, std, t, p,
        ic_skew, ic_kurtosis, ann_ir

    t、p 为普通单样本 t 检验；ann_ir 为未调整自相关的描述指标。
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year 必须大于 0")
    _validate_keys(ic_data, ["date", "horizon"])

    # IC 和 Rank IC 转成长表
    data = ic_data.unpivot(
        on=["ic", "rank_ic"],
        index=["date", "horizon"],
        variable_name="metric",
        value_name="value",
    ).sort("date")

    # 基础描述统计
    stats = data.group_by(["horizon", "metric"]).agg(
        pl.col("value").count().alias("n"),
        pl.len().alias("n_dates"),
        pl.col("value").mean().alias("ic_mean"),
        pl.col("value").std(ddof=1).alias("ic_std"),
        pl.col("value").skew(bias=False).alias("ic_skew"),
        pl.col("value").kurtosis(fisher=True, bias=False).alias("ic_kurtosis"),
    )

    # t 值、年化 ICIR
    has_variation = (pl.col("n") >= 2) & (pl.col("ic_std") > 0)
    stats = stats.with_columns(
        pl.when(has_variation)
        .then(pl.col("ic_mean") / (pl.col("ic_std") / pl.col("n").sqrt()))
        .otherwise(None)
        .alias("t_stat"),
        pl.when(has_variation)
        .then(pl.col("ic_mean") / pl.col("ic_std") * (periods_per_year**0.5))
        .otherwise(None)
        .alias("ann_ir"),
        pl.when(pl.col("n") >= 3)
        .then(pl.col("ic_skew"))
        .otherwise(None)
        .alias("ic_skew"),
        pl.when(pl.col("n") >= 4)
        .then(pl.col("ic_kurtosis"))
        .otherwise(None)
        .alias("ic_kurtosis"),
    )

    # 双侧 p 值
    stats = stats.with_columns(
        pl.struct(["t_stat", "n"])
        .map_elements(
            _two_sided_t_p_value,
            return_dtype=pl.Float64,
        )
        .alias("p_value")
    )

    return (
        stats.with_columns((pl.col("n") / pl.col("n_dates")).alias("coverage_dates"))
        .select(
            "horizon",
            "metric",
            "n",
            "n_dates",
            "coverage_dates",
            "ic_mean",
            "ic_std",
            "t_stat",
            "p_value",
            "ic_skew",
            "ic_kurtosis",
            "ann_ir",
        )
        .sort(["horizon", "metric"])
    )


def factor_weights(
    factor_data: pl.LazyFrame,
    weight_method: str = "factor",
    long_short: bool = True,
    quantile_pct: float = 0.2,
) -> pl.LazyFrame:
    """
    根据因子暴露计算组合目标权重。

    输入列:
        date, symbol, factor

    参数:
        weight_method:
            factor   : 根据因子强度加权
            equal    : 中位数划分多空，等权
            quantile : 按分位数选择股票，等权

        long_short:
            True  : 多空组合
            False : 纯多头组合

        quantile_pct:
            quantile 模式下单侧选股比例。
            例如 0.2 表示 Top/Bottom 20%。

    返回:
        保留有效因子行，新增 weight 列；不会使用未来收益筛选股票。
    """
    if weight_method not in ("factor", "equal", "quantile"):
        raise ValueError(f"不支持的 weight_method: {weight_method}")

    if not 0 < quantile_pct <= 0.5:
        raise ValueError("quantile_pct 必须在 (0, 0.5] 之间")

    # 只使用有效因子计算权重
    # 并列因子使用平均排名，不按代码制造排序信号。
    data = (
        factor_data.filter(
            pl.col("factor").cast(pl.Float64).is_finite().fill_null(False)
        )
        .sort(["date", "symbol"])
        .with_columns(pl.col("factor").cast(pl.Float64).alias("_factor"))
    )

    factor = pl.col("_factor")

    # 生成原始仓位信号
    if weight_method == "factor":
        if long_short:
            # 因子减均值：正值做多，负值做空
            signal = factor - factor.mean().over("date")
        else:
            # 纯多头：因子减最小值，确保非负
            low = factor.min().over("date")
            high = factor.max().over("date")

            signal = pl.when(high > low).then(factor - low).otherwise(1.0)

    elif weight_method == "equal":
        if long_short:
            # 中位数划分多空
            centered = factor - factor.median().over("date")

            signal = (
                pl.when(centered > 0)
                .then(1.0)
                .when(centered < 0)
                .then(-1.0)
                .otherwise(0.0)
            )
        else:
            # 所有有效股票等权做多
            signal = pl.lit(1.0)

    else:  # quantile
        data = data.with_columns(
            pl.len().over("date").alias("_n"),
            factor.n_unique().over("date").alias("_n_unique"),
            factor.rank(method="average").over("date").alias("_rank"),
        )

        # 每侧选择 floor(n * quantile_pct) 只
        # 至少选一只
        data = data.with_columns(
            (
                pl.col("_n")
                .cast(pl.Float64)
                .mul(quantile_pct)
                .floor()
                .cast(pl.UInt32)
                .clip(lower_bound=1)
            ).alias("_k")
        )

        is_top = pl.col("_rank") > pl.col("_n") - pl.col("_k")
        is_bottom = pl.col("_rank") <= pl.col("_k")

        if long_short:
            signal = (
                pl.when(pl.col("_n_unique") < 2)
                .then(0.0)
                .when(is_top)
                .then(1.0)
                .when(is_bottom)
                .then(-1.0)
                .otherwise(0.0)
            )
        else:
            signal = (
                pl.when(is_top & (pl.col("_n_unique") >= 2)).then(1.0).otherwise(0.0)
            )

    data = data.with_columns(signal.alias("_signal"))

    # 按方向归一化
    signal = pl.col("_signal")

    if long_short:
        # 多头和空头各自的原始绝对信号合计
        long_total = pl.when(signal > 0).then(signal).otherwise(0.0).sum().over("date")
        short_total = (
            pl.when(signal < 0).then(-signal).otherwise(0.0).sum().over("date")
        )

        # 两侧都存在时，分配 +50% / -50%
        weight = (
            pl.when((long_total > 0) & (short_total > 0))
            .then(
                pl.when(signal > 0)
                .then(0.5 * signal / long_total)
                .otherwise(0.5 * signal / short_total)
            )
            .otherwise(0.0)
        )
    else:
        total = signal.sum().over("date")
        weight = pl.when(total > 0).then(signal / total).otherwise(0.0)

    return data.with_columns(weight.alias("weight")).drop(cs.starts_with("_"))


def portfolio_returns(
    factor_data: pl.LazyFrame,
    prices: pl.LazyFrame | None = None,
    *,
    rebalance_every: int | str = 5,
    weight_method: str = "factor",
    long_short: bool = True,
    quantile_pct: float = 0.2,
    **backtest_options,
) -> BacktestResult:
    """每日与多日调仓的统一账户接口，使用不重叠持有区间。

    rebalance_every=5（默认）每五个交易日调仓，=1 每日调仓。
    ="monthly" 使用每月最后交易日信号，持有至下一月度执行日。
    在首个信号日及其后每隔 rebalance_every 个交易日观察信号，
    按 entry_lag（默认 1）延迟建仓；固定股数持有至下一调仓日。
    prices 省略时使用 factor_data 内的行情；稀疏信号需单独提供完整行情。
    返回 result.daily（每日收益、现金与净值）、positions、trades、unfilled_orders。
    不需要预先计算远期收益，也不会将重叠 ret_h 当作账户日收益累计。
    账户模拟需要按时间顺序执行，BacktestResult 内为已物化的 DataFrame。
    """
    _validate_keys(factor_data, ["date", "symbol"])
    weights = factor_weights(factor_data, weight_method, long_short, quantile_pct)
    # 无有效因子的行使用零权重，保留输入 date 对应的调仓起点。
    weights = (
        factor_data.select("date", "symbol")
        .join(
            weights.select("date", "symbol", "weight"),
            on=["date", "symbol"],
            how="left",
        )
        .with_columns(pl.col("weight").fill_null(0.0))
    )
    return backtest_portfolio(
        weights,
        factor_data if prices is None else prices,
        rebalance_every=rebalance_every,
        **backtest_options,
    )


