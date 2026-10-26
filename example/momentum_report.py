"""Static figures and a Markdown research report for momentum_backtest."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from factor_test.analyzer import information_coefficient


def write_polars_report(
    output, nav, metrics, annual, ic, coverage, quality, daily, metadata
):
    """Presentation boundary: convert only small result tables for existing charts.

    Price preparation, factor values, IC and account execution stay in Polars.
    The existing Matplotlib/Markdown renderer retains Pandas display formatting.
    """

    def presentation(frame):
        result = pd.DataFrame(frame.to_dict(as_series=False))
        if "date" in result:
            result["date"] = pd.to_datetime(result["date"])
        return result

    write_report(
        output,
        presentation(nav).set_index("date"),
        presentation(metrics),
        presentation(annual).set_index("year"),
        presentation(ic),
        presentation(coverage),
        presentation(quality),
        presentation(daily),
        metadata,
    )


def summarize_ic(ic):
    """Adapt monthly ICs to the existing src summary; do not recompute its statistics."""
    summaries = []
    for direction, sign in (("raw", 1), ("negative", -1)):
        data = pl.DataFrame(
            {
                "date": pd.to_datetime(ic["signal_date"]).dt.date.to_list(),
                "horizon": [1]
                * len(ic),  # One following holding month, not one trading day.
                "ic": (ic["ic"] * sign).to_list(),
                "rank_ic": (ic["rank_ic"] * sign).to_list(),
            },
            schema={
                "date": pl.Date,
                "horizon": pl.Int64,
                "ic": pl.Float64,
                "rank_ic": pl.Float64,
            },
        )
        data = data.with_columns(
            pl.when(pl.col("ic", "rank_ic").is_finite())
            .then(pl.col("ic", "rank_ic"))
            .otherwise(None)
        )
        stats = information_coefficient(data.lazy(), periods_per_year=12).collect()
        summary = pd.DataFrame(stats.to_dict(as_series=False))
        # These two display fields supplement the statistics supplied by src.
        summary["positive_ratio"] = [
            data.select((pl.col(metric) > 0).sum()).item() / count if count else np.nan
            for metric, count in zip(summary["metric"], summary["n"])
        ]
        summary["icir"] = summary["ann_ir"] / np.sqrt(12)
        summaries.append(summary.assign(direction=direction, periods_per_year=12))
    return pd.concat(summaries, ignore_index=True)


def summarize_turnover(daily):
    """Aggregate src's one-way turnover: absolute traded value / prior equity / 2."""
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = data["date"].dt.to_period("M").astype(str)
    data["year"] = data["date"].dt.year
    data["has_trade"] = data["traded_notional"] > 0
    monthly = (
        data.groupby(["portfolio", "month"], sort=True)
        .agg(
            turnover=("turnover", "sum"),
            traded_notional=("traded_notional", "sum"),
            trading_days=("date", "size"),
            trade_days=("has_trade", "sum"),
        )
        .reset_index()
    )
    annual = (
        data.groupby(["portfolio", "year"], sort=True)
        .agg(
            turnover=("turnover", "sum"),
            traded_notional=("traded_notional", "sum"),
            trading_days=("date", "size"),
            trade_days=("has_trade", "sum"),
        )
        .reset_index()
    )
    summary = data.groupby("portfolio", sort=False).agg(
        trading_days=("date", "size"),
        trade_days=("has_trade", "sum"),
        mean_daily_turnover=("turnover", "mean"),
        total_turnover=("turnover", "sum"),
    )
    summary["mean_trade_day_turnover"] = (
        data.loc[data["has_trade"]].groupby("portfolio")["turnover"].mean()
    )
    summary["mean_trade_day_turnover"] = summary["mean_trade_day_turnover"].fillna(0.0)
    summary["mean_monthly_turnover"] = monthly.groupby("portfolio")["turnover"].mean()
    summary["annualized_turnover"] = summary["mean_daily_turnover"] * 252
    return summary.reset_index(), monthly, annual


def write_report(output, nav, metrics, annual, ic, coverage, quality, daily, metadata):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
        }
    )
    colors = {
        "Momentum_LS": "#155e75",
        "Momentum_LongOnly": "#39916a",
        "CSI500": "#d97706",
    }
    names = {
        "Momentum_LS": "负向乖离率多空",
        "Momentum_LongOnly": f"低乖离率{metadata['quantile_pct']:.0%}纯多头",
        "CSI500": "中证500",
    }
    labels = {
        label: name + ("价格指数" if label == "CSI500" else "")
        for label, name in names.items()
    }
    strategies = ["Momentum_LS", "Momentum_LongOnly"]
    comparison = strategies + ["CSI500"]
    figures = []
    turnover_summary, monthly_turnover, annual_turnover = summarize_turnover(daily)
    for filename, table in (
        ("turnover_summary", turnover_summary),
        ("monthly_turnover", monthly_turnover),
        ("annual_turnover", annual_turnover),
    ):
        table.to_csv(
            output / f"{filename}.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.10g",
        )

    def save(fig, name, title):
        fig.savefig(
            output / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)
        figures.append((name, title))

    def line(ax, series, label, color, **kwargs):
        ax.plot(series.index, series, label=label, color=color, **kwargs)

    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    for key in comparison:
        line(ax, nav[key], labels[key], colors[key], lw=1.7)
    ax.set(
        title=f"{metadata['lookback']}日负向乖离率：多空、纯多头与中证500累计净值",
        ylabel="净值（初始资金 = 1）",
    )
    ax.legend(loc="upper left")
    save(fig, "01_nav", "累计净值")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.1), sharex=True, layout="constrained")
    for key in strategies:
        line(axes[0], nav[key] / nav["CSI500"] - 1, labels[key], colors[key])
    axes[0].axhline(0, color="#64748b", lw=0.7)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("相对中证500的复合超额收益（组合净值 / 基准净值 − 1）")
    axes[0].legend(loc="upper left")
    for key in comparison:
        values = nav[key].to_numpy()
        drawdown = values / np.maximum.accumulate(np.r_[1.0, values])[1:] - 1
        line(axes[1], pd.Series(drawdown, index=nav.index), labels[key], colors[key])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("从初始资金开始计量的每日开盘估值回撤")
    axes[1].legend(loc="lower left")
    save(fig, "02_excess_drawdown", "超额收益与回撤")

    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    x = np.arange(len(annual))
    for offset, key in zip((-0.25, 0, 0.25), comparison):
        ax.bar(
            x + offset,
            annual[key],
            width=0.24,
            label=labels.get(key, "年度收益差（百分点）"),
            color=colors.get(key, "#85945b"),
        )
    ax.set_xticks(x, annual.index.astype(str))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("分年度收益与中证500对比")
    ax.legend(ncol=3)
    save(fig, "03_annual", "年度收益")

    if not ic.empty:
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 6.3), sharex=True, layout="constrained"
        )
        series = ic.set_index("signal_date")["direction_adjusted_rank_ic"]
        axes[0].bar(
            series.index,
            series,
            width=22,
            color=np.where(series >= 0, "#1c7b85", "#b97568"),
        )
        line(
            axes[0],
            series.rolling(12, min_periods=6).mean(),
            "12个月滚动均值",
            "#d97706",
            lw=2,
        )
        axes[0].set_title("月度 Rank IC：负向调整后信号（−BIAS）与下一持有月收益")
        axes[0].legend()
        line(axes[1], series.cumsum(), "Rank IC累加（诊断指标）", colors["Momentum_LS"])
        axes[1].set_title("累计 Rank IC")
        save(fig, "04_rank_ic", "因子预测力")

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    line(
        ax,
        coverage.set_index("date")["factor_stocks"],
        "乖离率因子可用股票数",
        colors["Momentum_LS"],
    )
    line(
        ax,
        coverage.set_index("date")["listed"],
        "股票清单内当时已上市数",
        "#a3adb8",
        ls="--",
    )
    ax.set(title="逐月股票池覆盖", ylabel="股票数量")
    ax.legend(loc="upper left")
    save(fig, "05_coverage", "样本覆盖")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.1), layout="constrained")
    for key in strategies:
        monthly_values = monthly_turnover.loc[monthly_turnover["portfolio"] == key]
        line(
            axes[0],
            pd.Series(
                monthly_values["turnover"].to_numpy(),
                index=pd.to_datetime(monthly_values["month"]),
            ),
            labels[key],
            colors[key],
        )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("月度单边换手率（当月每日换手率求和）")
    axes[0].legend(loc="upper left")
    annual_pivot = annual_turnover.pivot(
        index="year", columns="portfolio", values="turnover"
    )
    years = np.arange(len(annual_pivot))
    for offset, key in zip((-0.18, 0.18), strategies):
        axes[1].bar(
            years + offset,
            annual_pivot[key],
            width=0.34,
            label=labels[key],
            color=colors[key],
        )
    axes[1].set_xticks(years, annual_pivot.index.astype(str))
    axes[1].set(title="年度单边换手倍数（当年每日换手率求和）", ylabel="倍")
    axes[1].legend(loc="upper left")
    save(fig, "06_turnover", "换手率")

    metric_labels = {
        "portfolio": "组合",
        "total_return": "累计收益",
        "annual_return": "年化收益",
        "annual_volatility": "年化波动",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "relative_return": "复合超额",
        "annual_relative_return": "年化复合超额",
        "return_difference": "累计收益差",
        "tracking_error": "跟踪误差",
        "information_ratio": "信息比率",
        "relative_max_drawdown": "相对净值最大回撤",
        "mean_turnover": "日均单边换手率",
    }
    display = metrics[list(metric_labels)].copy()
    for col in display.columns:
        if col == "portfolio":
            continue
        display[col] = display[col].map(
            lambda value: (
                "—"
                if pd.isna(value)
                else (
                    f"{value * 100:.2f}个百分点"
                    if col == "return_difference"
                    else (
                        f"{value:.3f}"
                        if col in ("sharpe", "information_ratio")
                        else f"{value:.2%}"
                    )
                )
            )
        )
    display = display.rename(columns=metric_labels)
    annual_display = annual.map(lambda x: f"{x:.2%}").reset_index(names="年份")
    quality_summary = quality.groupby(["exchange", "status"], dropna=False).agg(
        requested=("symbol", "count"),
        downloaded=("downloaded", "sum"),
        price_rows=("rows", "sum"),
    )
    account_end = (
        daily.sort_values("date").groupby("portfolio", sort=False).tail(1).copy()
    )
    account_end["cash_ratio"] = account_end["cash"] / account_end["equity"]
    account_end = account_end[
        ["portfolio", "stale_positions", "suspended_positions", "cash_ratio"]
    ]
    account_end["cash_ratio"] = account_end["cash_ratio"].map(
        lambda value: f"{value:.2%}"
    )
    account_end = account_end.rename(
        columns={
            "portfolio": "组合",
            "stale_positions": "末日陈旧报价持仓数",
            "suspended_positions": "末日显式停牌持仓数",
            "cash_ratio": "末日现金比例",
        }
    )
    account_end["组合"] = account_end["组合"].map(names)
    ending = (
        daily.sort_values("date")
        .groupby("portfolio", sort=False)
        .tail(1)
        .set_index("portfolio")
    )
    stale_note = "；".join(
        f"{names[key]}{int(ending.loc[key, 'stale_positions'])}个" for key in strategies
    )
    cost_summary = (
        daily.groupby("portfolio", sort=False)[
            ["traded_notional", "commission", "sell_tax", "borrow_cost", "cost"]
        ]
        .sum()
        .reset_index()
    )
    cost_summary["portfolio"] = cost_summary["portfolio"].map(names)
    for column in cost_summary.columns[1:]:
        cost_summary[column] = cost_summary[column].map(lambda value: f"{value:,.2f}")
    cost_summary = cost_summary.rename(
        columns={
            "portfolio": "组合",
            "traded_notional": "成交总额（元）",
            "commission": "佣金（元）",
            "sell_tax": "卖出印花税（元）",
            "borrow_cost": "借券费（元）",
            "cost": "费用合计（元）",
        }
    )
    turnover_labels = {
        "trading_days": "交易日数",
        "trade_days": "有成交日数",
        "mean_daily_turnover": "日均单边换手率",
        "mean_trade_day_turnover": "有成交日平均单边换手率",
        "mean_monthly_turnover": "月均单边换手率",
        "annualized_turnover": "年化单边换手倍数",
        "total_turnover": "累计单边换手倍数",
    }
    turnover_display = turnover_summary.set_index("portfolio")[
        list(turnover_labels)
    ].copy()
    for column in turnover_labels:
        turnover_display[column] = turnover_display[column].map(
            lambda value: (
                str(int(value))
                if column in ("trading_days", "trade_days")
                else (
                    f"{value:.2f}倍"
                    if column in ("annualized_turnover", "total_turnover")
                    else f"{value:.2%}"
                )
            )
        )
    turnover_table = turnover_display.T.rename(
        index=turnover_labels, columns=names
    ).reset_index(names="指标")
    annual_turnover_display = (
        annual_pivot.map(lambda value: f"{value:.2f}倍")
        .rename(columns=names)
        .reset_index(names="年份")
    )
    strategy_metrics = metrics.set_index("portfolio").loc[strategies]
    failed = int((~quality["downloaded"]).sum())
    ic_summary = summarize_ic(ic)
    ic_summary.to_csv(
        output / "ic_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    warnings = [
        "股票池由获取时点的沪深京上市清单及沪深退市清单合并，并按历史上市日期筛选；"
        "无法保证覆盖全部历史证券、北交所退市证券和代码变更，仍有幸存者偏差及数据可得性偏差。",
        "后复权收益包含供应商对分红等公司行动的调整；000905为价格指数，未含现金分红再投资，二者分红口径不同。",
        f"多空组合目标各50%，总敞口100%、净敞口0%；纯多头组合将100%资金等权配置到低乖离率20%股票。两种策略风险暴露不同。多空组合融券年化费率设为{metadata['short_borrow_rate']:.2%}，"
        "未模拟真实券源、融券可得性、担保品限制与追加保证金。现金余额包含卖空所得，不代表全部可提取资金。",
        "停牌或缺少有效报价时冻结持仓并沿用最近价格；未实现涨跌停、百股整手、冲击成本、历史ST限制和退市清算回收。"
        "历史行情终止后若仍持有会持续冻结估值，可能扭曲可实现回报；这是因子研究回测，不是可成交收益承诺。",
        f"乖离率为当日后复权收盘价相对含当日在内最近{metadata['lookback']}个交易日简单移动均价的偏离；"
        "上市满60个交易日且均价历史完整才可入池，不使用上市前的新三板报价补足A股历史。",
        f"佣金按买卖成交额的{metadata['commission_rate']:.2%}收取，印花税只按卖出成交额的{metadata['sell_tax_rate']:.2%}收取（含卖空开仓），滑点费率{metadata['slippage_rate']:.2%}；全样本采用固定税率，未按历史政策逐日切换。"
        "现金无利息，使用框架的整数复权价格单位（1股粒度），末日按开盘估值且不强制清仓。复权价格单位与实际未复权股数存在差异。",
    ]
    if failed:
        warnings.insert(
            0,
            f"有{failed}只股票未取得历史行情，缺失股票没有替换为模拟数据。详见 data_quality.csv 与 data/download_status.csv。",
        )
    source_failures = [item for item in metadata["universe_sources"] if not item["ok"]]
    if source_failures:
        warnings.insert(
            0,
            f"{len(source_failures)}个股票清单接口获取失败；清单覆盖不完整，详见 run_metadata.json。",
        )

    def markdown_table(frame):
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        headings = "| " + " | ".join(map(cell, frame.columns)) + " |"
        separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = [
            "| " + " | ".join(map(cell, row)) + " |"
            for row in frame.fillna("—").itertuples(index=False, name=None)
        ]
        return "\n".join([headings, separator, *rows])

    ic_labels = {
        "n_dates": "输入月份",
        "n": "有效月份",
        "coverage_dates": "有效月份覆盖率",
        "ic_mean": "均值",
        "ic_std": "标准差",
        "icir": "月度ICIR",
        "ann_ir": "年化ICIR（√12）",
        "positive_ratio": "正值比例",
        "t_stat": "t值",
        "p_value": "双侧p值",
        "ic_skew": "偏度",
        "ic_kurtosis": "超额峰度",
    }
    ic_display = ic_summary.copy()
    ic_display["series"] = [
        ("原始BIAS" if row.direction == "raw" else "负向−BIAS")
        + (" IC" if row.metric == "ic" else " Rank IC")
        for row in ic_display.itertuples()
    ]
    for column in ic_labels:

        def format_ic(value):
            if pd.isna(value) or not np.isfinite(value):
                return "—"
            if column in ("n_dates", "n"):
                return str(int(value))
            if column in ("coverage_dates", "positive_ratio"):
                return f"{value:.2%}"
            return f"{value:.4g}" if column == "p_value" else f"{value:.4f}"

        ic_display[column] = ic_display[column].map(format_ic)
    ic_table = (
        ic_display.set_index("series")[list(ic_labels)]
        .T.rename(index=ic_labels)
        .reset_index(names="指标")
    )
    ic_coverage = (
        pd.DataFrame(
            [
                {
                    "完整持有月数": len(ic),
                    "每月平均有效股票数": f"{ic['n_assets'].mean():.1f}",
                    "每月最少有效股票数": int(ic["n_assets"].min()),
                    "每月平均截面覆盖率": f"{ic['coverage'].mean():.2%}",
                    "每月最低截面覆盖率": f"{ic['coverage'].min():.2%}",
                }
            ]
        )
        if not ic.empty
        else pd.DataFrame()
    )
    metric_table = (
        display.set_index("组合").T.reset_index(names="指标").rename(columns=names)
    )
    annual_display = annual_display.rename(
        columns={
            **names,
            "LS_minus_CSI500": "多空年度收益差",
            "LS_relative": "多空年度复合超额",
            "LO_minus_CSI500": "纯多头年度收益差",
            "LO_relative": "纯多头年度复合超额",
        }
    )
    charts = {
        name: f"![{title}]({(output / (name + '.png')).resolve().as_posix()})"
        for name, title in figures
    }
    files = [
        "performance.csv",
        "annual_returns.csv",
        "daily_nav.csv",
        "monthly_ic.csv",
        "ic_summary.csv",
        "universe_coverage.csv",
        "data_quality.csv",
        "rebalance_summary.csv",
        "trades.csv.gz",
        "unfilled_orders.csv.gz",
        "factor_signals.csv.gz",
        "turnover_summary.csv",
        "monthly_turnover.csv",
        "annual_turnover.csv",
        "run_metadata.json",
    ]
    links = "\n".join(
        f"- [{name}]({(output / name).resolve().as_posix()})" for name in files
    )
    benchmark_result = metrics.set_index("portfolio").loc["CSI500"]
    result_notes = "\n\n".join(
        f"{names[key]}扣费后累计收益 **{row['total_return']:.2%}**，年化收益 **{row['annual_return']:.2%}**，"
        f"最大回撤 **{row['max_drawdown']:.2%}**；相对中证500的复合超额为 **{row['relative_return']:.2%}**，"
        f"累计收益差为 **{row['return_difference'] * 100:.2f}个百分点**。"
        for key, row in strategy_metrics.iterrows()
    )
    report = f"""# A股{metadata['lookback']}日负向乖离率：多空、纯多头与中证500比较

回测区间：**{metadata['first_session']} 至 {metadata['last_session']}**，共{metadata['trading_days']:,}个交易日。
AKShare {metadata['akshare_version']}；后复权行情；月度调仓；开盘到开盘估值。

## 因子预测力与IC指标

原始因子为{metadata['lookback']}日乖离率：`BIAS = 后复权收盘价 / 最近{metadata['lookback']}个交易日后复权收盘价的简单均值 − 1`，均价包含当日，策略使用负向因子 `-BIAS`。
直接调用 `src/factor_test/analyzer.py` 的 `information_coefficient()`，设置 `periods_per_year=12`，
分别统计原始乖离率和负向因子的Pearson IC及Spearman Rank IC。

### IC统计

{markdown_table(ic_table)}

### 月度Rank IC表现

{charts.get('04_rank_ic', '')}

### 统计口径与样本覆盖

{markdown_table(ic_coverage)}

IC使用月末信号与下一开盘至再下一调仓开盘的收益，只统计完整持有月。
缺少有效买卖报价的股票仅从IC统计中排除，不事后从策略持仓中剔除。
图表使用 `-BIAS` 的Rank IC；CSV同时保留原始乖离率IC与负向调整后的IC。

月度ICIR为均值/标准差；年化ICIR为月度ICIR乘以√12。t/p值为均值等于0的普通双侧单样本t检验，
均未校正时间序列自相关；峰度使用超额峰度。缺失或非有限IC不计为0，标准差为0或有效样本不足时不报告ICIR及t/p值。
两种策略共用同一因子和股票池，IC统计相同；IC基于个股持有期毛收益，独立于组合交易费用，不等同于组合信息比率或Sharpe。

## 组合绩效与中证500比较

{result_notes}

同期中证500累计收益 **{benchmark_result['total_return']:.2%}**。

**估值提示：期末沿用陈旧报价的持仓数为{stale_note}，结果受停牌及退市后的冻结估值影响。**

{markdown_table(metric_table)}

### 累计净值

{charts['01_nav']}

### 超额收益与回撤

{charts['02_excess_drawdown']}

复合超额：`组合净值 / 中证500净值 - 1`；累计收益差：`组合累计收益 - 中证500累计收益`。
信息比率使用每日算术超额收益，Sharpe使用组合自身收益。组合收益均已计入交易费用。

## 年度收益

{charts['03_annual']}

{markdown_table(annual_display)}

## 策略与执行口径

- 原始乖离率：`BIAS(t) = C(t) / mean(C(t−{metadata['lookback'] - 1}), …, C(t)) − 1`，`C`为后复权收盘价。
  简单均价覆盖含信号日在内的最近{metadata['lookback']}个共同交易日；停牌或缺价沿用已知的最近有效收盘价，不向未来填充。
- 按负向因子处理，传给框架的策略因子为 `factor = -BIAS`。
- 调用 `src/factor_test/analyzer.py` 的 `portfolio_returns`，设置 `weight_method="quantile"`、
  `quantile_pct={metadata['quantile_pct']}`。两种策略使用相同信号、行情、日期及费率，各自独立运行。
- 多空组合设置 `long_short=True`：做多原始乖离率最低的{metadata['quantile_pct']:.0%}股票，做空最高的{metadata['quantile_pct']:.0%}股票，目标多空各50%，每侧内部等权。
- 纯多头组合设置 `long_short=False`：仅做多原始乖离率最低的{metadata['quantile_pct']:.0%}股票，目标100%资金等权配置；20%指选股比例。
  选股使用框架的平均排名，不按代码拆分并列值。
- 月末收盘形成信号，下一交易日开盘调仓，使用框架 `rebalance_every="monthly"`、`entry_lag=1`。
  持有期间股数固定，成交、现金、整数股、费用、停牌与缺价均由 `src/factor_test/portfolio.py` 核算。
- 每个信号时点剔除上市不足60个交易日的股票；均价须完整{metadata['lookback']}日A股价格历史，不使用上市前新三板报价补足历史。
- 每种策略初始资金{metadata['initial_value']:,.0f}元；佣金 `commission_rate={metadata['commission_rate']}`（买卖双向），
  印花税 `sell_tax_rate={metadata['sell_tax_rate']}`（仅卖出方向），滑点 `slippage_rate={metadata['slippage_rate']}`。
  多空组合融券年化费率{metadata['short_borrow_rate']:.2%}；纯多头不收取融券费用。
- 中证500使用000905价格指数，首日开盘买入，基准与组合均按每日开盘估值。
- 绩效通过框架 `portfolio_metrics` 计算。年化按252日，无风险利率0，现金不计息；样本末端保留未到期持仓。

## 交易费用核对

{markdown_table(cost_summary)}

费用直接汇总框架每日账户；卖出印花税包括卖空开仓，买入平空不征收卖出税。

## 换手率

{markdown_table(turnover_table)}

{charts['06_turnover']}

{markdown_table(annual_turnover_display)}

直接使用框架每日账户的 `turnover = 当日买卖成交额绝对值之和 / 前一日净资产 / 2`，为单边换手口径。
日均值包含无交易日，月度和年度值为每日换手率求和；月均值为各月合计的均值；年化倍数为日均值乘252。
统计包含首次建仓、正常调仓及未成交订单后续重试产生的实际成交，样本末日不额外清仓。
中证500为价格指数参考，不模拟其内部成分调整，因此不估计指数内部换手率；绩效表中的基准换手率0仅为参考账户占位值。

## 数据覆盖

股票清单共{metadata['requested_stocks']:,}只A股，成功获取{metadata['downloaded_stocks']:,}只；
包括上市与可取得的沪深退市证券，689前缀CDR不属于A股，已排除。

{markdown_table(quality_summary.reset_index())}

{charts['05_coverage']}

## 期末账户诊断

{markdown_table(account_end)}

陈旧报价持仓包括停牌或缺少行情的持仓，沿用最近有效价格估值。计数来自框架每日账户。
多空账户现金包含卖空所得，不代表全部可提取资金。

## 解释结果时的限制

{chr(10).join('- ' + item for item in warnings)}

## 可复核文件

{links}

接口依据：[AKShare股票文档](https://akshare.akfamily.xyz/data/stock/stock.html)、
[AKShare指数文档](https://akshare.akfamily.xyz/data/index/index.html)。
运行参数、执行框架、价格数据源和生成时间保存在 `run_metadata.json`。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
