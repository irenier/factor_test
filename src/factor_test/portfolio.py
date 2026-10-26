"""Sequential portfolio accounting with non-overlapping rebalance intervals."""

from dataclasses import dataclass
from math import fsum, isfinite, sqrt, ulp
from numbers import Real
from statistics import mean, stdev
from typing import Literal
import polars as pl

from ._utils import validate_keys


def _validate_number(name, value, *, integer=False, positive=False):
    """所有数值参数共用同一套类型、有限性和范围校验。"""
    expected = "正" if positive else "非负"
    expected += "整数" if integer else "有限数"
    if integer:
        valid_type = type(value) is int
    else:
        valid_type = isinstance(value, Real) and not isinstance(value, bool)
    if not valid_type or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} 必须为{expected}")


def _validate_costs(commission_rate: float, slippage_rate: float, sell_tax_rate: float):
    rates = (commission_rate, slippage_rate, sell_tax_rate)
    for name, value in zip(
        ("commission_rate", "slippage_rate", "sell_tax_rate"), rates
    ):
        _validate_number(name, value)
    if sum(rates) >= 1:
        raise ValueError("commission_rate、slippage_rate、sell_tax_rate 合计必须小于 1")


def _prepare_backtest_inputs(weights, prices, price_col, suspended_col):
    """只读取所需列，统一校验并规范化；逐日循环不再检查输入格式。"""
    frames = (
        ("weights", weights, ["date", "symbol", "weight"]),
        ("prices", prices, ["date", "symbol", price_col]),
    )
    for name, frame, columns in frames:
        if not isinstance(frame, pl.LazyFrame):
            raise ValueError(f"{name} 必须为 Polars LazyFrame")
        missing = set(columns) - set(frame.collect_schema().names())
        if missing:
            raise ValueError(f"{name} 缺少列：{', '.join(sorted(missing))}")

    weight_schema, price_schema = weights.collect_schema(), prices.collect_schema()
    for name, dtype in (
        ("weight", weight_schema["weight"]),
        (price_col, price_schema[price_col]),
    ):
        if dtype != pl.Null and not dtype.is_numeric():
            raise ValueError(f"{name} 必须为数值列")
    if suspended_col is not None:
        if suspended_col not in price_schema:
            raise ValueError(f"prices 缺少停牌列：{suspended_col}")
        if price_schema[suspended_col] not in (pl.Boolean, pl.Null):
            raise ValueError(f"{suspended_col} 必须为布尔列")

    weight_data = weights.select(
        "date", "symbol", pl.col("weight").cast(pl.Float64)
    ).collect()
    is_suspended = (
        pl.col(suspended_col).cast(pl.Boolean).fill_null(True)
        if suspended_col
        else pl.lit(False)
    )
    price = pl.col(price_col).cast(pl.Float64)
    valid_price = price.is_finite().fill_null(False) & (price > 0).fill_null(False)
    price_data = (
        prices.select(
            "date",
            "symbol",
            pl.when(valid_price & ~is_suspended).then(price).alias("price"),
            is_suspended.alias("suspended"),
        )
        .sort("date")
        .collect()
    )
    if price_data.is_empty():
        raise ValueError("行情日期不能为空")

    for name, frame in (("weights", weight_data), ("prices", price_data)):
        try:
            validate_keys(frame.lazy(), ["date", "symbol"])
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    for key in ("date", "symbol"):
        weight_type, price_type = weight_data.schema[key], price_data.schema[key]
        if weight_data.is_empty():
            weight_data = weight_data.with_columns(pl.col(key).cast(price_type))
        elif weight_type.is_integer() and price_type.is_integer():
            weight_data = weight_data.with_columns(pl.col(key).cast(pl.Int64))
            price_data = price_data.with_columns(pl.col(key).cast(pl.Int64))
        elif weight_type != price_type:
            raise ValueError(f"weights 与 prices 的 {key} 类型必须一致")

    if weight_data.select(
        (~pl.col("weight").is_finite().fill_null(False)).any()
    ).item():
        raise ValueError("weight 必须有限且非空")
    gross = pl.col("weight").abs().sum().over("date")
    tolerance = 1e-6 if weight_schema["weight"] == pl.Float32 else 1e-12
    if weight_data.select((gross > 1 + tolerance).any()).item():
        raise ValueError("每个信号日的总绝对权重不能超过 1")
    # 只修正求和产生的浮点误差，不接受实际超额仓位。
    weight_data = weight_data.with_columns(
        (pl.col("weight") / pl.max_horizontal(pl.lit(1.0), gross)).alias("weight")
    )
    unknown_dates = (
        weight_data.select("date")
        .unique()
        .join(price_data.select("date").unique(), on="date", how="anti")
    )
    if not unknown_dates.is_empty():
        raise ValueError("信号日期必须包含在行情 date 中")
    return weight_data, price_data


@dataclass
class BacktestResult:
    daily: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    unfilled_orders: pl.DataFrame
    initial_value: float


def _trade_fees(
    notional: float, rates: tuple[float, float, float]
) -> tuple[float, float, float]:
    commission_rate, slippage_rate, sell_tax_rate = rates
    return (
        abs(notional) * commission_rate,
        abs(notional) * slippage_rate,
        max(-notional, 0.0) * sell_tax_rate,
    )


def _target_shares(weights, prices, shares, equity, rates) -> dict[str | int, int]:
    """预留连续目标的费用后取整数股，并按实际成交费用确认资金足够。"""
    if not weights:
        return {}
    symbols = shares.keys() | weights.keys()

    def cost(capital):
        return fsum(
            fsum(
                _trade_fees(
                    weights.get(symbol, 0.0) * capital
                    - shares.get(symbol, 0) * prices[symbol],
                    rates,
                )
            )
            for symbol in symbols
        )

    capital = equity
    if any(rates) and cost(equity) > 0:
        if cost(0.0) >= equity:
            raise ValueError("平仓费用超过账户权益")
        low, high = 0.0, equity
        for _ in range(60):
            middle = (low + high) / 2
            if middle + cost(middle) > equity:
                high = middle
            else:
                low = middle
        capital = low
    target = {}
    for symbol, weight in weights.items():
        quantity = weight * capital / prices[symbol]
        nearest = round(quantity)
        # 连续资金求解与除法会产生几个 ULP 的误差，不能因此误卖一整股。
        if abs(quantity - nearest) <= 4 * ulp(quantity):
            quantity = nearest
        if quantity := int(quantity):
            target[symbol] = quantity

    def required_capital():
        gross_value = fsum(abs(qty) * prices[symbol] for symbol, qty in target.items())
        fees = fsum(
            fsum(
                _trade_fees(
                    (target.get(symbol, 0) - shares.get(symbol, 0)) * prices[symbol],
                    rates,
                )
            )
            for symbol in symbols
        )
        return gross_value + fees

    # 近整数修正必须通过真实资金检查；刚好买不起时保守减少一股。
    for symbol in sorted(target):
        if required_capital() <= equity:
            break
        quantity = target[symbol]
        target[symbol] -= 1 if quantity > 0 else -1
    if required_capital() > equity:
        raise ValueError("账户权益不足以支付目标持仓及交易费用")
    return {symbol: quantity for symbol, quantity in target.items() if quantity}


def _rebalance(shares, weights, prices, tradable_symbols, equity, rates):
    """冻结受限旧仓，剩余资金按可交易目标比例分配，再成交股数差额。"""
    frozen = {
        symbol: qty for symbol, qty in shares.items() if symbol not in tradable_symbols
    }
    rejected = [
        dict(symbol=symbol, quantity=-qty, target_weight=None, reason="exit_blocked")
        for symbol, qty in frozen.items()
    ]
    eligible = {}
    for symbol, weight in weights.items():
        if not weight:
            continue
        if symbol in tradable_symbols:
            eligible[symbol] = weight
            continue
        rejected.append(
            dict(
                symbol=symbol,
                quantity=None,
                target_weight=weight,
                reason="entry_blocked",
            )
        )

    available = max(
        0.0,
        equity - fsum(abs(qty) * prices[symbol] for symbol, qty in frozen.items()),
    )
    if eligible and (frozen or rejected):
        gross = fsum(abs(weight) for weight in eligible.values())
        eligible = {symbol: weight / gross for symbol, weight in eligible.items()}
    movable = {symbol: qty for symbol, qty in shares.items() if symbol not in frozen}
    target = dict(frozen)
    if available > 0:
        target.update(_target_shares(eligible, prices, movable, available, rates))
    trades = []
    for symbol in sorted(shares.keys() | target.keys()):
        delta = target.get(symbol, 0) - shares.get(symbol, 0)
        if delta == 0:
            continue
        notional = delta * prices[symbol]
        fee, slippage, tax = _trade_fees(notional, rates)
        trades.append(
            dict(
                symbol=symbol,
                quantity=delta,
                price=prices[symbol],
                notional=notional,
                commission=fee,
                slippage=slippage,
                sell_tax=tax,
                cost=fsum((fee, slippage, tax)),
            )
        )
    return target, trades, rejected


def backtest_portfolio(
    weights: pl.LazyFrame,
    prices: pl.LazyFrame,
    *,
    rebalance_every: int | Literal["monthly"] = 5,
    entry_lag: int = 1,
    price_col: str = "open",
    initial_value: float = 1.0,
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    short_borrow_rate: float = 0.0,
    periods_per_year: int = 252,
    suspended_col: str | None = None,
    missing_price: str = "raise",
    liquidate_at_end: bool = False,
    record_positions: bool = True,
) -> BacktestResult:
    """按 prices.date 执行目标权重，维护现金和固定股数，而不是累计远期收益。

    持有期与调仓间隔统一为 rebalance_every 个交易区间；默认每五日调仓，
    =1 为每日调仓。始终使用一个完整资金账户，不建立重叠资金批次。
    ="monthly" 使用行情中每个自然月的最后交易日信号，并按 entry_lag
    延迟执行，持有至下一月度执行日；date 须为 Date 或 Datetime。
    首个信号日为调仓日历起点，之后仅使用间隔日的信号；其他日信号忽略。
    样本尾部未到调仓日的头寸继续按市值计价。
    日期序列直接取 prices 中去重并排序后的 date，不补充缺失日期。

    输入权重仅允许总绝对值 <= 1。先预留费用，再按 int(weight * capital / price)
    向零取整为整数股；不足一股的资金留现金，不再分配。调仓时冻结无法交易
    的旧仓，扣除其绝对市值后，剩余资金按可交易目标的绝对权重归一化分配，
    保留多空方向；无可交易目标时留现金。新信号仍按原调仓周期执行，
    冻结仓位复牌后在下一次调仓处理；到期无新信号时每日重试退出。
    到期同时调仓仅成交股数差额，
    不重复收取平仓、开仓费用。缺少调仓日信号时退出并持有现金，
    待下一预定调仓日再建仓，不跨日填充信号。

    suspended_col 指定停牌标记，True 或 null 表示停牌：不更新估值价格，
    不允许买卖，已有持仓沿用停牌前价格。停牌新仓跳过，其权重分配给其余
    可交易目标，复牌后不追单。无交易限制时保留原权重及其现金比例。
    未标记停牌的持仓缺价仍由 missing_price 控制："raise" 报错，"carry"
    沿用最近价格。非停牌且有有效报价才能买卖。
    最小交易单位为 1 股，现金不计息，无杠杆保证金模型；账户权益耗尽时抛错。
    record_positions=False 不保存逐日持仓快照，以节省大样本内存。
    """
    monthly = isinstance(rebalance_every, str) and rebalance_every == "monthly"
    if not monthly:
        _validate_number(
            "rebalance_every", rebalance_every, integer=True, positive=True
        )
    _validate_number("periods_per_year", periods_per_year, integer=True, positive=True)
    _validate_number("entry_lag", entry_lag, integer=True)
    _validate_number("initial_value", initial_value, positive=True)
    _validate_number("short_borrow_rate", short_borrow_rate)
    for name, value in (
        ("liquidate_at_end", liquidate_at_end),
        ("record_positions", record_positions),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} 必须为布尔值")
    if not isinstance(price_col, str) or not price_col:
        raise ValueError("price_col 必须为非空列名")
    if suspended_col is not None and (
        not isinstance(suspended_col, str) or not suspended_col
    ):
        raise ValueError("suspended_col 必须为非空列名或 None")
    if missing_price not in ("raise", "carry"):
        raise ValueError("missing_price 必须为 raise 或 carry")
    _validate_costs(commission_rate, slippage_rate, sell_tax_rate)
    weight_data, price_data = _prepare_backtest_inputs(
        weights, prices, price_col, suspended_col
    )
    date_index = (
        price_data.select("date")
        .unique(maintain_order=True)
        .with_row_index("signal_index")
        .with_columns(pl.col("signal_index").cast(pl.Int64))
    )
    dates = date_index["date"].to_list()
    signal_data = weight_data.join(date_index, on="date")
    monthly_expiries = {}
    if monthly:
        if price_data.schema["date"] not in (pl.Date, pl.Datetime):
            raise ValueError("monthly 调仓要求 date 为 Date 或 Datetime")
        month_ends = date_index.filter(
            pl.col("signal_index")
            == pl.col("signal_index").max().over(pl.col("date").dt.strftime("%Y-%m"))
        )["signal_index"]
        entries = [
            index + entry_lag for index in month_ends if index + entry_lag < len(dates)
        ]
        monthly_expiries = dict(zip(entries, entries[1:] + [None]))
        signal_data = signal_data.filter(
            pl.col("signal_index").is_in(month_ends.implode())
        )
    else:
        signal_data = signal_data.filter(
            (pl.col("signal_index") - pl.col("signal_index").min()) % rebalance_every
            == 0
        )

    # 按首个信号的位置筛选调仓日，再偏移到执行位置；全程保留为表。
    schedule = (
        signal_data.with_columns(
            (pl.col("signal_index") + entry_lag).alias("entry_index")
        )
        .filter(pl.col("entry_index") < len(dates))
        .group_by("entry_index")
        .agg(
            pl.col("date").first().alias("signal_date"),
            pl.col("symbol").alias("symbols"),
            pl.col("weight").alias("weights"),
        )
        .sort("entry_index")
    )
    scheduled_entries = iter(schedule.iter_rows(named=True))
    next_entry = next(scheduled_entries, None)

    cash = initial_value
    shares: dict[str | int, int] = {}
    expiry_index = None
    portfolio_signal_date = None
    previous_short_value = 0.0
    rates = (commission_rate, slippage_rate, sell_tax_rate)
    last_prices = {}
    daily_rows, position_rows, trade_rows, unfilled_rows = [], [], [], []
    previous_equity = peak = initial_value

    daily_quotes = price_data.group_by("date", maintain_order=True)
    for day_index, ((date,), quotes_today) in enumerate(daily_quotes):
        fresh = dict(
            quotes_today.drop_nulls("price").select("symbol", "price").iter_rows()
        )
        suspended = set(quotes_today.filter("suspended")["symbol"])
        last_prices.update(fresh)
        if missing_price == "raise" and shares.keys() - fresh.keys() - suspended:
            raise ValueError(
                f"{date}: 已有持仓缺少有效报价；可选择 missing_price='carry'"
            )
        trades_today, unfilled_today = [], []
        borrow_cost = previous_short_value * short_borrow_rate / periods_per_year
        cash -= borrow_cost
        before = cash + fsum(
            qty * last_prices[symbol] for symbol, qty in shares.items()
        )
        if before <= 0:
            raise ValueError(f"{date}: 账户权益耗尽，无法继续回测")
        entry = None
        if next_entry is not None and next_entry["entry_index"] == day_index:
            entry = next_entry
            next_entry = next(scheduled_entries, None)
        due = expiry_index is not None and expiry_index <= day_index
        final_exit = liquidate_at_end and day_index == len(dates) - 1

        # 到期退出、新信号调仓和期末清仓共用同一条成交流程。
        if due or entry is not None or final_exit:
            has_new_signal = entry is not None and not final_exit
            if has_new_signal:
                order_signal_date = entry["signal_date"]
                requested = dict(zip(entry["symbols"], entry["weights"]))
            else:
                order_signal_date = portfolio_signal_date
                requested = {}
            shares, trades_today, unfilled_today = _rebalance(
                shares, requested, last_prices, fresh.keys(), before, rates
            )
            cash = fsum(
                (
                    cash,
                    -fsum(row["notional"] for row in trades_today),
                    -fsum(row["cost"] for row in trades_today),
                )
            )
            # 仅消除浮点求和尾差，不把真实资金缺口当作可用资金。
            if -8 * ulp(before) <= cash < 0:
                cash = 0.0
            trade_rows.extend(
                dict(row, date=date, signal_date=order_signal_date)
                for row in trades_today
            )
            unfilled_rows.extend(
                dict(
                    row,
                    date=date,
                    signal_date=(
                        portfolio_signal_date
                        if row["reason"] == "exit_blocked"
                        else order_signal_date
                    ),
                )
                for row in unfilled_today
            )
            if has_new_signal:
                expiry_index = (
                    monthly_expiries[day_index]
                    if monthly
                    else day_index + rebalance_every
                )
                portfolio_signal_date = order_signal_date
            if not shares:
                expiry_index = None

        values = {symbol: qty * last_prices[symbol] for symbol, qty in shares.items()}
        long_value = fsum(value for value in values.values() if value > 0)
        short_value = -fsum(value for value in values.values() if value < 0)
        previous_short_value = short_value
        equity = fsum((cash, long_value, -short_value))
        if equity <= 0:
            raise ValueError(f"{date}: 账户权益耗尽")
        fees = fsum(row["commission"] for row in trades_today)
        slippage = fsum(row["slippage"] for row in trades_today)
        tax = fsum(row["sell_tax"] for row in trades_today)
        cost = fsum((fees, slippage, tax, borrow_cost))
        traded = fsum(abs(row["notional"]) for row in trades_today)
        peak = max(peak, equity)
        daily_rows.append(
            dict(
                date=date,
                equity=equity,
                cash=cash,
                long_value=long_value,
                short_value=short_value,
                gross_exposure=(long_value + short_value) / equity,
                net_exposure=(long_value - short_value) / equity,
                gross_return=(equity + cost) / previous_equity - 1,
                net_return=equity / previous_equity - 1,
                nav=equity / initial_value,
                drawdown=equity / peak - 1,
                commission=fees,
                slippage=slippage,
                sell_tax=tax,
                borrow_cost=borrow_cost,
                cost=cost,
                traded_notional=traded,
                turnover=traded / previous_equity / 2,
                unfilled_orders=len(unfilled_today),
                stale_positions=sum(symbol not in fresh for symbol in shares),
                suspended_positions=sum(symbol in suspended for symbol in shares),
            )
        )
        if record_positions:
            expiry_date = (
                dates[expiry_index]
                if expiry_index is not None and expiry_index < len(dates)
                else None
            )
            for symbol, qty in shares.items():
                value = values[symbol]
                position_rows.append(
                    dict(
                        date=date,
                        symbol=symbol,
                        quantity=qty,
                        price=last_prices[symbol],
                        value=value,
                        weight=value / equity,
                        stale_price=symbol not in fresh,
                        suspended=symbol in suspended,
                        signal_date=portfolio_signal_date,
                        expiry_date=expiry_date,
                    )
                )
        previous_equity = equity

    date_type, symbol_type = price_data.schema["date"], weight_data.schema["symbol"]
    return BacktestResult(
        daily=pl.DataFrame(daily_rows),
        positions=pl.DataFrame(
            position_rows,
            schema={
                "date": date_type,
                "symbol": symbol_type,
                "quantity": pl.Int64,
                "price": pl.Float64,
                "value": pl.Float64,
                "weight": pl.Float64,
                "stale_price": pl.Boolean,
                "suspended": pl.Boolean,
                "signal_date": date_type,
                "expiry_date": date_type,
            },
        ),
        trades=pl.DataFrame(
            trade_rows,
            schema={
                "date": date_type,
                "signal_date": date_type,
                "symbol": symbol_type,
                "quantity": pl.Int64,
                "price": pl.Float64,
                "notional": pl.Float64,
                "commission": pl.Float64,
                "slippage": pl.Float64,
                "sell_tax": pl.Float64,
                "cost": pl.Float64,
            },
        ),
        unfilled_orders=pl.DataFrame(
            unfilled_rows,
            schema={
                "date": date_type,
                "symbol": symbol_type,
                "signal_date": date_type,
                "quantity": pl.Int64,
                "target_weight": pl.Float64,
                "reason": pl.String,
            },
        ),
        initial_value=initial_value,
    )


def portfolio_metrics(
    result: BacktestResult,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    benchmark: pl.LazyFrame | None = None,
) -> pl.DataFrame:
    """账户总收益、年化收益/波动/Sharpe、最大回撤、成本和换手。

    年化期数为 daily 的观测数（含建仓等待期和现金日）。benchmark 可传
    date, benchmark_return，必须覆盖所有账户日期；额外给出相对收益及 beta。
    """
    if type(periods_per_year) is not int or periods_per_year <= 0:
        raise ValueError("periods_per_year 必须为正整数")
    if not isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate 必须为大于 -1 的有限年化利率")
    daily = result.daily.sort("date")
    values = daily["net_return"].to_list()
    if not values:
        raise ValueError("账户结果不能为空")
    n = len(values)
    rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = [value - rf for value in values]
    volatility = stdev(values) if n > 1 else None
    total = daily["nav"][-1] - 1
    annual = (1 + total) ** (periods_per_year / n) - 1
    drawdown = daily["drawdown"].min()
    metrics = dict(
        n_periods=n,
        total_return=total,
        annual_return=annual,
        annual_volatility=(
            volatility * sqrt(periods_per_year) if volatility is not None else None
        ),
        sharpe=(
            mean(excess) / volatility * sqrt(periods_per_year)
            if volatility and volatility > 0
            else None
        ),
        max_drawdown=drawdown,
        calmar=annual / abs(drawdown) if drawdown < 0 else None,
        total_cost=daily["cost"].sum(),
        mean_turnover=daily["turnover"].mean(),
        total_traded_notional=daily["traded_notional"].sum(),
    )
    if benchmark is not None:
        validate_keys(benchmark, ["date"])
        joined = (
            daily.lazy()
            .join(benchmark.select("date", "benchmark_return"), on="date", how="left")
            .sort("date")
            .collect()
        )
        market = joined["benchmark_return"].to_list()
        if any(x is None or not isfinite(x) or x <= -1 for x in market):
            raise ValueError("benchmark 必须覆盖账户所有日期，收益有限且大于 -1")
        active = [x - y for x, y in zip(values, market)]
        tracking = stdev(active) if n > 1 else None
        benchmark_nav = 1.0
        for value in market:
            benchmark_nav *= 1 + value
        mean_market, mean_return = mean(market), mean(values)
        variance = sum((x - mean_market) ** 2 for x in market)
        beta = (
            sum((x - mean_return) * (y - mean_market) for x, y in zip(values, market))
            / variance
            if variance
            else None
        )
        alpha = mean_return - beta * mean_market if beta is not None else None
        metrics.update(
            relative_return=(1 + total) / benchmark_nav - 1,
            annual_tracking_error=(
                tracking * sqrt(periods_per_year) if tracking is not None else None
            ),
            information_ratio=(
                mean(active) / tracking * sqrt(periods_per_year)
                if tracking and tracking > 0
                else None
            ),
            beta=beta,
            annual_alpha=(
                (1 + alpha) ** periods_per_year - 1
                if alpha is not None and alpha > -1
                else None
            ),
        )
    return pl.DataFrame([metrics])
