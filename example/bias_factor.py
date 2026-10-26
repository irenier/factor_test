"""Price-bias factor implemented through the project's existing factor interface."""

import polars as pl

from factor_test.factor import BaseFactor

DEFAULT_BIAS_WINDOW = 20


class BiasFactor(BaseFactor):
    """C(t) / SMA(C, window) - 1 on a caller-supplied common trading calendar.

    Input requires unique date/symbol rows and a close column. The caller owns
    calendar alignment and any suspension-price carry. No future values or
    prices from another symbol enter the rolling mean. Invalid prices produce
    null factor values rather than non-finite signals.
    """

    def __init__(self, window: int = DEFAULT_BIAS_WINDOW):
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window

    def compute(self, data: pl.LazyFrame) -> pl.LazyFrame:
        required = {"date", "symbol", "close"}
        if missing := required - set(data.collect_schema().names()):
            raise ValueError(f"BiasFactor missing columns: {sorted(missing)}")
        close = pl.col("close").cast(pl.Float64)
        valid_close = pl.when(close.is_finite() & (close > 0)).then(close)
        prepared = data.sort(["symbol", "date"]).with_columns(
            valid_close.rolling_mean(window_size=self.window, min_samples=self.window)
            .over("symbol")
            .alias("_bias_mean")
        )
        value = valid_close / pl.col("_bias_mean") - 1
        return prepared.with_columns(
            pl.when(value.is_finite()).then(value).alias("factor")
        ).drop("_bias_mean")
