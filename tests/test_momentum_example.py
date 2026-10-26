"""Economic invariants for the real-data monthly example (no network)."""

import unittest

import numpy as np
import pandas as pd
import polars as pl

from example.momentum_backtest import build_signals as polars_build_signals
from example.framework_account import run_accounts as polars_run_accounts


def build_signals(
    close, tradable, listing, *, delisting=None, signal_dates=None, **kwargs
):
    """Keep the original Pandas regression fixtures; adapt shapes, not calculations."""
    quotes = pl.DataFrame(
        [
            dict(
                date=day.date(),
                symbol=symbol,
                close=float(close.loc[day, symbol]),
                open=float(close.loc[day, symbol]),
                volume=1.0 if tradable.loc[day, symbol] else 0.0,
            )
            for day in close.index
            for symbol in close.columns
        ]
    ).lazy()
    stocks = pl.DataFrame(
        [
            dict(
                symbol=symbol,
                listing_date=listing[symbol].date(),
                delisting_date=(
                    None
                    if delisting is None or pd.isna(delisting[symbol])
                    else delisting[symbol].date()
                ),
            )
            for symbol in close.columns
        ],
        schema_overrides={"listing_date": pl.Date, "delisting_date": pl.Date},
    )
    calendar = pl.DataFrame({"date": close.index.date.tolist()})
    selected = None if signal_dates is None else [day.date() for day in signal_dates]
    signals, coverage = polars_build_signals(
        quotes, stocks, calendar, signal_dates=selected, **kwargs
    )
    result = {}
    for day in coverage["date"]:
        frame = signals.filter(pl.col("date") == day).drop("date")
        result[pd.Timestamp(day)] = pd.DataFrame(
            frame.to_dict(as_series=False)
        ).set_index("symbol")
    return result, pd.DataFrame(coverage.to_dict(as_series=False))


def run_accounts(signals, prices, benchmark, **kwargs):
    """Retain the pre-refactor economic assertions as an independent regression oracle."""
    market = pl.DataFrame(
        {"date": benchmark.index.date.tolist(), "benchmark_nav": benchmark.to_list()}
    )
    return tuple(
        pd.DataFrame(frame.to_dict(as_series=False))
        for frame in polars_run_accounts(signals, prices, market, **kwargs)
    )


class MomentumExampleTests(unittest.TestCase):
    def test_listing_age_uses_exchange_sessions_and_accepts_day_60(self):
        days = pd.bdate_range("2020-01-01", periods=320)
        close = pd.DataFrame(
            {"old": np.arange(320) + 100.0, "new": np.arange(320) + 100.0}, index=days
        )
        listing = pd.Series({"old": days[0], "new": days[210]})
        signals, _ = build_signals(
            close, close.notna(), listing, lookback=10, min_age=60
        )
        # At April 2021 month-end the new stock has 73 exchange sessions.
        for day, frame in signals.items():
            age = days.get_loc(day) - 210 + 1
            self.assertEqual("new" in frame.index, age >= 60)
        signals, _ = build_signals(
            close,
            close.notna(),
            listing,
            lookback=10,
            min_age=60,
            signal_dates=[days[268], days[269]],
        )
        self.assertNotIn("new", signals[days[268]].index)
        self.assertIn("new", signals[days[269]].index)

    def test_default_bias_is_price_over_last_20_session_mean_and_uses_no_future_data(
        self,
    ):
        days = pd.bdate_range("2020-01-01", periods=300)
        close = pd.DataFrame(
            {"A": np.arange(300) + 100.0, "B": np.arange(300) + 200.0}, index=days
        )
        listing = pd.Series({"A": days[0], "B": days[0]})
        chosen = days[270]
        first, _ = build_signals(close, close.notna(), listing, signal_dates=[chosen])
        changed = close.copy()
        changed.loc[days[271] :, "A"] *= 9
        second, _ = build_signals(
            changed, close.notna(), listing, signal_dates=[chosen]
        )
        self.assertAlmostEqual(
            first[chosen].loc["A", "factor"],
            close.loc[days[270], "A"] / close["A"].iloc[251:271].mean() - 1,
        )
        pd.testing.assert_frame_equal(first[chosen], second[chosen])
        changed.loc[days[250], "A"] *= 7
        outside, _ = build_signals(
            changed, close.notna(), listing, signal_dates=[chosen]
        )
        pd.testing.assert_frame_equal(first[chosen], outside[chosen])
        changed.loc[chosen, "A"] *= 2
        current, _ = build_signals(
            changed, close.notna(), listing, signal_dates=[chosen]
        )
        self.assertAlmostEqual(
            current[chosen].loc["A", "factor"],
            changed.loc[chosen, "A"] / changed["A"].iloc[251:271].mean() - 1,
        )

    def test_default_20_day_bias_can_include_stock_on_60th_listing_session(self):
        days = pd.bdate_range("2020-01-01", periods=300)
        close = pd.DataFrame({"A": np.arange(300) + 100.0}, index=days)
        signals, _ = build_signals(
            close,
            close.notna(),
            pd.Series({"A": days[200]}),
            signal_dates=[days[258], days[259]],
        )
        self.assertTrue(signals[days[258]].empty)
        self.assertEqual(signals[days[259]].loc["A", "listing_age"], 60)
        self.assertAlmostEqual(
            signals[days[259]].loc["A", "factor"],
            close.loc[days[259], "A"] / close["A"].iloc[240:260].mean() - 1,
        )

    def test_bias_needs_twenty_inclusive_post_listing_prices_and_constant_price_has_zero_bias(
        self,
    ):
        days = pd.bdate_range("2020-01-01", periods=40)
        close = pd.DataFrame({"A": 10.0}, index=days)
        signals, _ = build_signals(
            close,
            close.notna(),
            pd.Series({"A": days[10]}),
            min_age=1,
            signal_dates=[days[28], days[29]],
        )
        self.assertTrue(signals[days[28]].empty)
        self.assertEqual(signals[days[29]].loc["A", "factor"], 0.0)

    def test_delisted_stock_leaves_the_signal_universe_on_delisting_day(self):
        days = pd.bdate_range("2020-01-01", periods=300)
        close = pd.DataFrame({"A": np.arange(300) + 100.0}, index=days)
        signals, coverage = build_signals(
            close,
            close.notna(),
            pd.Series({"A": days[0]}),
            delisting=pd.Series({"A": days[270]}),
            signal_dates=[days[269], days[270]],
        )
        self.assertEqual(len(signals[days[269]]), 1)
        self.assertTrue(signals[days[270]].empty)
        self.assertEqual(coverage["listed"].to_list(), [1, 0])

    def test_pre_listing_quotes_do_not_supply_the_252_day_a_share_history(self):
        days = pd.bdate_range("2020-01-01", periods=320)
        close = pd.DataFrame({"A": np.arange(320) + 100.0}, index=days)
        signals, coverage = build_signals(
            close,
            close.notna(),
            pd.Series({"A": days[200]}),
            signal_dates=[days[270]],
            lookback=252,
        )
        self.assertEqual(coverage["eligible_age"].iloc[0], 1)
        self.assertTrue(signals[days[270]].empty)

    def test_example_uses_framework_integer_shares_and_metrics(self):
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
        prices = pl.DataFrame(
            [
                dict(
                    date=day.date(),
                    symbol=symbol,
                    suspended=False,
                    open=(
                        20.0
                        if t == 2 and symbol == "A"
                        else 5.0 if t == 2 and symbol == "E" else 10.0
                    ),
                )
                for t, day in enumerate(days)
                for symbol in "ABCDE"
            ]
        ).lazy()
        signals = pl.DataFrame(
            {
                "date": [days[0].date()] * 5,
                "symbol": list("ABCDE"),
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
                "quantile": [1, 2, 3, 4, 5],
            }
        )
        benchmark = pd.Series([1.0, 2.0], index=days[1:])
        daily, metrics, trades, _, _ = run_accounts(
            signals,
            prices,
            benchmark,
            start=days[1].date(),
            initial_value=1000.0,
            commission_rate=0.001,
            sell_tax_rate=0.0,
        )
        long_short = daily[daily["portfolio"] == "Momentum_LS"]
        self.assertAlmostEqual(long_short["nav"].iloc[0], 0.99902)
        self.assertAlmostEqual(long_short["nav"].iloc[-1], 1.73402)
        self.assertEqual(
            trades[trades["portfolio"] == "Momentum_LS"]
            .set_index("symbol")["quantity"]
            .to_dict(),
            {"A": 49, "E": -49},
        )
        self.assertAlmostEqual(
            metrics.set_index("portfolio").loc["Momentum_LS", "total_return"], 0.73402
        )

    def test_long_only_uses_lowest_20_percent_with_full_capital_and_new_default_fees(
        self,
    ):
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
        prices = pl.DataFrame(
            [
                dict(date=day.date(), symbol=symbol, open=10.0, suspended=False)
                for day in days
                for symbol in "ABCDE"
            ]
        ).lazy()
        signals = pl.DataFrame(
            {
                "date": [days[0].date()] * 5,
                "symbol": list("ABCDE"),
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        daily, metrics, trades, _, _ = run_accounts(
            signals,
            prices,
            pd.Series(1.0, index=days[1:]),
            start=days[1].date(),
            initial_value=1000.0,
        )
        self.assertEqual(
            set(metrics["portfolio"]), {"Momentum_LS", "Momentum_LongOnly", "CSI500"}
        )
        long_only = trades[trades["portfolio"] == "Momentum_LongOnly"]
        self.assertEqual(long_only.set_index("symbol")["quantity"].to_dict(), {"A": 99})
        self.assertAlmostEqual(long_only["commission"].sum(), 0.297)
        self.assertEqual(long_only["sell_tax"].sum(), 0.0)
        self.assertAlmostEqual(
            daily.loc[daily["portfolio"] == "Momentum_LongOnly", "nav"].iloc[-1],
            0.999703,
        )
        long_short = trades[trades["portfolio"] == "Momentum_LS"]
        self.assertAlmostEqual(long_short["commission"].sum(), 0.294)
        self.assertAlmostEqual(long_short["sell_tax"].sum(), 0.49)
        self.assertAlmostEqual(
            daily.loc[daily["portfolio"] == "Momentum_LS", "nav"].iloc[-1], 0.999216
        )

    def test_both_portfolios_charge_commission_on_all_trades_and_tax_on_sells_only(
        self,
    ):
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"])
        prices = pl.DataFrame(
            [
                dict(date=day.date(), symbol=symbol, open=10.0, suspended=False)
                for day in days
                for symbol in "ABCDE"
            ]
        ).lazy()
        signals = pl.DataFrame(
            [
                dict(date=days[0].date(), symbol=symbol, factor=float(rank))
                for rank, symbol in enumerate("ABCDE", 1)
            ]
            + [
                dict(date=days[2].date(), symbol=symbol, factor=float(rank))
                for rank, symbol in enumerate("EDCBA", 1)
            ]
        )
        daily, _, trades, _, _ = run_accounts(
            signals,
            prices,
            pd.Series(1.0, index=days[1:]),
            start=days[1].date(),
            initial_value=1000.0,
        )
        self.assertEqual(set(trades["portfolio"]), {"Momentum_LS", "Momentum_LongOnly"})
        np.testing.assert_allclose(
            trades["commission"], trades["notional"].abs() * 0.0003
        )
        np.testing.assert_allclose(
            trades["sell_tax"], (-trades["notional"]).clip(lower=0) * 0.001
        )
        np.testing.assert_allclose(
            trades["cost"], trades["commission"] + trades["sell_tax"]
        )
        self.assertTrue((trades["slippage"] == 0).all())
        for label in ("Momentum_LS", "Momentum_LongOnly"):
            account = daily[daily["portfolio"] == label]
            ledger = trades[trades["portfolio"] == label]
            self.assertTrue(
                (ledger["notional"] > 0).any() and (ledger["notional"] < 0).any()
            )
            self.assertAlmostEqual(account["cost"].sum(), ledger["cost"].sum())
            self.assertAlmostEqual(
                account["nav"].iloc[-1], 1 - ledger["cost"].sum() / 1000.0
            )


if __name__ == "__main__":
    unittest.main()
