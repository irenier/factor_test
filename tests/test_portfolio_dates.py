from datetime import date
import unittest

import polars as pl

from factor_test.portfolio import backtest_portfolio


class PortfolioDateTests(unittest.TestCase):
    def test_final_liquidation_keeps_the_held_portfolios_signal_date(self):
        weights = pl.DataFrame(
            {
                "date": [1, 2, 3],
                "symbol": ["A", "A", "B"],
                "weight": [1.0] * 3,
            }
        ).lazy()
        prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=100.0)
                for day in (1, 2, 3)
                for symbol in "AB"
            ]
        ).lazy()
        result = backtest_portfolio(
            weights,
            prices,
            initial_value=1000,
            rebalance_every=1,
            entry_lag=0,
            liquidate_at_end=True,
        )
        self.assertEqual(
            result.trades.select("date", "signal_date").rows(), [(1, 1), (3, 2)]
        )
        self.assertEqual(result.trades["symbol"].to_list(), ["A", "A"])

    def test_unsorted_multi_asset_signals_use_first_signal_as_anchor(self):
        days = [10, 20, 40, 80, 160, 320]
        prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=price)
                for day in days
                for symbol, price in [("A", 10.0), ("B", 20.0)]
            ]
        ).lazy()
        weights = pl.DataFrame(
            {
                "date": [80, 20, 40, 20, 320],
                "symbol": ["B", "B", "A", "A", "A"],
                "weight": [1.0, 0.5, 1.0, 0.5, 1.0],
            }
        ).lazy()

        result = backtest_portfolio(
            weights, prices, initial_value=100, rebalance_every=2
        )

        self.assertEqual(
            result.trades.select("date", "symbol", "quantity").rows(),
            [
                (40, "A", 5),
                (40, "B", 2),
                (160, "A", -5),
                (160, "B", 3),
            ],
        )

    def test_zero_weight_anchor_and_same_day_execution(self):
        prices = pl.DataFrame(
            {"date": [1, 2, 3, 4, 5], "symbol": ["A"] * 5, "open": [10.0] * 5}
        ).lazy()
        weights = pl.DataFrame(
            {
                "date": [4, 2, 1, 3, 5],
                "symbol": ["A"] * 5,
                "weight": [1.0, 1.0, 0.0, 1.0, 0.0],
            }
        ).lazy()

        result = backtest_portfolio(
            weights, prices, initial_value=100, rebalance_every=2, entry_lag=0
        )

        self.assertEqual(
            result.trades.select("date", "quantity").rows(), [(3, 10), (5, -10)]
        )

    def test_price_dates_are_sorted_unique_and_missing_dates_are_skipped(self):
        days = [date(2026, 10, day) for day in (19, 21, 23, 26)]
        prices = pl.DataFrame(
            {
                "date": [days[3], days[1], days[0], days[2]] * 2,
                "symbol": ["A"] * 4 + ["B"] * 4,
                "open": [100.0] * 8,
            }
        ).lazy()
        weights = pl.DataFrame(
            {
                "date": [days[0]],
                "symbol": ["A"],
                "weight": [1.0],
            }
        ).lazy()

        result = backtest_portfolio(
            weights, prices, rebalance_every=2, initial_value=100
        )

        self.assertEqual(result.daily["date"].to_list(), days)
        self.assertEqual(result.trades["date"].to_list(), [days[1], days[3]])
        self.assertEqual(result.positions.schema["date"], pl.Date)

    def test_signal_dates_must_exist_in_prices(self):
        prices = pl.DataFrame(
            {"date": [1, 3], "symbol": ["A"] * 2, "open": [100.0] * 2}
        ).lazy()
        weights = pl.DataFrame({"date": [2], "symbol": ["A"], "weight": [1.0]}).lazy()

        with self.assertRaisesRegex(ValueError, "信号日期必须包含在行情 date 中"):
            backtest_portfolio(weights, prices)

    def test_empty_weights_keep_cash_on_all_price_dates(self):
        prices = pl.DataFrame(
            {"date": [3, 1], "symbol": ["A"] * 2, "open": [100.0] * 2}
        ).lazy()
        weights = pl.DataFrame(
            schema={"date": pl.Int64, "symbol": pl.String, "weight": pl.Float64}
        ).lazy()

        result = backtest_portfolio(weights, prices)

        self.assertEqual(result.daily["date"].to_list(), [1, 3])
        self.assertEqual(result.daily["nav"].to_list(), [1.0, 1.0])
        self.assertTrue(result.trades.is_empty())

    def test_empty_prices_are_rejected(self):
        prices = pl.DataFrame(
            schema={"date": pl.Int64, "symbol": pl.String, "open": pl.Float64}
        ).lazy()
        weights = pl.DataFrame(
            schema={"date": pl.Int64, "symbol": pl.String, "weight": pl.Float64}
        ).lazy()

        with self.assertRaisesRegex(ValueError, "行情日期不能为空"):
            backtest_portfolio(weights, prices)

    def test_missing_quote_on_an_observed_date_can_be_carried(self):
        prices = pl.DataFrame(
            {"date": [1, 2, 3, 4], "symbol": ["A"] * 4, "open": [100.0, 100, None, 110]}
        ).lazy()
        weights = pl.DataFrame({"date": [1], "symbol": ["A"], "weight": [1.0]}).lazy()

        with self.assertRaises(ValueError):
            backtest_portfolio(weights, prices, rebalance_every=1, initial_value=100)
        result = backtest_portfolio(
            weights, prices, rebalance_every=1, initial_value=100, missing_price="carry"
        )

        self.assertEqual(result.daily["stale_positions"].to_list(), [0, 0, 1, 0])
        self.assertEqual(result.trades["date"].to_list(), [2, 4])
        self.assertAlmostEqual(result.daily["nav"][-1], 1.1)


if __name__ == "__main__":
    unittest.main()
