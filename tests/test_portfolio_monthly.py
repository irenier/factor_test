"""Calendar-month scheduling in the shared account engine."""

from datetime import date
import unittest

import polars as pl

from factor_test.portfolio import backtest_portfolio


class MonthlyPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.days = [
            date(2020, 1, 30),
            date(2020, 1, 31),
            date(2020, 2, 3),
            date(2020, 2, 27),
            date(2020, 2, 28),
            date(2020, 3, 2),
            date(2020, 3, 30),
            date(2020, 3, 31),
            date(2020, 4, 1),
        ]
        self.prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=10.0)
                for day in self.days
                for symbol in "AB"
            ]
        ).lazy()

    def test_actual_month_ends_execute_on_next_observed_session(self):
        weights = pl.DataFrame(
            {
                "date": [self.days[i] for i in (0, 1, 4, 7)],
                "symbol": ["B", "A", "B", "A"],
                "weight": [1.0] * 4,
            }
        ).lazy()
        account = backtest_portfolio(
            weights, self.prices, rebalance_every="monthly", initial_value=100.0
        )
        self.assertEqual(
            account.trades.select("date", "symbol", "quantity").rows(),
            [
                (self.days[2], "A", 10),
                (self.days[5], "A", -10),
                (self.days[5], "B", 10),
                (self.days[8], "A", 10),
                (self.days[8], "B", -10),
            ],
        )
        self.assertTrue(all(day in self.days for day in account.daily["date"]))

    def test_missing_monthly_signal_exits_at_next_monthly_execution(self):
        weights = pl.DataFrame(
            {"date": [self.days[1]], "symbol": ["A"], "weight": [1.0]}
        ).lazy()
        account = backtest_portfolio(
            weights, self.prices, rebalance_every="monthly", initial_value=100.0
        )
        self.assertEqual(
            account.trades.select("date", "quantity").rows(),
            [(self.days[2], 10), (self.days[5], -10)],
        )

    def test_monthly_mode_requires_calendar_dates(self):
        prices = pl.DataFrame(
            {"date": [1, 2], "symbol": ["A"] * 2, "open": [10.0] * 2}
        ).lazy()
        weights = pl.DataFrame({"date": [1], "symbol": ["A"], "weight": [1.0]}).lazy()
        with self.assertRaisesRegex(ValueError, "monthly.*Date"):
            backtest_portfolio(weights, prices, rebalance_every="monthly")


if __name__ == "__main__":
    unittest.main()
