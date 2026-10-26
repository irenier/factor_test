"""Guard minute-session handling, observation quality and forward signal timing."""

from datetime import date, datetime, timedelta
import unittest

import numpy as np
import pandas as pd
import polars as pl

from example.jump.factor import compute_monthly_jump_factors, prepare_intraday_returns
from example.jump.jump_inputs import complete_trading_days, eligible_monthly_factors
from example.framework_account import run_accounts


class JumpInputTests(unittest.TestCase):
    def test_account_direction_selects_high_or_low_quantiles_without_changing_fees(
        self,
    ):
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])
        prices = pl.DataFrame(
            [
                dict(date=day.date(), symbol=symbol, open=10.0, suspended=False)
                for day in dates
                for symbol in "ABCDE"
            ]
        ).lazy()
        signals = pl.DataFrame(
            {
                "date": [dates[0].date()] * 5,
                "symbol": list("ABCDE"),
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        for direction, expected_long in [(1, "E"), (-1, "A")]:
            daily, _, trades, _, _ = run_accounts(
                signals,
                prices,
                pl.DataFrame(
                    {"date": dates[1:].date.tolist(), "benchmark_nav": [1.0, 1.0]}
                ),
                start=dates[1].date(),
                initial_value=1000.0,
                factor_direction=direction,
                portfolio_prefix="JArr",
            )
            long_only = trades.filter(pl.col("portfolio") == "JArr_LongOnly")
            self.assertEqual(long_only["symbol"].to_list(), [expected_long])
            self.assertAlmostEqual(long_only["commission"].sum(), 0.297)
            self.assertEqual(long_only["sell_tax"].sum(), 0.0)
            self.assertTrue(
                (
                    daily.filter(pl.col("portfolio") == "JArr_LongOnly")["short_value"]
                    == 0
                ).all()
            )

    def test_returns_restart_at_lunch_and_do_not_include_the_lunch_price_gap(self):
        frame = pl.DataFrame(
            {
                "symbol": ["A"] * 4,
                "tradeday": [date(2024, 1, 2)] * 4,
                "ts": [
                    datetime(2024, 1, 2, 11, 25),
                    datetime(2024, 1, 2, 11, 30),
                    datetime(2024, 1, 2, 13, 5),
                    datetime(2024, 1, 2, 13, 10),
                ],
                "timeframe": [5] * 4,
                "open": [100.0, 101.0, 150.0, 151.0],
                "close": [101.0, 102.0, 151.0, 152.0],
            }
        )
        result = prepare_intraday_returns(frame.lazy()).collect()
        self.assertEqual(result["session_id"].to_list(), [1, 1, 2, 2])
        self.assertAlmostEqual(result["log_return"][2], np.log(151.0 / 150.0))
        self.assertAlmostEqual(result["log_return"][3], np.log(152.0 / 151.0))

    def test_monthly_factors_keep_the_supplied_definitions_and_day_denominator(self):
        daily = pl.DataFrame(
            {
                "symbol": ["A"] * 3,
                "trade_date": [date(2024, 1, d) for d in [2, 3, 4]],
                "log_return_daily": [0.1, -0.2, 0.3],
                "T_JO": [3.0, -4.0, 1.0],
                "I_Jump": [1, 1, 0],
                "open_daily": [10.0, 11.0, 9.0],
                "close_daily": [11.0, 9.0, 12.0],
            }
        )
        row = compute_monthly_jump_factors(daily.lazy()).collect().row(0, named=True)
        self.assertEqual(row["D"], 3)
        self.assertAlmostEqual(row["JArr"], 2.0 / 3.0)
        self.assertAlmostEqual(row["JArr_pos"], 1.0 / 3.0)
        self.assertAlmostEqual(row["JArr_neg"], 1.0 / 3.0)
        self.assertAlmostEqual(row["JIntensity"], 8.0 / 3.0)
        self.assertAlmostEqual(row["JR"], np.expm1(-0.1))
        self.assertAlmostEqual(row["JR_abs"], np.expm1(0.3))
        self.assertAlmostEqual(row["JR_pos"], np.expm1(0.1))
        self.assertAlmostEqual(row["JR_neg"], np.expm1(-0.2))

    def test_partial_or_duplicate_minute_days_cannot_be_treated_as_complete(self):
        day = datetime(2024, 1, 2)
        times = [
            day.replace(hour=h, minute=m) + timedelta(minutes=step)
            for h, m in [(9, 30), (13, 0)]
            for step in range(5, 121, 5)
        ]
        frame = pl.DataFrame(
            {
                "symbol": ["A"] * len(times),
                "ts": times,
                "open": [10.0] * len(times),
                "close": [10.0] * len(times),
                "volume": [1.0] * len(times),
            }
        )
        self.assertEqual(complete_trading_days(frame, 5).height, 48)
        self.assertEqual(complete_trading_days(frame.head(47), 5).height, 0)
        self.assertEqual(
            complete_trading_days(pl.concat([frame, frame.head(1)]), 5).height, 0
        )

    def test_signal_eligibility_uses_exchange_age_and_valid_daily_coverage(self):
        calendar = pd.bdate_range("2023-10-02", "2024-02-29")
        month = pd.Timestamp("2024-02-01")
        n_days = int((calendar.to_period("M") == month.to_period("M")).sum())
        factors = pl.DataFrame(
            {
                "symbol": ["OLD", "NEW", "THIN"],
                "month": [month.date()] * 3,
                "D": [n_days, n_days, 1],
                "JArr": [0.2, 0.3, 0.4],
            }
        )
        stocks = pd.DataFrame(
            {
                "listing_date": [calendar[0], calendar[-59], calendar[0]],
                "delisting_date": [pd.NaT] * 3,
            },
            index=["OLD", "NEW", "THIN"],
        )
        result = eligible_monthly_factors(
            factors, stocks, calendar, data_first="2024-01-02", data_last="2024-02-29"
        )
        self.assertEqual(result["symbol"].to_list(), ["OLD"])
        self.assertEqual(result["date"].item(), date(2024, 2, 29))
        self.assertGreaterEqual(result["listing_age"].item(), 60)


if __name__ == "__main__":
    unittest.main()
