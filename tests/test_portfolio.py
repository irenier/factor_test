from datetime import date, timedelta
import math
import unittest

import polars as pl

from factor_test.portfolio import backtest_portfolio, portfolio_metrics


def single_prices(values):
    return pl.DataFrame(
        {
            "date": range(1, len(values) + 1),
            "symbol": ["A"] * len(values),
            "open": [float(x) for x in values],
        }
    ).lazy()


def single_weights(days):
    return pl.DataFrame(
        {"date": days, "symbol": ["A"] * len(days), "weight": [1.0] * len(days)}
    ).lazy()


class AccountingTests(unittest.TestCase):
    def assert_values(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y)

    def test_default_five_day_schedule_uses_full_capital_without_overlap(self):
        # Scheduled signal days 1/6/11; execution days 2/7/12.
        weights = pl.DataFrame(
            {
                "date": range(1, 14),
                "symbol": ["A"] + ["D"] * 4 + ["B"] + ["D"] * 4 + ["C"] + ["D"] * 2,
                "weight": [1.0] * 13,
            }
        ).lazy()
        prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=100.0)
                for day in range(1, 14)
                for symbol in "ABCD"
            ]
        ).lazy()
        result = backtest_portfolio(weights, prices, initial_value=1000)
        self.assertEqual(result.trades["date"].to_list(), [2, 7, 7, 12, 12])
        self.assertEqual(result.trades["symbol"].to_list(), ["A", "A", "B", "B", "C"])
        self.assertAlmostEqual(result.daily["cash"][1], 0.0)
        for day, symbol in [(2, "A"), (6, "A"), (7, "B"), (11, "B"), (12, "C")]:
            self.assertEqual(
                result.positions.filter(pl.col("date") == day)["symbol"].to_list(),
                [symbol],
            )

    def test_non_overlapping_schedule(self):
        result = backtest_portfolio(
            single_weights([1, 2, 3, 4, 5]),
            single_prices([100, 110, 121, 133.1, 146.41]),
            rebalance_every=2,
            initial_value=1000,
        )
        self.assert_values(result.daily["nav"], [1, 1, 1.099, 1.2079, 1.32769])

    def test_five_day_endpoints_include_integer_rounding_cash(self):
        prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=100 * growth**day)
                for day in range(1, 14)
                for symbol, growth in [("A", 1.01), ("B", 0.995)]
            ]
        ).lazy()
        weights = pl.DataFrame(
            {"date": [1, 6, 11], "symbol": ["A", "B", "A"], "weight": [1.0] * 3}
        ).lazy()
        result = backtest_portfolio(weights, prices, initial_value=1000)
        a_quantity = int(1000 / (100 * 1.01**2))
        equity_7 = 1000 + a_quantity * 100 * (1.01**7 - 1.01**2)
        b_quantity = int(equity_7 / (100 * 0.995**7))
        equity_12 = equity_7 + b_quantity * 100 * (0.995**12 - 0.995**7)
        self.assertAlmostEqual(
            result.daily.filter(pl.col("date") == 7)["equity"][0], equity_7
        )
        self.assertAlmostEqual(
            result.daily.filter(pl.col("date") == 12)["equity"][0], equity_12
        )
        for first, last, symbol in [(2, 6, "A"), (7, 11, "B")]:
            held = result.positions.filter(pl.col("date").is_between(first, last))
            self.assertEqual(held["symbol"].unique().to_list(), [symbol])
            self.assertEqual(held["quantity"].n_unique(), 1)

    def test_five_days_count_market_intervals_across_weekends(self):
        days = [
            date(2026, 10, 19) + timedelta(days=i)
            for i in range(10)
            if (date(2026, 10, 19) + timedelta(days=i)).weekday() < 5
        ]
        weights = pl.DataFrame(
            {"date": [days[0], days[5]], "symbol": ["A", "B"], "weight": [1.0, 1.0]}
        ).lazy()
        prices = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=100.0)
                for day in days
                for symbol in "AB"
            ]
        ).lazy()
        result = backtest_portfolio(weights, prices, initial_value=1000)
        self.assertEqual(
            result.trades["date"].to_list(),
            [date(2026, 10, 20), date(2026, 10, 27), date(2026, 10, 27)],
        )

    def test_missing_scheduled_signal_exits_to_cash_until_next_interval(self):
        # Signal day 6 is missing; day 7's signal must not be replayed.
        result = backtest_portfolio(
            single_weights([1, 7, 11]), single_prices([100] * 13), initial_value=1000
        )
        self.assert_values(result.daily["nav"], [1] * 13)
        self.assertEqual(result.trades["date"].to_list(), [2, 7, 12])
        for day in range(7, 12):
            self.assertEqual(result.positions.filter(pl.col("date") == day).height, 0)
            self.assertAlmostEqual(
                result.daily.filter(pl.col("date") == day)["cash"][0], 1000
            )

    def test_positions_drift_with_prices_instead_of_daily_rebalancing(self):
        weights = pl.DataFrame(
            {"date": [1, 1], "symbol": ["A", "B"], "weight": [0.5, 0.5]}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2, 3, 4] * 2,
                "symbol": ["A"] * 4 + ["B"] * 4,
                "open": [100.0, 100, 200, 100, 100, 100, 100, 100],
            }
        ).lazy()
        result = backtest_portfolio(
            weights, prices, rebalance_every=3, initial_value=1000
        )
        self.assert_values(result.daily["equity"], [1000, 1000, 1500, 1000])
        third = result.positions.filter(
            (pl.col("date") == 3) & (pl.col("symbol") == "A")
        )
        self.assertAlmostEqual(third["weight"][0], 2 / 3)
        self.assertEqual(result.trades.height, 2)

    def test_short_positions_and_cash_proceeds(self):
        weights = pl.DataFrame(
            {"date": [1, 1], "symbol": ["A", "B"], "weight": [0.5, -0.5]}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2, 3] * 2,
                "symbol": ["A"] * 3 + ["B"] * 3,
                "open": [100.0, 100, 110, 100, 100, 90],
            }
        ).lazy()
        result = backtest_portfolio(
            weights, prices, rebalance_every=1, initial_value=1000
        )
        self.assert_values(result.daily["equity"], [1000, 1000, 1100])
        self.assert_values(result.daily["cash"], [1000, 1000, 1100])
        self.assertEqual(result.positions.filter(pl.col("date") == 3).height, 0)

    def test_opening_fee_is_funded_without_negative_long_only_cash(self):
        result = backtest_portfolio(
            single_weights([1]),
            single_prices([100, 100]),
            rebalance_every=1,
            initial_value=1000,
            commission_rate=0.01,
        )
        self.assertAlmostEqual(result.daily["equity"][-1], 991)
        self.assertGreaterEqual(result.daily["cash"][-1], -1e-10)
        self.assertAlmostEqual(result.trades["commission"][0], 9)

    def test_exit_fee_and_sell_tax(self):
        result = backtest_portfolio(
            single_weights([1]),
            single_prices([100] * 3),
            rebalance_every=1,
            initial_value=1000,
            commission_rate=0.01,
            sell_tax_rate=0.02,
        )
        self.assertAlmostEqual(result.daily["equity"][-1], 964)
        self.assertAlmostEqual(result.trades["sell_tax"].sum(), 18)
        self.assertAlmostEqual(
            result.daily["equity"][-1] + result.daily["cost"].sum(), 1000
        )

    def test_same_asset_renewal_does_not_generate_commission(self):
        result = backtest_portfolio(
            single_weights([1, 2, 3]),
            single_prices([100] * 4),
            rebalance_every=1,
            commission_rate=0.01,
            initial_value=1000,
        )
        self.assertEqual(result.trades.height, 1)
        self.assertEqual(result.daily["turnover"].to_list()[2:], [0.0, 0.0])
        self.assertAlmostEqual(result.daily["equity"][-1], 991)

    def test_short_borrow_is_deducted_each_holding_interval(self):
        weights = pl.DataFrame({"date": [1], "symbol": ["A"], "weight": [-1.0]}).lazy()
        result = backtest_portfolio(
            weights,
            single_prices([100] * 4),
            rebalance_every=3,
            initial_value=100,
            short_borrow_rate=0.252,
            periods_per_year=252,
        )
        self.assertAlmostEqual(result.daily["borrow_cost"].sum(), 0.2)
        self.assertAlmostEqual(result.daily["equity"][-1], 99.8)

    def test_future_prices_and_signals_cannot_change_the_past(self):
        prices = single_prices([100, 110, 121, 133.1, 146.41])
        weights = single_weights([1, 2, 3, 4])
        base = backtest_portfolio(
            weights, prices, rebalance_every=2, initial_value=1000
        )
        altered_prices = prices.with_columns(
            pl.when(pl.col("date") > 3)
            .then(2000.0)
            .otherwise(pl.col("open"))
            .alias("open")
        )
        altered_weights = weights.with_columns(
            pl.when(pl.col("date") >= 3)
            .then(0.0)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        altered = backtest_portfolio(
            altered_weights, altered_prices, rebalance_every=2, initial_value=1000
        )
        self.assertEqual(
            base.daily.head(3).to_dicts(), altered.daily.head(3).to_dicts()
        )
        self.assertEqual(
            base.trades.filter(pl.col("date") <= 3).to_dicts(),
            altered.trades.filter(pl.col("date") <= 3).to_dicts(),
        )

    def test_snapshot_switch_does_not_change_accounting(self):
        prices = single_prices([100, 110, 121, 133.1])
        weights = single_weights([1, 2, 3])
        full = backtest_portfolio(
            weights, prices, rebalance_every=2, initial_value=1000
        )
        compact = backtest_portfolio(
            weights,
            prices,
            rebalance_every=2,
            record_positions=False,
            initial_value=1000,
        )
        self.assertEqual(full.daily.to_dicts(), compact.daily.to_dicts())
        self.assertEqual(full.trades.to_dicts(), compact.trades.to_dicts())
        self.assertTrue(compact.positions.is_empty())

    def test_capital_below_one_share_stays_in_cash(self):
        result = backtest_portfolio(
            single_weights([1]),
            single_prices([100, 100, 110]),
            rebalance_every=1,
            initial_value=1e-15,
        )
        self.assertAlmostEqual(result.daily["nav"][-1], 1.0)
        self.assertTrue(result.trades.is_empty())

    def test_tail_cohort_is_not_discarded_using_future_exit_availability(self):
        result = backtest_portfolio(
            single_weights([3]),
            single_prices([100] * 4),
            rebalance_every=5,
            initial_value=1000,
        )
        self.assertEqual(result.positions.filter(pl.col("date") == 4).height, 1)
        self.assertAlmostEqual(result.daily["cash"][-1], 0.0)
        self.assertIsNone(result.positions["expiry_date"][-1])
        closed = backtest_portfolio(
            single_weights([1]),
            single_prices([100] * 4),
            rebalance_every=5,
            liquidate_at_end=True,
            initial_value=1000,
        )
        self.assertEqual(closed.positions.filter(pl.col("date") == 4).height, 0)

    def test_cash_holdings_and_daily_return_reconciliation(self):
        prices = pl.DataFrame(
            {
                "date": list(range(1, 7)) * 2,
                "symbol": ["A"] * 6 + ["B"] * 6,
                "open": [100.0, 110, 90, 120, 115, 130, 100, 95, 105, 90, 99, 97],
            }
        ).lazy()
        weights = pl.DataFrame(
            {
                "date": [1, 1, 2, 2, 3, 3, 4, 4],
                "symbol": ["A", "B"] * 4,
                "weight": [0.6, -0.4, -0.2, 0.8, 0.5, -0.5, 0.8, 0.2],
            }
        ).lazy()
        result = backtest_portfolio(
            weights,
            prices,
            rebalance_every=2,
            initial_value=10000,
            commission_rate=0.001,
            slippage_rate=0.002,
            sell_tax_rate=0.001,
        )
        previous_equity = result.initial_value
        previous_positions = {}
        quote_map = {
            (r["date"], r["symbol"]): r["open"] for r in prices.collect().to_dicts()
        }
        nav = 1.0
        for row in result.daily.to_dicts():
            day = row["date"]
            positions = result.positions.filter(pl.col("date") == day)
            self.assertAlmostEqual(
                row["cash"] + positions["value"].sum(), row["equity"]
            )
            pnl = (
                sum(
                    qty * (quote_map[(day, symbol)] - quote_map[(day - 1, symbol)])
                    for symbol, qty in previous_positions.items()
                )
                if day > 1
                else 0
            )
            self.assertAlmostEqual(row["equity"] - previous_equity, pnl - row["cost"])
            self.assertAlmostEqual(
                row["gross_return"] - row["net_return"], row["cost"] / previous_equity
            )
            previous_equity = row["equity"]
            previous_positions = dict(
                positions.select("symbol", "quantity").iter_rows()
            )
            nav *= 1 + row["net_return"]
            self.assertAlmostEqual(nav, row["nav"])


class ExecutionTests(unittest.TestCase):
    def test_fully_frozen_portfolio_remains_until_next_scheduled_rebalance(self):
        weights = pl.DataFrame(
            {"date": [1, 6, 11], "symbol": ["A", "B", "C"], "weight": [1.0] * 3}
        ).lazy()
        prices = pl.DataFrame(
            [
                dict(
                    date=day,
                    symbol=symbol,
                    open=100.0,
                    suspended=symbol == "A" and day == 7,
                )
                for day in range(1, 14)
                for symbol in "ABC"
            ]
        ).lazy()
        result = backtest_portfolio(
            weights, prices, suspended_col="suspended", initial_value=1000
        )
        self.assertEqual(result.trades["date"].to_list(), [2, 12, 12])
        self.assertEqual(result.trades["symbol"].to_list(), ["A", "A", "C"])
        for day in range(8, 12):
            self.assertEqual(
                result.positions.filter(pl.col("date") == day)["symbol"].to_list(),
                ["A"],
            )

    def test_missing_entry_redistributes_to_tradable_targets(self):
        weights = pl.DataFrame(
            {"date": [1, 1], "symbol": ["A", "B"], "weight": [0.5, 0.5]}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2, 3, 1, 3],
                "symbol": ["A", "A", "A", "B", "B"],
                "open": [100.0, 100, 110, 100, 200],
            }
        ).lazy()
        result = backtest_portfolio(
            weights, prices, rebalance_every=1, initial_value=1000
        )
        self.assertAlmostEqual(result.daily["cash"][1], 0)
        self.assertAlmostEqual(result.daily["equity"][-1], 1100)
        self.assertEqual(result.trades.filter(pl.col("symbol") == "B").height, 0)
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["entry_blocked"])

    def test_missing_held_price_requires_an_explicit_stale_valuation_policy(self):
        prices = pl.DataFrame(
            {"date": [1, 2, 3, 4], "symbol": ["A"] * 4, "open": [100.0, 100, None, 110]}
        ).lazy()
        with self.assertRaises(ValueError):
            backtest_portfolio(
                single_weights([1]), prices, rebalance_every=1, initial_value=1000
            )
        result = backtest_portfolio(
            single_weights([1]),
            prices,
            rebalance_every=1,
            missing_price="carry",
            initial_value=1000,
        )
        self.assertEqual(result.daily["stale_positions"].to_list(), [0, 0, 1, 0])
        self.assertEqual(result.trades["date"].to_list(), [2, 4])
        self.assertAlmostEqual(result.daily["equity"][-1], 1100)
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["exit_blocked"])

    def test_blocked_exit_with_no_remaining_budget_retains_old_position(self):
        weights = pl.DataFrame(
            {"date": [1, 2, 3], "symbol": ["A", "B", "B"], "weight": [1.0] * 3}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2, 3, 4] * 2,
                "symbol": ["A"] * 4 + ["B"] * 4,
                "open": [100.0] * 8,
                "suspended": [False, False, True, False] + [False] * 4,
            }
        ).lazy()
        result = backtest_portfolio(
            weights,
            prices,
            rebalance_every=1,
            suspended_col="suspended",
            initial_value=1000,
        )
        self.assertEqual(
            result.positions.filter(pl.col("date") == 3)["symbol"].to_list(), ["A"]
        )
        self.assertEqual(
            result.positions.filter(pl.col("date") == 4)["symbol"].to_list(), ["B"]
        )
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["exit_blocked"])
        self.assertEqual(result.trades.filter(pl.col("date") == 3).height, 0)

    def test_suspended_quote_waits_for_resumption_to_update_valuation(self):
        prices = single_prices([100, 100, 120, 125]).with_columns(
            (pl.col("date") == 3).alias("suspended")
        )
        result = backtest_portfolio(
            single_weights([1]),
            prices,
            rebalance_every=1,
            suspended_col="suspended",
            initial_value=1000,
        )
        self.assertEqual(result.daily["equity"].to_list(), [1000, 1000, 1000, 1250])
        self.assertEqual(result.daily["stale_positions"][2], 1)
        self.assertEqual(result.trades["date"].to_list(), [2, 4])

    def test_real_dates_and_empty_signal_portfolio(self):
        from factor_test.analyzer import portfolio_returns

        days = [date(2026, 10, 19) + timedelta(days=i) for i in range(4)]
        prices = pl.DataFrame(
            {"date": days, "symbol": ["A"] * 4, "open": [100.0] * 4}
        ).lazy()
        signals = pl.DataFrame(
            {"date": days, "symbol": ["A"] * 4, "factor": [None] * 4},
            schema_overrides={"factor": pl.Float64},
        ).lazy()
        result = portfolio_returns(signals, prices, rebalance_every=2)
        self.assertEqual(result.daily["nav"].to_list(), [1.0] * 4)
        self.assertTrue(result.trades.is_empty())
        self.assertEqual(result.positions.schema["date"], pl.Date)

    def test_invalid_and_leveraged_inputs_are_rejected(self):
        prices = single_prices([100, 100, 100])
        for options in [
            dict(rebalance_every=0),
            dict(entry_lag=-1),
            dict(commission_rate=-0.1),
            dict(commission_rate=0.5, slippage_rate=0.5),
            dict(initial_value=0),
            dict(missing_price="fill"),
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                backtest_portfolio(single_weights([1]), prices, **options)
        with self.assertRaises(ValueError):
            backtest_portfolio(
                single_weights([1]).with_columns(pl.lit(1.1).alias("weight")),
                prices,
                initial_value=1000,
            )
        with self.assertRaises(ValueError):
            backtest_portfolio(
                single_weights([1]).with_columns(pl.lit(float("nan")).alias("weight")),
                prices,
                initial_value=1000,
            )


class UtilityTests(unittest.TestCase):
    def test_daily_and_five_day_modes_share_one_public_implementation(self):
        from factor_test.analyzer import (
            portfolio_returns,
        )

        prices = single_prices([100, 100, 110, 105, 120, 125, 130, 140])
        factors = pl.DataFrame(
            {"date": list(range(1, 9)), "symbol": ["A"] * 8, "factor": [1.0] * 8}
        ).lazy()
        for interval in [1, 5]:
            with self.subTest(interval=interval):
                result = portfolio_returns(
                    factors,
                    prices,
                    weight_method="equal",
                    long_short=False,
                    rebalance_every=interval,
                    commission_rate=0.01,
                    initial_value=1000,
                )
                expected = backtest_portfolio(
                    single_weights(list(range(1, 9))),
                    prices,
                    rebalance_every=interval,
                    commission_rate=0.01,
                    initial_value=1000,
                )
                self.assertEqual(result.daily.to_dicts(), expected.daily.to_dicts())
                self.assertEqual(result.trades.to_dicts(), expected.trades.to_dicts())
        combined = factors.join(prices, on=["date", "symbol"])
        default = portfolio_returns(
            combined, weight_method="equal", long_short=False, initial_value=1000
        )
        five = portfolio_returns(
            factors,
            prices,
            weight_method="equal",
            long_short=False,
            rebalance_every=5,
            initial_value=1000,
        )
        self.assertEqual(default.daily.to_dicts(), five.daily.to_dicts())

    def test_trade_fees_agree_with_account_ledger(self):
        result = backtest_portfolio(
            single_weights([1]),
            single_prices([100, 100, 100]),
            rebalance_every=1,
            commission_rate=0.001,
            slippage_rate=0.002,
            sell_tax_rate=0.003,
            initial_value=1000,
        )
        for row in result.trades.iter_rows(named=True):
            self.assertAlmostEqual(row["commission"], abs(row["notional"]) * 0.001)
            self.assertAlmostEqual(row["slippage"], abs(row["notional"]) * 0.002)
            self.assertAlmostEqual(row["sell_tax"], max(-row["notional"], 0) * 0.003)
            self.assertAlmostEqual(
                row["cost"], row["commission"] + row["slippage"] + row["sell_tax"]
            )
        self.assertAlmostEqual(result.daily["cost"].sum(), result.trades["cost"].sum())

    def test_metrics_and_benchmark(self):
        result = backtest_portfolio(
            single_weights([1]),
            single_prices([100, 100, 110, 99]),
            rebalance_every=3,
            initial_value=1000,
        )
        benchmark = pl.DataFrame(
            {"date": [1, 2, 3, 4], "benchmark_return": [0.0, 0, 0.1, -0.1]}
        ).lazy()
        metrics = portfolio_metrics(result, benchmark=benchmark).row(0, named=True)
        self.assertAlmostEqual(metrics["total_return"], -0.01)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["beta"], 1.0)
        self.assertAlmostEqual(metrics["relative_return"], 0.0)
        self.assertTrue(math.isfinite(metrics["sharpe"]))
        with self.assertRaises(ValueError):
            portfolio_metrics(result, benchmark=benchmark.filter(pl.col("date") != 2))


if __name__ == "__main__":
    unittest.main()
