import random
import unittest

import polars as pl

from factor_test.portfolio import backtest_portfolio


def quotes(values):
    return pl.DataFrame(
        {
            "date": range(1, len(values) + 1),
            "symbol": ["A"] * len(values),
            "open": values,
        }
    ).lazy()


def targets(days, values=None):
    return pl.DataFrame(
        {
            "date": days,
            "symbol": ["A"] * len(days),
            "weight": values if values is not None else [1.0] * len(days),
        }
    ).lazy()


class IntegerShareTests(unittest.TestCase):
    def test_rounding_leaves_cash_and_uses_integer_schemas(self):
        result = backtest_portfolio(targets([1]), quotes([30.0] * 3), initial_value=100)
        self.assertEqual(result.trades["quantity"].to_list(), [3])
        self.assertEqual(result.daily["cash"].to_list(), [100, 10, 10])
        for frame in (result.positions, result.trades, result.unfilled_orders):
            self.assertEqual(frame.schema["quantity"], pl.Int64)

    def test_rounding_short_positions_is_toward_zero(self):
        result = backtest_portfolio(
            targets([1], [-1.0]), quotes([30.0, 30, 24]), initial_value=100
        )
        self.assertEqual(result.trades["quantity"].to_list(), [-3])
        self.assertEqual(result.daily["cash"].to_list(), [100, 190, 190])
        self.assertEqual(result.daily["equity"].to_list(), [100, 100, 118])

    def test_insufficient_capital_never_trades(self):
        result = backtest_portfolio(targets([1]), quotes([100.0] * 3))
        self.assertTrue(result.trades.is_empty())
        self.assertTrue(result.positions.is_empty())
        self.assertEqual(result.daily["cash"].to_list(), [1.0] * 3)

    def test_rounding_remainder_is_not_redistributed(self):
        weights = pl.DataFrame(
            {"date": [1, 1], "symbol": ["A", "B"], "weight": [0.5, 0.5]}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2] * 2,
                "symbol": ["A", "A", "B", "B"],
                "open": [60.0, 60, 20, 20],
            }
        ).lazy()
        result = backtest_portfolio(weights, prices, initial_value=100)
        self.assertEqual(result.trades["symbol"].to_list(), ["B"])
        self.assertEqual(result.trades["quantity"].to_list(), [2])
        self.assertEqual(result.daily["cash"][-1], 60)

    def test_opening_fees_are_reserved_before_rounding(self):
        result = backtest_portfolio(
            targets([1]), quotes([100.0] * 2), initial_value=1000, commission_rate=0.01
        )
        self.assertEqual(result.trades["quantity"].to_list(), [9])
        self.assertEqual(result.daily["cash"][-1], 91)
        self.assertEqual(result.daily["equity"][-1], 991)
        self.assertEqual(result.trades["commission"].to_list(), [9])

    def test_capital_for_one_share_must_also_cover_fees(self):
        result = backtest_portfolio(
            targets([1]), quotes([100.0] * 2), initial_value=100, commission_rate=0.01
        )
        self.assertTrue(result.trades.is_empty())
        self.assertEqual(result.daily["cash"][-1], 100)

    def test_exact_capital_including_fees_buys_one_share(self):
        result = backtest_portfolio(
            targets([1]), quotes([100.0] * 2), initial_value=101, commission_rate=0.01
        )
        self.assertEqual(result.trades["quantity"].to_list(), [1])
        self.assertAlmostEqual(result.daily["cash"][-1], 0)

    def test_exit_costs_and_renewal_use_actual_share_differences(self):
        result = backtest_portfolio(
            targets([1, 2]),
            quotes([100.0] * 4),
            initial_value=1000,
            rebalance_every=1,
            commission_rate=0.01,
            sell_tax_rate=0.02,
        )
        self.assertEqual(result.trades["quantity"].to_list(), [9, -9])
        self.assertEqual(result.trades["date"].to_list(), [2, 4])
        self.assertEqual(result.daily["cash"][-1], 964)
        self.assertEqual(result.trades["commission"].to_list(), [9, 9])
        self.assertEqual(result.trades["sell_tax"].to_list(), [0, 18])
        self.assertEqual(result.trades["cost"].to_list(), [9, 27])

    def test_rebalance_trades_only_integer_difference(self):
        result = backtest_portfolio(
            targets([1, 2], [0.3, 0.5]),
            quotes([100.0] * 3),
            initial_value=1000,
            rebalance_every=1,
        )
        self.assertEqual(result.trades["quantity"].to_list(), [3, 2])

    def test_default_schedule_and_missing_signal_keep_original_anchor(self):
        result = backtest_portfolio(
            targets([1, 7, 11]), quotes([100.0] * 13), initial_value=1000
        )
        self.assertEqual(result.trades["date"].to_list(), [2, 7, 12])
        self.assertEqual(result.trades["quantity"].to_list(), [10, -10, 10])
        self.assertEqual(
            result.positions.filter(pl.col("date").is_between(7, 11)).height, 0
        )

    def test_fully_frozen_budget_waits_until_next_rebalance(self):
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
            weights, prices, initial_value=1000, suspended_col="suspended"
        )
        self.assertEqual(result.trades["date"].to_list(), [2, 12, 12])
        self.assertEqual(result.trades["symbol"].to_list(), ["A", "A", "C"])
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["exit_blocked"])

    def test_blocked_new_entry_redistributes_weight(self):
        weights = pl.DataFrame(
            {"date": [1, 1], "symbol": ["A", "B"], "weight": [0.5, 0.5]}
        ).lazy()
        prices = pl.DataFrame(
            {
                "date": [1, 2, 3] * 2,
                "symbol": ["A"] * 3 + ["B"] * 3,
                "open": [100.0] * 6,
                "suspended": [False] * 3 + [False, True, False],
            }
        ).lazy()
        result = backtest_portfolio(
            weights, prices, initial_value=1000, suspended_col="suspended"
        )
        self.assertEqual(result.trades["quantity"].to_list(), [10])
        self.assertEqual(result.daily["cash"][-1], 0)
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["entry_blocked"])

    def test_borrow_fees_use_previous_short_value(self):
        result = backtest_portfolio(
            targets([1], [-1.0]),
            quotes([100.0] * 4),
            initial_value=1000,
            short_borrow_rate=0.252,
            rebalance_every=3,
        )
        self.assertEqual(result.daily["borrow_cost"].to_list(), [0, 0, 1, 1])
        self.assertEqual(result.daily["equity"][-1], 998)

    def test_tail_and_position_recording_options(self):
        base = backtest_portfolio(targets([1]), quotes([100.0] * 3), initial_value=1000)
        compact = backtest_portfolio(
            targets([1]),
            quotes([100.0] * 3),
            initial_value=1000,
            record_positions=False,
        )
        closed = backtest_portfolio(
            targets([1]), quotes([100.0] * 3), initial_value=1000, liquidate_at_end=True
        )
        self.assertEqual(base.daily.to_dicts(), compact.daily.to_dicts())
        self.assertEqual(base.trades.to_dicts(), compact.trades.to_dicts())
        self.assertTrue(compact.positions.is_empty())
        self.assertIsNone(base.positions["expiry_date"][-1])
        self.assertEqual(closed.trades["quantity"].to_list(), [10, -10])

    def test_future_data_cannot_change_past_trades(self):
        prices = quotes([100.0, 110, 121, 133, 146])
        weights = targets([1, 2, 3, 4])
        base = backtest_portfolio(
            weights, prices, initial_value=1000, rebalance_every=2
        )
        changed = backtest_portfolio(
            weights.with_columns(
                pl.when(pl.col("date") >= 3)
                .then(0.0)
                .otherwise(pl.col("weight"))
                .alias("weight")
            ),
            prices.with_columns(
                pl.when(pl.col("date") > 3)
                .then(2000.0)
                .otherwise(pl.col("open"))
                .alias("open")
            ),
            initial_value=1000,
            rebalance_every=2,
        )
        self.assertEqual(
            base.daily.head(3).to_dicts(), changed.daily.head(3).to_dicts()
        )

    def test_randomized_multi_asset_accounting_and_cash(self):
        rng = random.Random(24)
        for long_short in (False, True):
            with self.subTest(long_short=long_short):
                prices_rows = []
                levels = dict.fromkeys("ABC", 100.0)
                for day in range(1, 31):
                    for symbol in "ABC":
                        levels[symbol] *= rng.uniform(0.95, 1.05)
                        prices_rows.append(
                            dict(date=day, symbol=symbol, open=levels[symbol])
                        )
                prices = pl.DataFrame(prices_rows)
                rows = []
                for day in range(1, 30):
                    raw = [rng.uniform(-1 if long_short else 0, 1) for _ in "ABC"]
                    gross = sum(map(abs, raw))
                    rows.extend(
                        dict(date=day, symbol=symbol, weight=value / gross * 0.99)
                        for symbol, value in zip("ABC", raw)
                    )
                result = backtest_portfolio(
                    pl.DataFrame(rows).lazy(),
                    prices.lazy(),
                    initial_value=10000,
                    rebalance_every=1,
                    commission_rate=0.003,
                    slippage_rate=0.002,
                    sell_tax_rate=0.001,
                    short_borrow_rate=0.05,
                )
                price_map = {
                    (r["date"], r["symbol"]): r["open"]
                    for r in prices.iter_rows(named=True)
                }
                previous_equity, previous_positions = 10000, {}
                for row in result.daily.iter_rows(named=True):
                    day = row["date"]
                    current = result.positions.filter(pl.col("date") == day)
                    pnl = (
                        sum(
                            qty * (price_map[day, symbol] - price_map[day - 1, symbol])
                            for symbol, qty in previous_positions.items()
                        )
                        if day > 1
                        else 0
                    )
                    self.assertGreaterEqual(row["cash"], -1e-8)
                    self.assertAlmostEqual(
                        row["equity"], row["cash"] + current["value"].sum()
                    )
                    self.assertAlmostEqual(
                        row["equity"] - previous_equity, pnl - row["cost"]
                    )
                    self.assertAlmostEqual(
                        row["net_return"], row["equity"] / previous_equity - 1
                    )
                    previous_equity = row["equity"]
                    previous_positions = dict(
                        current.select("symbol", "quantity").iter_rows()
                    )


if __name__ == "__main__":
    unittest.main()
