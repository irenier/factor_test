import unittest

import polars as pl

from factor_test.portfolio import _rebalance, backtest_portfolio


def weights(rows):
    return pl.DataFrame(rows).lazy()


def quotes(rows):
    return pl.DataFrame(rows).lazy()


class SuspensionTests(unittest.TestCase):
    def test_suspended_and_null_flags_keep_last_price_even_with_a_quote(self):
        result = backtest_portfolio(
            weights([dict(date=1, symbol="A", weight=1.0)]),
            quotes(
                [
                    dict(date=1, symbol="A", open=100.0, suspended=False),
                    dict(date=2, symbol="A", open=5000.0, suspended=True),
                    dict(date=3, symbol="A", open=7000.0, suspended=None),
                    dict(date=4, symbol="A", open=120.0, suspended=False),
                ]
            ),
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(result.daily["equity"].to_list(), [1000, 1000, 1000, 1200])
        self.assertEqual(result.daily["suspended_positions"].to_list(), [0, 1, 1, 0])
        self.assertEqual(result.positions["price"].to_list(), [100, 100, 100, 120])
        self.assertEqual(
            result.positions["stale_price"].to_list(), [False, True, True, False]
        )
        self.assertEqual(
            result.positions["suspended"].to_list(), [False, True, True, False]
        )
        self.assertEqual(result.trades["quantity"].to_list(), [10])

    def test_suspended_position_with_no_quote_uses_last_price_by_default(self):
        result = backtest_portfolio(
            weights([dict(date=1, symbol="A", weight=1.0)]),
            quotes(
                [
                    dict(date=1, symbol="A", open=100.0, suspended=False),
                    dict(date=2, symbol="A", open=None, suspended=True),
                    dict(date=3, symbol="A", open=None, suspended=None),
                ]
            ),
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(result.daily["equity"].to_list(), [1000, 1000, 1000])
        self.assertEqual(result.daily["stale_positions"].to_list(), [0, 1, 1])

    def test_blocked_exit_freezes_old_position_and_funds_new_signal(self):
        signals = weights(
            [
                dict(date=1, symbol="A", weight=0.5),
                dict(date=1, symbol="B", weight=0.5),
                dict(date=2, symbol="C", weight=1.0),
                dict(date=4, symbol="D", weight=1.0),
            ]
        )
        prices = quotes(
            [
                dict(
                    date=day,
                    symbol=symbol,
                    open=100.0,
                    suspended=symbol == "A" and day == 2,
                )
                for day in range(1, 6)
                for symbol in "ABCD"
            ]
        )
        result = backtest_portfolio(
            signals,
            prices,
            rebalance_every=1,
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(
            result.trades.select("date", "symbol", "quantity").rows(),
            [
                (1, "A", 5),
                (1, "B", 5),
                (2, "B", -5),
                (2, "C", 5),
                (3, "A", -5),
                (3, "C", -5),
                (4, "D", 10),
                (5, "D", -10),
            ],
        )
        self.assertEqual(
            result.positions.filter(pl.col("date") == 2)["symbol"].to_list(), ["A", "C"]
        )
        self.assertTrue(result.positions.filter(pl.col("date") == 3).is_empty())
        self.assertEqual(result.daily["cash"].to_list(), [0, 0, 1000, 0, 1000])
        self.assertTrue(
            result.unfilled_orders.filter(pl.col("symbol") == "C").is_empty()
        )

    def test_suspended_new_position_is_skipped_without_catch_up(self):
        result = backtest_portfolio(
            weights(
                [
                    dict(date=1, symbol="A", weight=0.5),
                    dict(date=1, symbol="B", weight=0.5),
                ]
            ),
            quotes(
                [
                    dict(
                        date=day,
                        symbol=symbol,
                        open=None if symbol == "A" and day == 1 else 100.0,
                        suspended=symbol == "A" and day == 1,
                    )
                    for day in range(1, 4)
                    for symbol in "AB"
                ]
            ),
            rebalance_every=3,
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(
            result.trades.select("date", "symbol", "quantity").rows(), [(1, "B", 10)]
        )
        self.assertEqual(result.daily["cash"].to_list(), [0, 0, 0])
        self.assertEqual(result.daily["suspended_positions"].to_list(), [0, 0, 0])
        self.assertEqual(
            result.unfilled_orders.select("symbol", "reason").rows(),
            [("A", "entry_blocked")],
        )

    def test_frozen_position_leaves_remaining_budget_for_weighted_targets(self):
        result = backtest_portfolio(
            weights(
                [
                    dict(date=1, symbol="A", weight=0.4),
                    dict(date=1, symbol="B", weight=0.6),
                    dict(date=3, symbol="A", weight=-0.2),
                    dict(date=3, symbol="C", weight=0.2),
                    dict(date=3, symbol="D", weight=0.6),
                ]
            ),
            quotes(
                [
                    dict(
                        date=day,
                        symbol=symbol,
                        open=120.0 if symbol == "A" and day >= 4 else price,
                        suspended=symbol == "A" and day == 3,
                    )
                    for day in range(1, 6)
                    for symbol, price in [
                        ("A", 100.0),
                        ("B", 100.0),
                        ("C", 50.0),
                        ("D", 50.0),
                    ]
                ]
            ),
            rebalance_every=2,
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        held = result.positions.filter(pl.col("date") == 3)
        self.assertEqual(
            dict(held.select("symbol", "quantity").iter_rows()),
            {"A": 4, "C": 3, "D": 9},
        )
        self.assertEqual(result.daily.filter(pl.col("date") == 3)["cash"].item(), 0)
        # Resumption updates valuation but cannot liquidate the accepted new portfolio early.
        self.assertTrue(result.trades.filter(pl.col("date") == 4).is_empty())
        self.assertEqual(
            result.daily.filter(pl.col("date") == 4)["equity"].item(), 1080
        )
        self.assertTrue(result.positions.filter(pl.col("date") == 5).is_empty())

    def test_frozen_short_reserves_absolute_value_and_preserves_target_signs(self):
        result = backtest_portfolio(
            weights(
                [
                    dict(date=1, symbol="A", weight=-0.4),
                    dict(date=3, symbol="B", weight=0.2),
                    dict(date=3, symbol="C", weight=-0.6),
                ]
            ),
            quotes(
                [
                    dict(
                        date=day,
                        symbol=symbol,
                        open=50.0 if symbol != "A" else 100.0,
                        suspended=symbol == "A" and day == 3,
                    )
                    for day in range(1, 4)
                    for symbol in "ABC"
                ]
            ),
            rebalance_every=2,
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        held = result.positions.filter(pl.col("date") == 3)
        self.assertEqual(
            dict(held.select("symbol", "quantity").iter_rows()),
            {"A": -4, "B": 3, "C": -9},
        )
        row = result.daily.row(-1, named=True)
        self.assertEqual(row["cash"], 1700)
        self.assertEqual(row["equity"], 1000)
        self.assertEqual(row["gross_exposure"], 1)

    def test_remaining_budget_covers_exit_and_entry_fees(self):
        target, trades, rejected = _rebalance(
            {"A": 4, "B": 6},
            {"C": 1.0},
            dict.fromkeys("ABC", 100.0),
            {"B", "C"},
            1000,
            (0.01, 0.002, 0.003),
        )
        self.assertEqual(target, {"A": 4, "C": 5})
        self.assertEqual(
            [(row["symbol"], row["quantity"]) for row in trades], [("B", -6), ("C", 5)]
        )
        self.assertAlmostEqual(sum(row["cost"] for row in trades), 15)
        self.assertEqual([row["reason"] for row in rejected], ["exit_blocked"])

    def test_frozen_short_can_exceed_equity_after_borrow_without_new_trades(self):
        result = backtest_portfolio(
            weights(
                [
                    dict(date=1, symbol="A", weight=-1.0),
                    dict(date=2, symbol="B", weight=1.0),
                ]
            ),
            quotes(
                [
                    dict(
                        date=day,
                        symbol=symbol,
                        open=100.0,
                        suspended=symbol == "A" and day == 2,
                    )
                    for day in (1, 2)
                    for symbol in "AB"
                ]
            ),
            rebalance_every=1,
            entry_lag=0,
            initial_value=1000,
            short_borrow_rate=0.252,
            suspended_col="suspended",
        )
        self.assertEqual(
            result.trades.select("symbol", "quantity").rows(), [("A", -10)]
        )
        self.assertEqual(result.daily["equity"].to_list(), [1000, 999])

    def test_all_targets_blocked_keep_cash_and_all_zero_targets_exit_movable(self):
        target, trades, rejected = _rebalance(
            {},
            {"A": 1.0},
            {},
            set(),
            1000,
            (0.01, 0.0, 0.0),
        )
        self.assertEqual(target, {})
        self.assertEqual(trades, [])
        self.assertEqual(rejected[0]["reason"], "entry_blocked")
        target, trades, _ = _rebalance(
            {"A": 4, "B": 6},
            {"B": 0.0},
            dict.fromkeys("AB", 100.0),
            {"B"},
            1000,
            (0.01, 0.0, 0.0),
        )
        self.assertEqual(target, {"A": 4})
        self.assertEqual(trades[0]["quantity"], -6)

    def test_unrestricted_partial_weights_still_keep_intended_cash(self):
        target, _, _ = _rebalance(
            {},
            {"A": 0.2, "B": -0.4},
            dict.fromkeys("AB", 100.0),
            {"A", "B"},
            1000,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(target, {"A": 2, "B": -4})

    def test_fully_frozen_budget_does_not_restart_rebalance_calendar(self):
        result = backtest_portfolio(
            weights(
                [
                    dict(date=1, symbol="A", weight=1.0),
                    dict(date=6, symbol="B", weight=1.0),
                    dict(date=11, symbol="C", weight=1.0),
                ]
            ),
            quotes(
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
            ),
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(
            result.trades.select("date", "symbol", "quantity").rows(),
            [(2, "A", 10), (12, "A", -10), (12, "C", 10)],
        )
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["exit_blocked"])

    def test_suspended_multi_asset_account_reconciles_with_fees_and_borrow(self):
        price_rows = [
            dict(
                date=day,
                symbol=symbol,
                open=100.0 + day * (index + 1),
                suspended=day > 1 and (day + index) % 5 in (0, 1),
            )
            for day in range(1, 31)
            for index, symbol in enumerate("ABC")
        ]
        result = backtest_portfolio(
            weights(
                [
                    dict(date=day, symbol=symbol, weight=weight)
                    for day in range(1, 30, 2)
                    for symbol, weight in zip(
                        "ABC",
                        (0.4, -0.3, 0.3) if day % 4 == 1 else (-0.2, 0.5, -0.3),
                    )
                ]
            ),
            quotes(price_rows),
            rebalance_every=2,
            entry_lag=0,
            initial_value=10000,
            commission_rate=0.003,
            slippage_rate=0.002,
            sell_tax_rate=0.001,
            short_borrow_rate=0.1,
            suspended_col="suspended",
        )
        previous_equity, previous_shares, last_prices = 10000, {}, {}
        for row in result.daily.iter_rows(named=True):
            day = row["date"]
            old_prices = dict(last_prices)
            suspended = set()
            for quote in price_rows:
                if quote["date"] == day:
                    if quote["suspended"]:
                        suspended.add(quote["symbol"])
                    else:
                        last_prices[quote["symbol"]] = quote["open"]
            held = result.positions.filter(pl.col("date") == day)
            current = dict(held.select("symbol", "quantity").iter_rows())
            pnl = sum(
                qty * (last_prices[symbol] - old_prices[symbol])
                for symbol, qty in previous_shares.items()
            )
            self.assertAlmostEqual(row["equity"] - previous_equity, pnl - row["cost"])
            self.assertAlmostEqual(row["equity"], row["cash"] + held["value"].sum())
            self.assertGreaterEqual(row["cash"], 0)
            # Between rebalances, price drift can push gross exposure above one.
            if day % 2 == 1:
                self.assertLessEqual(held["value"].abs().sum(), row["equity"] + 1e-8)
            for symbol in suspended & previous_shares.keys():
                self.assertEqual(current[symbol], previous_shares[symbol])
            today = result.trades.filter(pl.col("date") == day)
            self.assertFalse(set(today["symbol"]) & suspended)
            previous_equity, previous_shares = row["equity"], current

    def test_suspended_short_cannot_cover_and_keeps_paying_borrow_fees(self):
        result = backtest_portfolio(
            weights([dict(date=1, symbol="A", weight=-1.0)]),
            quotes(
                [
                    dict(date=1, symbol="A", open=100.0, suspended=False),
                    dict(date=2, symbol="A", open=5000.0, suspended=True),
                    dict(date=3, symbol="A", open=None, suspended=True),
                    dict(date=4, symbol="A", open=80.0, suspended=False),
                ]
            ),
            rebalance_every=1,
            entry_lag=0,
            initial_value=1000,
            short_borrow_rate=0.252,
            suspended_col="suspended",
        )
        self.assertEqual(
            result.trades.select("date", "quantity").rows(), [(1, -10), (4, 10)]
        )
        self.assertEqual(result.daily["borrow_cost"].to_list(), [0, 1, 1, 1])
        self.assertEqual(result.daily["equity"].to_list(), [1000, 999, 998, 1197])
        self.assertEqual(result.daily["cash"][-1], 1197)

    def test_final_liquidation_cannot_trade_a_suspended_position(self):
        result = backtest_portfolio(
            weights([dict(date=1, symbol="A", weight=1.0)]),
            quotes(
                [
                    dict(date=1, symbol="A", open=100.0, suspended=False),
                    dict(date=2, symbol="A", open=9000.0, suspended=True),
                ]
            ),
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
            liquidate_at_end=True,
        )
        self.assertEqual(result.trades.select("date", "quantity").rows(), [(1, 10)])
        self.assertEqual(result.positions["quantity"][-1], 10)
        self.assertEqual(result.daily["equity"][-1], 1000)
        self.assertEqual(result.unfilled_orders["reason"].to_list(), ["exit_blocked"])

    def test_resumption_uses_current_quote_for_valuation_and_pending_exit(self):
        result = backtest_portfolio(
            weights([dict(date=1, symbol="A", weight=1.0)]),
            quotes(
                [
                    dict(date=1, symbol="A", open=100.0, suspended=False),
                    dict(date=2, symbol="A", open=9000.0, suspended=True),
                    dict(date=3, symbol="A", open=120.0, suspended=False),
                ]
            ),
            rebalance_every=1,
            entry_lag=0,
            initial_value=1000,
            suspended_col="suspended",
        )
        self.assertEqual(result.daily["equity"].to_list(), [1000, 1000, 1200])
        self.assertEqual(result.daily["stale_positions"].to_list(), [0, 1, 0])
        self.assertEqual(
            result.trades.select("date", "price", "quantity").rows(),
            [(1, 100, 10), (3, 120, -10)],
        )
        self.assertTrue(result.positions.filter(pl.col("date") == 3).is_empty())

    def test_missing_quote_without_explicit_suspension_keeps_existing_policy(self):
        signals = weights([dict(date=1, symbol="A", weight=1.0)])
        prices = quotes(
            [
                dict(date=1, symbol="A", open=100.0, suspended=False),
                dict(date=2, symbol="A", open=None, suspended=False),
            ]
        )
        options = dict(entry_lag=0, initial_value=1000, suspended_col="suspended")
        with self.assertRaisesRegex(ValueError, "已有持仓缺少有效报价"):
            backtest_portfolio(signals, prices, **options)
        carried = backtest_portfolio(signals, prices, missing_price="carry", **options)
        self.assertEqual(carried.daily["equity"].to_list(), [1000, 1000])
        self.assertEqual(carried.daily["stale_positions"].to_list(), [0, 1])
        self.assertEqual(carried.daily["suspended_positions"].to_list(), [0, 0])


if __name__ == "__main__":
    unittest.main()
