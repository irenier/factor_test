"""Integer sizing boundaries must respect both existing shares and actual fees."""

from math import fsum, nextafter
import unittest

import polars as pl

from factor_test.portfolio import _target_shares, _trade_fees, backtest_portfolio


class SizingTests(unittest.TestCase):
    def test_fully_invested_renewal_does_not_sell_a_share_due_to_roundoff(self):
        for price, quantity in [(0.3, 1), (1.1, 3), (1.1, 100), (133.1, 1)]:
            for direction in (1, -1):
                with self.subTest(price=price, quantity=quantity, direction=direction):
                    shares = {"A": quantity * direction}
                    target = _target_shares(
                        {"A": float(direction)},
                        {"A": price},
                        shares,
                        price * quantity,
                        (0.01, 0.0, 0.0),
                    )
                    self.assertEqual(target, shares)

    def test_exact_funding_including_fees_buys_an_integer_share(self):
        for price in (0.3, 1.1, 100.0, 133.1):
            for direction in (1, -1):
                with self.subTest(price=price, direction=direction):
                    rates = (0.01, 0.002, 0.003)
                    equity = price + fsum(_trade_fees(direction * price, rates))
                    self.assertEqual(
                        _target_shares(
                            {"A": float(direction)},
                            {"A": price},
                            {},
                            equity,
                            rates,
                        ),
                        {"A": direction},
                    )

    def test_roundoff_correction_cannot_spend_more_than_available(self):
        for direction in (1, -1):
            rates = (0.01, 0.002, 0.003)
            required = 100.0 + fsum(_trade_fees(direction * 100.0, rates))
            with self.subTest(direction=direction):
                self.assertEqual(
                    _target_shares(
                        {"A": float(direction)},
                        {"A": 100.0},
                        {},
                        nextafter(required, 0.0),
                        rates,
                    ),
                    {},
                )

    def test_multi_asset_targets_cover_actual_integer_trade_fees(self):
        prices = {"A": 0.3, "B": 1.1, "C": 133.1}
        previous = {"A": 50, "B": -10}
        rates = (0.01, 0.002, 0.003)
        for equity in (20.0, 100.0, 1000.0):
            with self.subTest(equity=equity):
                target = _target_shares(
                    {"A": -0.2, "B": 0.3, "C": 0.5},
                    prices,
                    previous,
                    equity,
                    rates,
                )
                gross = fsum(
                    abs(quantity) * prices[symbol]
                    for symbol, quantity in target.items()
                )
                fees = fsum(
                    fsum(
                        _trade_fees(
                            (target.get(symbol, 0) - previous.get(symbol, 0))
                            * prices[symbol],
                            rates,
                        )
                    )
                    for symbol in previous.keys() | target.keys()
                )
                self.assertLessEqual(gross + fees, equity)
                self.assertTrue(
                    all(type(quantity) is int for quantity in target.values())
                )

    def test_public_backtest_keeps_exactly_funded_share_on_renewal(self):
        for price in (0.3, 1.1, 133.1):
            with self.subTest(price=price):
                weights = pl.DataFrame(
                    {
                        "date": [1, 2],
                        "symbol": ["A", "A"],
                        "weight": [1.0, 1.0],
                    }
                ).lazy()
                prices = pl.DataFrame(
                    {
                        "date": [1, 2],
                        "symbol": ["A", "A"],
                        "open": [price, price],
                    }
                ).lazy()
                result = backtest_portfolio(
                    weights,
                    prices,
                    rebalance_every=1,
                    entry_lag=0,
                    initial_value=price + price * 0.01,
                    commission_rate=0.01,
                )
                self.assertEqual(result.trades["quantity"].to_list(), [1])
                self.assertTrue(all(cash >= 0.0 for cash in result.daily["cash"]))

    def test_trade_sum_roundoff_does_not_create_negative_cash(self):
        prices = {"A": 18.98, "B": 0.45, "C": 8.53}
        total = sum(prices.values())
        weights = pl.DataFrame(
            [
                dict(date=1, symbol=symbol, weight=price / total)
                for symbol, price in prices.items()
            ]
        ).lazy()
        quotes = pl.DataFrame(
            [
                dict(date=1, symbol=symbol, open=price)
                for symbol, price in prices.items()
            ]
        ).lazy()
        result = backtest_portfolio(
            weights,
            quotes,
            initial_value=28.2396,
            entry_lag=0,
            commission_rate=0.01,
        )
        self.assertEqual(result.trades["quantity"].to_list(), [1, 1, 1])
        self.assertEqual(result.daily["cash"].to_list(), [0.0])


if __name__ == "__main__":
    unittest.main()
