"""The factor subclass stays lazy, isolates symbols and sanitizes invalid prices."""

from datetime import date, timedelta
import unittest

import polars as pl

from factor_test.factor import BaseFactor
from example.bias_factor import BiasFactor


class BiasFactorTests(unittest.TestCase):
    def test_subclass_is_lazy_and_groups_unsorted_prices_by_symbol(self):
        data = (
            pl.DataFrame(
                [
                    dict(
                        date=date(2020, 1, 1) + timedelta(days=i),
                        symbol=symbol,
                        close=value,
                    )
                    for symbol, prices in [
                        ("A", [10.0, 20.0, 30.0]),
                        ("B", [4.0, 4.0, 4.0]),
                    ]
                    for i, value in enumerate(prices)
                ]
            )
            .reverse()
            .lazy()
        )
        factor = BiasFactor(window=3)
        self.assertIsInstance(factor, BaseFactor)
        result = factor.compute(data)
        self.assertIsInstance(result, pl.LazyFrame)
        output = result.collect()
        self.assertEqual(
            output.filter(pl.col("symbol") == "A")["factor"].to_list(),
            [None, None, 0.5],
        )
        self.assertEqual(
            output.filter(pl.col("symbol") == "B")["factor"].to_list(),
            [None, None, 0.0],
        )

    def test_invalid_prices_are_null_and_never_enter_the_rolling_mean(self):
        for invalid in [None, 0.0, -1.0, float("nan"), float("inf")]:
            data = pl.DataFrame(
                {
                    "date": [1, 2, 3, 4, 5],
                    "symbol": ["A"] * 5,
                    "close": [10.0, invalid, 10.0, 10.0, 10.0],
                }
            )
            values = (
                BiasFactor(window=3).compute(data.lazy()).collect()["factor"].to_list()
            )
            self.assertEqual(values, [None, None, None, None, 0.0])

    def test_mean_includes_current_price_and_ignores_future_prices(self):
        data = pl.DataFrame(
            {
                "date": list(range(22)),
                "symbol": ["A"] * 22,
                "close": [float(i + 1) for i in range(22)],
            }
        )
        first = BiasFactor().compute(data.lazy()).collect()
        changed = data.with_columns(
            pl.when(pl.col("date") > 19)
            .then(1000.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        second = BiasFactor().compute(changed.lazy()).collect()
        self.assertIsNone(first["factor"][18])
        self.assertAlmostEqual(first["factor"][19], 20.0 / 10.5 - 1)
        self.assertEqual(first["factor"][19], second["factor"][19])

    def test_parameters_and_required_columns_fail_clearly(self):
        for invalid in [0, -1, True, 1.5]:
            with self.subTest(window=invalid), self.assertRaises(ValueError):
                BiasFactor(window=invalid)
        with self.assertRaises(ValueError):
            BiasFactor().compute(pl.DataFrame({"date": [1], "symbol": ["A"]}).lazy())


if __name__ == "__main__":
    unittest.main()
