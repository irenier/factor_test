import unittest
from math import prod

import polars as pl

from factor_test.analyzer import (
    compute_returns,
    daily_information_coefficient,
    portfolio_returns,
    information_coefficient,
)


class ReturnsTests(unittest.TestCase):
    def setUp(self):
        self.prices = pl.DataFrame(
            {
                "date": [1, 2, 3, 4],
                "symbol": ["A"] * 4,
                "factor": [1.0] * 4,
                "open": [100.0, 110.0, 121.0, 133.1],
            }
        )

    def test_unsorted_prices_and_entry_lag(self):
        data = self.prices.reverse().lazy()
        rows = compute_returns(data, horizons=(1, 2)).collect().to_dicts()
        self.assertAlmostEqual(rows[0]["ret_1"], 0.1)
        self.assertAlmostEqual(rows[0]["ret_2"], 0.21)
        self.assertEqual((rows[0]["entry_date"], rows[0]["exit_date_2"]), (2, 4))
        self.assertIsNone(rows[2]["ret_1"])
        self.assertIsNone(rows[3]["entry_date"])

    def test_zero_entry_lag(self):
        row = (
            compute_returns(self.prices.lazy(), horizons=(2,), entry_lag=0)
            .collect()
            .row(0, named=True)
        )
        self.assertEqual((row["entry_date"], row["exit_date_2"]), (1, 3))
        self.assertAlmostEqual(row["ret_2"], 0.21)

    def test_sparse_factor_uses_complete_price_calendar(self):
        signals = self.prices.filter(pl.col("date").is_in([1, 3])).select(
            "date", "symbol", "factor"
        )
        result = compute_returns(
            signals.lazy(), horizons=(1,), prices=self.prices.lazy()
        ).collect()
        self.assertEqual(result.height, 2)
        self.assertAlmostEqual(result["ret_1"][0], 0.1)
        self.assertIsNone(result["ret_1"][1])

    def test_missing_stock_quote_does_not_skip_calendar_date(self):
        # B provides the market date omitted from A's quotes.
        prices = pl.DataFrame(
            {
                "date": [1, 3, 4, 1, 2, 3, 4],
                "symbol": ["A", "A", "A", "B", "B", "B", "B"],
                "open": [100.0, 121.0, 133.1, 10.0, 11.0, 12.0, 13.0],
            }
        )
        result = (
            compute_returns(
                self.prices.head(1).lazy(), horizons=(1,), prices=prices.lazy()
            )
            .collect()
            .row(0, named=True)
        )
        self.assertEqual(result["entry_date"], 2)
        self.assertEqual(result["exit_date_1"], 3)
        self.assertIsNone(result["ret_1"])

    def test_dates_absent_from_all_prices_are_not_inserted(self):
        prices = self.prices.filter(pl.col("date") != 2)
        result = compute_returns(
            prices.lazy(),
            horizons=(1,),
        ).collect()
        self.assertAlmostEqual(result["ret_1"][0], 0.1)
        self.assertEqual(result["entry_date"][0], 3)

    def test_invalid_entry_or_exit_quote_has_unknown_return(self):
        for date in (2, 3):
            for price in (None, 0.0, -1.0, float("nan"), float("inf")):
                prices = self.prices.with_columns(
                    pl.when(pl.col("date") == date)
                    .then(pl.lit(price, dtype=pl.Float64))
                    .otherwise(pl.col("open"))
                    .alias("open")
                )
                with self.subTest(date=date, price=price):
                    result = compute_returns(prices.lazy(), horizons=(1,)).collect()
                    self.assertIsNone(result["ret_1"][0])

    def test_duplicate_and_null_keys_are_rejected(self):
        for data in [
            pl.concat([self.prices, self.prices.head(1)]),
            self.prices.with_columns(pl.lit(None).cast(pl.String).alias("symbol")),
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                compute_returns(data.lazy(), horizons=(1,))
        with self.assertRaises(ValueError):
            compute_returns(
                self.prices.lazy(),
                horizons=(1,),
                prices=pl.concat([self.prices, self.prices.head(1)]).lazy(),
            )

    def test_invalid_horizons_and_lags_are_rejected(self):
        for horizons in [(), (0,), (-1,), (1, 1), (1.5,), (True,)]:
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                compute_returns(self.prices.lazy(), horizons=horizons)
        for lag in [-1, 1.5, True]:
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                compute_returns(self.prices.lazy(), entry_lag=lag)


class InformationCoefficientTests(unittest.TestCase):
    def test_long_format_and_summary_connect(self):
        data = pl.DataFrame(
            {
                "date": [1, 1, 1, 2, 2, 2],
                "symbol": ["A", "B", "C"] * 2,
                "factor": [1.0, 2.0, 3.0] * 2,
                "ret_1": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
                "ret_2": [0.3, 0.2, 0.1, 0.1, 0.2, 0.3],
            }
        )
        ic = daily_information_coefficient(data.lazy(), horizons=(1, 2))
        result = ic.collect()
        self.assertEqual(
            result.columns,
            ["date", "horizon", "n_total", "n_assets", "coverage", "ic", "rank_ic"],
        )
        self.assertEqual(result.height, 4)
        self.assertEqual(result["n_assets"].to_list(), [3] * 4)
        self.assertAlmostEqual(result["ic"][0], 1.0)
        self.assertAlmostEqual(result["rank_ic"][1], -1.0)
        summary = information_coefficient(ic).collect()
        self.assertEqual(summary.height, 4)
        self.assertEqual(summary["n"].to_list(), [2] * 4)
        for mean in summary["ic_mean"]:
            self.assertAlmostEqual(mean, 0.0)

    def test_undefined_correlations_preserve_dates_and_counts(self):
        data = pl.DataFrame(
            {
                "date": [1, 1, 2, 2],
                "symbol": ["A", "B", "A", "B"],
                "factor": [1.0, 1.0, None, float("nan")],
                "ret_1": [0.1, 0.2, 0.1, 0.2],
            }
        )
        result = daily_information_coefficient(data.lazy(), horizons=(1,)).collect()
        self.assertEqual(result["n_assets"].to_list(), [2, 0])
        self.assertEqual(result["ic"].to_list(), [None, None])


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame(
            {
                "date": [1, 2, 3, 4],
                "symbol": ["A"] * 4,
                "factor": [1.0] * 4,
                "open": [100.0, 110.0, 121.0, 133.1],
            }
        ).lazy()

    def test_daily_rebalanced_pipeline_compounds_on_realization_dates(self):
        prepared = compute_returns(self.data, horizons=(1, 2))
        account = portfolio_returns(
            prepared,
            weight_method="equal",
            long_short=False,
            rebalance_every=1,
            initial_value=1100,
        )
        self.assertEqual(account.daily["date"].to_list(), [1, 2, 3, 4])
        self.assertEqual(account.trades["date"].to_list(), [2])
        self.assertAlmostEqual(account.daily["nav"][-1], 1.21)
        self.assertAlmostEqual(
            prod(1 + value for value in account.daily["net_return"]), 1.21
        )

    def test_multi_horizon_research_stays_separate(self):
        prepared = compute_returns(self.data, horizons=(1, 2))
        account = portfolio_returns(
            prepared,
            weight_method="equal",
            long_short=False,
            rebalance_every=1,
            initial_value=1100,
        )
        direct = portfolio_returns(
            self.data,
            weight_method="equal",
            long_short=False,
            rebalance_every=1,
            initial_value=1100,
        )
        self.assertEqual(account.daily.to_dicts(), direct.daily.to_dicts())
        with self.assertRaises(ValueError):
            portfolio_returns(prepared, rebalance_every=0)


if __name__ == "__main__":
    unittest.main()
