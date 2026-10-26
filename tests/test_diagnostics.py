import unittest

import polars as pl

from factor_test.analyzer import (
    daily_information_coefficient,
    factor_rank_autocorrelation,
    factor_weights,
    information_coefficient,
    mean_return_by_quantile,
    quantile_monotonicity,
    quantile_return_summary,
    quantile_spread,
    quantile_turnover,
    quantize_factor,
    spread_return_summary,
)


class InformationCoefficientSummaryTests(unittest.TestCase):
    def summary(self, values):
        data = pl.DataFrame(
            {
                "date": range(1, len(values) + 1),
                "horizon": [5] * len(values),
                "ic": values,
                "rank_ic": values,
            },
            schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64},
        )
        return information_coefficient(data.lazy()).collect().row(0, named=True)

    def test_missing_observations_preserve_counts_and_coverage(self):
        result = self.summary([0.1, None, 0.3, 0.4])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_dates"], 4)
        self.assertEqual(result["coverage_dates"], 0.75)
        self.assertAlmostEqual(result["ic_mean"], 0.8 / 3)

    def test_no_valid_dates_still_reports_the_horizon(self):
        result = self.summary([None, None])
        self.assertEqual(result["horizon"], 5)
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["t_stat"])
        self.assertIsNone(result["p_value"])

    def test_minimum_cross_section_and_coverage(self):
        data = pl.DataFrame(
            {
                "date": [1, 1, 1],
                "symbol": ["A", "B", "C"],
                "factor": [1.0, 2.0, 3.0],
                "ret_1": [0.1, 0.2, None],
            }
        )
        row = (
            daily_information_coefficient(data.lazy(), horizons=(1,))
            .collect()
            .row(0, named=True)
        )
        self.assertEqual(row["n_assets"], 2)
        self.assertAlmostEqual(row["coverage"], 2 / 3)
        self.assertIsNone(row["ic"])
        row = (
            daily_information_coefficient(data.lazy(), horizons=(1,), min_assets=2)
            .collect()
            .row(0, named=True)
        )
        self.assertAlmostEqual(row["ic"], 1.0)


class QuantileTests(unittest.TestCase):
    def panel(self):
        return pl.DataFrame(
            {
                "date": [1] * 10 + [2] * 10,
                "symbol": list("ABCDEFGHIJ") * 2,
                "factor": [float(i) for i in range(1, 11)] * 2,
                "ret_1": [i / 100 for i in range(1, 11)] * 2,
            }
        )

    def test_ties_never_split_by_stock_code(self):
        data = self.panel().with_columns(
            pl.when(pl.col("symbol").is_in(["A", "B", "C", "D"]))
            .then(1.0)
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        result = quantize_factor(data.lazy()).collect()
        labels = result.filter(pl.col("factor") == 1.0)["factor_quantile"].unique()
        self.assertEqual(len(labels), 1)
        self.assertEqual(
            result.to_dicts(),
            quantize_factor(data.reverse().lazy()).collect().to_dicts(),
        )

    def test_constant_factors_have_no_artificial_quantile_position(self):
        data = self.panel().with_columns(pl.lit(1.0).alias("factor"))
        result = quantize_factor(data.lazy()).collect()
        self.assertTrue(result["factor_quantile"].is_null().all())
        for long_short in [True, False]:
            weights = factor_weights(data.lazy(), "quantile", long_short).collect()
            self.assertEqual(weights["weight"].abs().sum(), 0)

    def test_group_means_spread_and_monotonicity(self):
        groups = mean_return_by_quantile(self.panel().lazy(), horizons=(1,))
        result = groups.collect()
        self.assertEqual(result.height, 10)
        self.assertEqual(result["n_assets"].to_list(), [2] * 10)
        self.assertAlmostEqual(result["mean_return"][0], 0.015)
        self.assertAlmostEqual(result["mean_return"][4], 0.095)
        self.assertAlmostEqual(result["std_error"][0], 0.005)
        self.assertAlmostEqual(quantile_spread(groups).collect()["spread"][0], 0.04)
        for value in quantile_monotonicity(groups).collect()["monotonicity"]:
            self.assertAlmostEqual(value, 1.0)
        summary = quantile_return_summary(groups).collect()
        self.assertEqual(summary.height, 5)
        self.assertEqual(summary["n_dates"].to_list(), [2] * 5)
        spread_stats = spread_return_summary(quantile_spread(groups)).collect()
        self.assertAlmostEqual(spread_stats["mean_spread"][0], 0.04)
        self.assertEqual(spread_stats["n_dates"][0], 2)

    def test_missing_group_returns_and_empty_groups_are_not_zero(self):
        data = (
            self.panel()
            .filter(pl.col("symbol").is_in(["A", "J"]))
            .with_columns(
                pl.when(pl.col("symbol") == "J")
                .then(None)
                .otherwise(pl.col("ret_1"))
                .alias("ret_1")
            )
        )
        result = mean_return_by_quantile(
            data.lazy(), quantiles=5, horizons=(1,)
        ).collect()
        self.assertTrue(
            result.filter(pl.col("n_assets") == 0)["mean_return"].is_null().all()
        )
        self.assertEqual(result.filter(pl.col("n_total") == 0).height, 6)

    def test_turnover_and_rank_persistence(self):
        data = self.panel()
        turns = quantile_turnover(data.lazy()).collect()
        self.assertTrue(turns.filter(pl.col("date") == 1)["turnover"].is_null().all())
        self.assertEqual(
            turns.filter(pl.col("date") == 2)["turnover"].to_list(), [0.0] * 5
        )
        reversed_day = data.with_columns(
            pl.when(pl.col("date") == 2)
            .then(-pl.col("factor"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        auto = factor_rank_autocorrelation(reversed_day.lazy()).collect()
        self.assertIsNone(auto["rank_autocorrelation"][0])
        self.assertAlmostEqual(auto["rank_autocorrelation"][1], -1.0)
        flips = quantile_turnover(reversed_day.lazy()).collect()
        self.assertEqual(
            flips.filter((pl.col("date") == 2) & (pl.col("factor_quantile") == 5))[
                "turnover"
            ][0],
            1.0,
        )

    def test_persistence_preserves_observed_dates_with_missing_signals(self):
        data = self.panel().with_columns(
            pl.when(pl.col("date") == 2).then(3).otherwise(pl.col("date")).alias("date")
        )
        missing = (
            self.panel()
            .filter(pl.col("date") == 2)
            .with_columns(pl.lit(None, dtype=pl.Float64).alias("factor"))
        )
        data = pl.concat([data, missing])
        result = factor_rank_autocorrelation(data.lazy()).collect()
        self.assertEqual(result["date"].to_list(), [1, 2, 3])
        self.assertTrue(result["rank_autocorrelation"].is_null().all())
        turns = quantile_turnover(data.lazy()).collect()
        self.assertTrue(turns["turnover"].is_null().all())

    def test_persistence_uses_only_observed_dates(self):
        data = self.panel().with_columns(
            pl.when(pl.col("date") == 2).then(3).otherwise(pl.col("date")).alias("date")
        )
        result = factor_rank_autocorrelation(data.lazy()).collect()
        self.assertEqual(result["date"].to_list(), [1, 3])
        self.assertAlmostEqual(result["rank_autocorrelation"][1], 1.0)
        turns = quantile_turnover(data.lazy()).collect()
        self.assertEqual(
            turns.filter(pl.col("date") == 3)["turnover"].to_list(), [0.0] * 5
        )


if __name__ == "__main__":
    unittest.main()
