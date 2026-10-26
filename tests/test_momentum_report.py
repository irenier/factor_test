"""Monthly IC report adapts the shared framework statistics."""

import unittest

import numpy as np
import pandas as pd

from example.momentum_report import summarize_ic, summarize_turnover


class MomentumICReportTests(unittest.TestCase):
    def test_turnover_uses_all_sessions_and_sums_monthly_instead_of_averaging_active_dates(
        self,
    ):
        dates = pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-03",
                "2020-01-31",
                "2020-02-03",
                "2020-02-04",
                "2020-02-28",
            ]
        )
        daily = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "portfolio": "A",
                        "turnover": [0.5, 0.0, 0.8, 0.0, 0.2, 0.0],
                        "traded_notional": [100.0, 0.0, 160.0, 0.0, 40.0, 0.0],
                    }
                ),
                pd.DataFrame(
                    {
                        "date": dates,
                        "portfolio": "B",
                        "turnover": 0.0,
                        "traded_notional": 0.0,
                    }
                ),
            ],
            ignore_index=True,
        )
        summary, monthly, annual = summarize_turnover(daily)
        row = summary.set_index("portfolio").loc["A"]
        self.assertEqual(row["trading_days"], 6)
        self.assertEqual(row["trade_days"], 3)
        self.assertAlmostEqual(row["mean_daily_turnover"], 0.25)
        self.assertAlmostEqual(row["mean_trade_day_turnover"], 0.5)
        self.assertAlmostEqual(row["mean_monthly_turnover"], 0.75)
        self.assertAlmostEqual(row["annualized_turnover"], 63.0)
        self.assertAlmostEqual(row["total_turnover"], 1.5)
        np.testing.assert_allclose(
            monthly.loc[monthly["portfolio"] == "A", "turnover"], [1.3, 0.2]
        )
        self.assertAlmostEqual(annual.set_index("portfolio").loc["A", "turnover"], 1.5)
        self.assertEqual(
            summary.set_index("portfolio").loc["B", "mean_trade_day_turnover"], 0.0
        )

    def test_reports_both_directions_with_monthly_annualization_and_valid_counts(self):
        data = pd.DataFrame(
            {
                "signal_date": pd.date_range("2020-01-31", periods=5, freq="ME"),
                "ic": [-0.2, 0.1, 0.3, np.nan, -0.1],
                "rank_ic": [-0.4, 0.0, 0.2, 0.1, np.nan],
            }
        )
        result = summarize_ic(data).set_index(["direction", "metric"])
        self.assertEqual(len(result), 4)
        raw = result.loc[("raw", "ic")]
        negative = result.loc[("negative", "ic")]
        self.assertEqual(raw["n"], 4)
        self.assertEqual(raw["n_dates"], 5)
        self.assertAlmostEqual(raw["coverage_dates"], 0.8)
        self.assertAlmostEqual(raw["ic_mean"], 0.025)
        self.assertAlmostEqual(negative["ic_mean"], -0.025)
        self.assertAlmostEqual(negative["ic_std"], raw["ic_std"])
        self.assertAlmostEqual(negative["t_stat"], -raw["t_stat"])
        self.assertAlmostEqual(negative["p_value"], raw["p_value"])
        self.assertAlmostEqual(raw["ann_ir"], raw["icir"] * np.sqrt(12))
        self.assertEqual(raw["periods_per_year"], 12)
        self.assertAlmostEqual(result.loc[("raw", "rank_ic"), "positive_ratio"], 0.5)
        self.assertAlmostEqual(
            result.loc[("negative", "rank_ic"), "positive_ratio"], 0.25
        )

    def test_nonfinite_values_are_missing_and_constant_series_have_no_ir_or_t_test(
        self,
    ):
        data = pd.DataFrame(
            {
                "signal_date": pd.date_range("2020-01-31", periods=4, freq="ME"),
                "ic": [0.2, np.nan, np.inf, -np.inf],
                "rank_ic": [0.1] * 4,
            }
        )
        result = summarize_ic(data).set_index(["direction", "metric"])
        self.assertEqual(result.loc[("raw", "ic"), "n"], 1)
        self.assertAlmostEqual(result.loc[("raw", "ic"), "coverage_dates"], 0.25)
        self.assertAlmostEqual(result.loc[("raw", "ic"), "positive_ratio"], 1.0)
        self.assertEqual(result.loc[("raw", "rank_ic"), "ic_std"], 0.0)
        for key in ("icir", "ann_ir", "t_stat", "p_value"):
            self.assertTrue(result[key].isna().all())


if __name__ == "__main__":
    unittest.main()
