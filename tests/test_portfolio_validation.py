"""Public input errors and normalization at the portfolio boundary."""

import unittest

import polars as pl

from factor_test.portfolio import backtest_portfolio


def sample_weights():
    return pl.DataFrame({"date": [1], "symbol": ["A"], "weight": [1.0]}).lazy()


def sample_prices():
    return pl.DataFrame(
        {"date": [1, 2, 3], "symbol": ["A"] * 3, "open": [100.0] * 3}
    ).lazy()


class PortfolioValidationTests(unittest.TestCase):
    def test_only_suspension_controls_trading_and_no_legacy_sleeve_fields_remain(self):
        for name in ("tradable_col", "buyable_col", "sellable_col"):
            with self.subTest(parameter=name), self.assertRaises(TypeError):
                backtest_portfolio(sample_weights(), sample_prices(), **{name: "flag"})
        result = backtest_portfolio(
            sample_weights(), sample_prices(), initial_value=1000
        )
        self.assertNotIn("active_sleeves", result.daily.columns)
        for frame in (result.positions, result.trades, result.unfilled_orders):
            self.assertNotIn("sleeve", frame.columns)

    def test_missing_required_columns_identify_the_source_and_column(self):
        for source, column in (
            ("weights", "date"),
            ("weights", "symbol"),
            ("weights", "weight"),
            ("prices", "date"),
            ("prices", "symbol"),
            ("prices", "open"),
        ):
            weights, prices = sample_weights(), sample_prices()
            if source == "weights":
                weights = weights.drop(column)
            else:
                prices = prices.drop(column)
            with self.subTest(source=source, column=column):
                with self.assertRaisesRegex(ValueError, rf"{source}.*{column}"):
                    backtest_portfolio(weights, prices)
        with self.assertRaisesRegex(ValueError, "halted"):
            backtest_portfolio(
                sample_weights(), sample_prices(), suspended_col="halted"
            )

    def test_non_numeric_weights_and_prices_are_rejected_before_execution(self):
        for source, column in (("weights", "weight"), ("prices", "open")):
            weights, prices = sample_weights(), sample_prices()
            if source == "weights":
                weights = weights.with_columns(pl.lit("1.0").alias(column))
            else:
                prices = prices.with_columns(pl.lit("100.0").alias(column))
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, column):
                    backtest_portfolio(weights, prices)

    def test_suspension_requires_boolean_values(self):
        for flag in (1, "False"):
            with self.subTest(flag=flag):
                prices = sample_prices().with_columns(pl.lit(flag).alias("halted"))
                with self.assertRaisesRegex(ValueError, "halted"):
                    backtest_portfolio(sample_weights(), prices, suspended_col="halted")

    def test_null_and_duplicate_keys_are_rejected_in_both_inputs(self):
        for source in ("weights", "prices"):
            for invalid_key in ("date", "symbol", "duplicate"):
                weights, prices = sample_weights(), sample_prices()
                frame = weights if source == "weights" else prices
                if invalid_key == "duplicate":
                    frame = pl.concat([frame, frame.head(1)])
                else:
                    frame = frame.with_columns(pl.lit(None).alias(invalid_key))
                if source == "weights":
                    weights = frame
                else:
                    prices = frame
                with self.subTest(source=source, invalid_key=invalid_key):
                    with self.assertRaisesRegex(ValueError, source):
                        backtest_portfolio(weights, prices)

    def test_missing_and_nonfinite_weights_are_rejected(self):
        for weight in (None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(weight=weight):
                weights = sample_weights().with_columns(
                    pl.lit(weight, dtype=pl.Float64).alias("weight")
                )
                with self.assertRaisesRegex(ValueError, "weight"):
                    backtest_portfolio(weights, sample_prices())

    def test_invalid_numeric_options_raise_value_error_with_parameter_name(self):
        options = {
            "initial_value": (True, "1000", float("nan"), float("inf"), 0, -1),
            "rebalance_every": (True, "5", float("nan"), 1.5, 0, -1),
            "entry_lag": (False, "1", float("nan"), 1.5, -1),
            "periods_per_year": (True, "252", float("nan"), 252.5, 0, -1),
            "commission_rate": (True, "0.01", float("nan"), float("inf"), -0.01, 1.0),
            "slippage_rate": (False, "0.01", float("nan"), float("inf"), -0.01, 1.0),
            "sell_tax_rate": (True, "0.01", float("nan"), float("inf"), -0.01, 1.0),
            "short_borrow_rate": (True, "0.05", float("nan"), float("inf"), -0.01),
        }
        for name, values in options.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        backtest_portfolio(
                            sample_weights(), sample_prices(), **{name: value}
                        )

    def test_invalid_flags_column_names_and_price_policy_identify_parameter(self):
        for name, value in (
            ("liquidate_at_end", 1),
            ("record_positions", "yes"),
            ("price_col", ""),
            ("price_col", True),
            ("suspended_col", ""),
            ("suspended_col", 1),
            ("missing_price", "fill"),
        ):
            with self.subTest(parameter=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    backtest_portfolio(
                        sample_weights(), sample_prices(), **{name: value}
                    )

    def test_floating_point_sum_noise_is_normalized_without_extra_exposure(self):
        cases = (
            pl.DataFrame({"date": [1], "symbol": ["A"], "weight": [1.0 + 5e-13]}),
            pl.DataFrame(
                {"date": [1] * 10, "symbol": list("ABCDEFGHIJ"), "weight": [0.1] * 10},
                schema_overrides={"weight": pl.Float32},
            ),
        )
        for weights in cases:
            prices = pl.DataFrame(
                [
                    dict(date=day, symbol=symbol, open=10.0)
                    for day in (1, 2)
                    for symbol in weights["symbol"]
                ]
            )
            with self.subTest(weight_dtype=weights.schema["weight"]):
                result = backtest_portfolio(
                    weights.lazy(), prices.lazy(), initial_value=1000
                )
                self.assertFalse(result.trades.is_empty())
                self.assertGreaterEqual(result.daily["cash"][-1], 0)
                self.assertLessEqual(result.daily["gross_exposure"][-1], 1)
                self.assertEqual(result.daily["equity"][-1], 1000)

    def test_true_excess_gross_exposure_is_rejected_even_when_net_weight_is_zero(self):
        cases = (
            ([1.0 + 1e-8], pl.Float64),
            ([0.51, 0.51], pl.Float32),
            ([0.8, -0.8], pl.Float64),
        )
        for values, dtype in cases:
            weights = pl.DataFrame(
                {
                    "date": [1] * len(values),
                    "symbol": list("AB")[: len(values)],
                    "weight": values,
                },
                schema_overrides={"weight": dtype},
            ).lazy()
            with self.subTest(values=values, dtype=dtype):
                with self.assertRaisesRegex(ValueError, "总绝对权重"):
                    backtest_portfolio(weights, sample_prices())

    def test_integer_key_widths_are_compatible(self):
        for weight_type, price_type in ((pl.Int32, pl.Int64), (pl.Int64, pl.Int32)):
            weights = pl.DataFrame(
                {"date": [1], "symbol": [7], "weight": [1.0]},
                schema_overrides={"date": weight_type, "symbol": weight_type},
            )
            prices = pl.DataFrame(
                {"date": [1, 2], "symbol": [7, 7], "open": [100.0, 100]},
                schema_overrides={"date": price_type, "symbol": price_type},
            )
            with self.subTest(weight_type=weight_type, price_type=price_type):
                result = backtest_portfolio(
                    weights.lazy(), prices.lazy(), initial_value=1000
                )
                self.assertEqual(
                    result.trades.select("date", "symbol", "quantity").rows(),
                    [(2, 7, 10)],
                )

    def test_incompatible_key_types_raise_clear_errors(self):
        for key in ("date", "symbol"):
            weights = sample_weights().with_columns(
                pl.lit("1" if key == "date" else 7).alias(key)
            )
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    backtest_portfolio(weights, sample_prices())

    def test_empty_or_zero_weight_portfolio_holds_cash(self):
        empty = pl.DataFrame({"date": [], "symbol": [], "weight": []}).lazy()
        zero = sample_weights().with_columns(pl.lit(0.0).alias("weight"))
        for label, weights in (("empty", empty), ("zero", zero)):
            with self.subTest(weights=label):
                result = backtest_portfolio(
                    weights, sample_prices(), initial_value=1000
                )
                self.assertEqual(result.daily["cash"].to_list(), [1000] * 3)
                self.assertEqual(result.daily["nav"].to_list(), [1.0] * 3)
                self.assertTrue(result.trades.is_empty())
                self.assertTrue(result.positions.is_empty())


if __name__ == "__main__":
    unittest.main()
