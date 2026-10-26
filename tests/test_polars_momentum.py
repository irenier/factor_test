"""Common-calendar, listing and IC invariants for the Polars pipeline."""

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from factor_test.diagnostics import quantize_factor
from example.momentum_inputs import build_signals, load_panel, monthly_ic
from example.framework_account import run_accounts


class PolarsMomentumTests(unittest.TestCase):
    def test_quantile_boundary_labels_match_the_original_per_month_source_calls(self):
        dates = [
            date(2020, 1, 30),
            date(2020, 1, 31),
            date(2020, 2, 27),
            date(2020, 2, 28),
        ]
        quotes = pl.DataFrame(
            [
                dict(
                    date=day,
                    symbol=f"S{i}",
                    open=10.0,
                    close=10.0 if t % 2 == 0 else 10.0 + i / 100.0,
                    volume=1.0,
                )
                for t, day in enumerate(dates)
                for i in range(3165 if t < 2 else 3172)
            ]
        ).lazy()
        stocks = pl.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(3172)],
                "listing_date": [date(2010, 1, 1)] * 3172,
                "delisting_date": [None] * 3172,
            },
            schema_overrides={"delisting_date": pl.Date},
        )
        signals, _ = build_signals(
            quotes, stocks, pl.DataFrame({"date": dates}), lookback=2, min_age=1
        )
        expected = pl.concat(
            [
                quantize_factor(
                    part.select("date", "symbol", "factor").lazy()
                ).collect()
                for part in signals.partition_by("date")
            ]
        ).sort(["date", "symbol"])
        assert_frame_equal(
            signals.select(
                "date", "symbol", pl.col("quantile").alias("factor_quantile")
            ),
            expected.select("date", "symbol", "factor_quantile"),
        )

    def test_account_accepts_and_returns_polars_without_changing_integer_shares_or_fees(
        self,
    ):
        days = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 4)]
        quotes = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, open=10.0, suspended=False)
                for day in days
                for symbol in "ABCDE"
            ]
        ).lazy()
        signals = pl.DataFrame(
            {
                "date": [days[0]] * 5,
                "symbol": list("ABCDE"),
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        benchmark = pl.DataFrame({"date": days[1:], "benchmark_nav": [1.0, 1.0]})
        result = run_accounts(
            signals, quotes, benchmark, start=days[1], initial_value=1000.0
        )
        self.assertTrue(all(isinstance(frame, pl.DataFrame) for frame in result))
        daily, metrics, trades, _, _ = result
        long_only = trades.filter(pl.col("portfolio") == "Momentum_LongOnly")
        self.assertEqual(long_only.select("symbol", "quantity").rows(), [("A", 99)])
        self.assertAlmostEqual(long_only["commission"].sum(), 0.297)
        self.assertEqual(long_only["sell_tax"].sum(), 0.0)
        self.assertAlmostEqual(
            daily.filter(pl.col("portfolio") == "Momentum_LS")["nav"][-1], 0.999216
        )
        self.assertEqual(metrics.height, 3)

    def test_missing_quote_keeps_the_exchange_day_and_carries_only_past_prices(self):
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(12)]
        quotes = pl.DataFrame(
            [
                dict(
                    date=day,
                    symbol="A",
                    open=float(i + 10),
                    close=float(i + 10),
                    volume=1.0,
                )
                for i, day in enumerate(days)
                if i != 6
            ]
        ).lazy()
        stocks = pl.DataFrame(
            {"symbol": ["A"], "listing_date": [days[0]], "delisting_date": [None]},
            schema_overrides={"delisting_date": pl.Date},
        )
        signals, _ = build_signals(
            quotes,
            stocks,
            pl.DataFrame({"date": days}),
            lookback=3,
            min_age=1,
            signal_dates=[days[8]],
        )
        self.assertAlmostEqual(
            signals["factor"].item(), 18.0 / ((15.0 + 17.0 + 18.0) / 3) - 1
        )

    def test_monthly_ic_uses_next_opens_and_excludes_missing_endpoints(self):
        days = [
            date(2020, 1, 31),
            date(2020, 2, 3),
            date(2020, 2, 28),
            date(2020, 3, 2),
            date(2020, 3, 31),
            date(2020, 4, 1),
        ]
        quotes = pl.DataFrame(
            [
                dict(
                    date=day,
                    symbol=symbol,
                    open=10.0 + (i if t >= 3 else 0),
                    close=10.0 + i,
                    volume=1.0,
                )
                for t, day in enumerate(days)
                for i, symbol in enumerate("ABCDE")
                if not (t == 3 and symbol == "B")
            ]
        ).lazy()
        signals = pl.DataFrame(
            [
                dict(date=day, symbol=symbol, factor=float(i))
                for day in days[::2]
                for i, symbol in enumerate("ABCDE")
            ]
        )
        result = monthly_ic(
            quotes, signals, pl.DataFrame({"date": days}), start=days[1], end=days[4]
        )
        row = result.row(0, named=True)
        self.assertEqual(result.height, 1)
        self.assertEqual((row["entry_date"], row["exit_date"]), (days[1], days[3]))
        self.assertEqual((row["n_total"], row["n_assets"]), (5, 4))
        self.assertAlmostEqual(row["coverage"], 0.8)
        self.assertAlmostEqual(row["ic"], 1.0)
        self.assertAlmostEqual(row["rank_ic"], 1.0)

    def test_cache_loader_is_lazy_and_records_missing_securities(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            stocks = pl.DataFrame(
                {
                    "symbol": ["A", "B"],
                    "name": ["Alpha", "Beta"],
                    "exchange": ["SH", "SH"],
                    "status": ["listed", "listed"],
                    "listing_date": [date(2010, 1, 1)] * 2,
                    "delisting_date": [None] * 2,
                },
                schema_overrides={"delisting_date": pl.Date},
            )
            stocks.write_parquet(cache / "universe.parquet")
            market = pl.DataFrame(
                {"date": [date(2020, 1, 2)], "open": [100.0], "close": [100.0]}
            )
            market.write_parquet(cache / "csi500_20140101_20251231.parquet")
            folder = cache / "hfq_20140101_20251231"
            folder.mkdir()
            market.with_columns(
                pl.lit("A").alias("symbol"),
                pl.lit(1.0).alias("volume"),
                pl.lit("test").alias("source"),
            ).write_parquet(folder / "A.parquet")
            quotes, result_stocks, calendar, quality = load_panel(
                cache, "20140101", "20251231"
            )
            self.assertIsInstance(quotes, pl.LazyFrame)
            self.assertEqual((result_stocks.height, calendar.height), (2, 1))
            self.assertEqual(
                quality.filter(pl.col("symbol") == "B")["downloaded"].item(), False
            )
            self.assertEqual(quality.filter(pl.col("symbol") == "A")["rows"].item(), 1)


if __name__ == "__main__":
    unittest.main()
