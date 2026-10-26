import unittest

import polars as pl

from factor_test._utils import calendar_for, validate_keys


class CalendarTests(unittest.TestCase):
    def test_calendar_sorts_unique_observed_dates(self):
        data = pl.DataFrame({"date": [3, 1, 3]}).lazy()
        self.assertEqual(calendar_for(data).collect()["date"].to_list(), [1, 3])

    def test_null_observed_date_is_rejected(self):
        with self.assertRaises(ValueError):
            calendar_for(pl.DataFrame({"date": [1, None, 3]}).lazy())


class KeyValidationTests(unittest.TestCase):
    def test_null_or_duplicate_key_is_rejected(self):
        for dates in ([1, None], [1, 1]):
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                validate_keys(pl.DataFrame({"date": dates}).lazy(), ["date"])


if __name__ == "__main__":
    unittest.main()
