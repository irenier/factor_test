"""Shared input validation and observed-date ordering."""

import polars as pl


def validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("horizons 必须是非空的正整数序列")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons 不能重复")


def validate_keys(data: pl.LazyFrame, keys: list[str]) -> None:
    invalid = (
        data.select(
            pl.any_horizontal(pl.col(key).is_null() for key in keys)
            .any()
            .alias("null"),
            pl.struct(keys).is_duplicated().any().alias("duplicate"),
        )
        .collect()
        .row(0, named=True)
    )
    if invalid["null"] or invalid["duplicate"]:
        raise ValueError(f"{keys} 必须非空且唯一")


def calendar_for(data: pl.LazyFrame) -> pl.LazyFrame:
    """返回输入中去重排序的已有日期，不补充缺失日期。"""
    observed = data.select("date").unique()
    validate_keys(observed, ["date"])
    return observed.sort("date")
