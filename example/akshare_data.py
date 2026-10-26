"""Download/cache the AKShare inputs used by momentum_backtest.py."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import argparse
import json
import os
from pathlib import Path
import time

import polars as pl


def configure_requests(timeout=25, direct=False):
    """AKShare exchange-list endpoints do not expose a timeout argument."""
    import requests

    os.environ.setdefault("TQDM_DISABLE", "1")

    original = requests.sessions.Session.request

    def bounded_request(self, *args, **kwargs):
        if direct:
            self.trust_env = False
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return original(self, *args, **kwargs)

    requests.sessions.Session.request = bounded_request


def universe(cache: Path):
    import akshare as ak
    import pandas as pd

    path = cache / "universe.parquet"
    if path.exists():
        return pl.read_parquet(path).filter(~pl.col("symbol").str.starts_with("689"))
    specifications = [
        (
            "SH",
            "listed",
            lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
            ["证券代码", "证券简称", "上市日期"],
        ),
        (
            "SH",
            "listed",
            lambda: ak.stock_info_sh_name_code(symbol="科创板"),
            ["证券代码", "证券简称", "上市日期"],
        ),
        (
            "SZ",
            "listed",
            lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
            ["A股代码", "A股简称", "A股上市日期"],
        ),
        (
            "BJ",
            "listed",
            ak.stock_info_bj_name_code,
            ["证券代码", "证券简称", "上市日期"],
        ),
        (
            "SH",
            "delisted",
            ak.stock_info_sh_delist,
            ["公司代码", "公司简称", "上市日期", "暂停上市日期"],
        ),
        (
            "SZ",
            "delisted",
            ak.stock_info_sz_delist,
            ["证券代码", "证券简称", "上市日期", "终止上市日期"],
        ),
    ]
    records, status = [], []
    for source_index, (exchange, state, fetch, columns) in enumerate(specifications):
        # Cache each source so an interrupted download does not repeat BSE paging.
        source = cache / f"list_{source_index}_{exchange}_{state}.json"
        try:
            if source.exists():
                data = pd.read_json(source, dtype=False)
            else:
                data = fetch()
                data.to_json(
                    source, orient="records", date_format="iso", force_ascii=False
                )
            for row in data[columns].itertuples(index=False, name=None):
                code, name, listing = row[:3]
                code = str(code).zfill(6)
                if exchange == "SH" and (
                    not code.startswith("6") or code.startswith("689")
                ):
                    continue
                if exchange == "SZ" and not code.startswith(("0", "3")):
                    continue
                listing = pd.to_datetime(listing, errors="coerce")
                if exchange == "BJ" and pd.notna(listing):
                    listing = max(listing, pd.Timestamp("2021-11-15"))
                ending = (
                    pd.to_datetime(row[3], errors="coerce") if len(row) > 3 else pd.NaT
                )
                records.append(
                    dict(
                        symbol=code,
                        name=str(name),
                        exchange=exchange,
                        listing_date=None if pd.isna(listing) else listing.date(),
                        delisting_date=None if pd.isna(ending) else ending.date(),
                        status=state,
                    )
                )
            status.append(
                dict(exchange=exchange, status=state, ok=True, rows=len(data))
            )
            print(f"Universe {exchange} {state}: {len(data)}", flush=True)
        except Exception as exc:
            status.append(
                dict(exchange=exchange, status=state, ok=False, error=str(exc))
            )
            print(f"Universe source failed: {exchange} {state}: {exc}", flush=True)
    (cache / "universe_sources.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not records or not all(
        any(
            s["ok"] and s["exchange"] == market and s["status"] == "listed"
            for s in status
        )
        for market in ("SH", "SZ", "BJ")
    ):
        raise RuntimeError(
            "沪深京上市股票列表未全部获取成功，修复后重新运行；已获取数据已缓存"
        )
    result = (
        pl.DataFrame(records, infer_schema_length=None)
        .with_columns(pl.col("listing_date", "delisting_date").cast(pl.Date))
        .unique("symbol", keep="first", maintain_order=True)
        .sort("symbol")
    )
    result.write_parquet(path)
    return result


def benchmark(cache: Path, start: str, end: str):
    import akshare as ak

    path = cache / f"csi500_{start}_{end}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    errors = []
    for fetch in (
        lambda: ak.stock_zh_index_daily_em(
            symbol="csi000905", start_date=start, end_date=end
        ),
        lambda: ak.stock_zh_index_daily(symbol="sh000905"),
    ):
        try:
            frame = pl.DataFrame(fetch().to_dict(orient="list")).with_columns(
                pl.col("date").cast(pl.Date)
            )
            frame = frame.filter(
                pl.col("date").is_between(
                    date.fromisoformat(f"{start[:4]}-{start[4:6]}-{start[6:]}"),
                    date.fromisoformat(f"{end[:4]}-{end[4:6]}-{end[6:]}"),
                )
            )
            frame = frame.select("date", "open", "close").sort("date")
            if frame.height < 200:
                raise ValueError("中证500行情过短")
            frame.write_parquet(path)
            return frame
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"中证500获取失败: {errors}")


def download_prices(
    stocks, cache: Path, start: str, end: str, workers=6, provider="sina"
):
    import akshare as ak

    folder = cache / f"hfq_{start}_{end}"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        symbol = row["symbol"]
        path = folder / f"{symbol}.parquet"
        if path.exists():
            return dict(symbol=symbol, ok=True, cached=True)
        error = ""
        for attempt in range(3):
            try:
                source = [
                    provider,
                    "em" if provider == "sina" else "sina",
                    "tx" if provider != "tx" else "em",
                ][attempt]
                prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}[row["exchange"]]
                if source == "sina":
                    data = ak.stock_zh_a_daily(
                        symbol=prefix + symbol,
                        start_date=start,
                        end_date=end,
                        adjust="hfq",
                    )
                elif source == "tx":
                    data = ak.stock_zh_a_hist_tx(
                        symbol=prefix + symbol,
                        start_date=start,
                        end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}",
                        adjust="hfq",
                        timeout=15,
                    )
                else:
                    data = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust="hfq",
                        timeout=20,
                    )
                if data.empty:
                    raise ValueError("empty history")
                frame = pl.DataFrame(data.to_dict(orient="list"))
                if source == "em":
                    frame = frame.rename(
                        {
                            "日期": "date",
                            "开盘": "open",
                            "收盘": "close",
                            "成交量": "volume",
                        }
                    )
                frame = frame.select(
                    pl.col("date").cast(pl.Date),
                    pl.lit(symbol).alias("symbol"),
                    pl.col("open", "close", "volume").cast(pl.Float64),
                    pl.lit(source).alias("source"),
                )
                if frame["date"].n_unique() != frame.height:
                    raise ValueError("duplicate dates")
                frame.sort("date").write_parquet(path)
                return dict(
                    symbol=symbol,
                    ok=True,
                    cached=False,
                    rows=frame.height,
                    source=source,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5 * (attempt + 1))
        return dict(symbol=symbol, ok=False, cached=False, error=error)

    records = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in stocks.iter_rows(named=True)]
        for future in as_completed(futures):
            records.append(future.result())
            if not records[-1]["ok"] and sum(not r["ok"] for r in records) <= 5:
                print(
                    f"Failed {records[-1]['symbol']}: {records[-1]['error']}",
                    flush=True,
                )
            if len(records) % 100 == 0 or len(records) == stocks.height:
                good = sum(record["ok"] for record in records)
                print(
                    f"Prices {len(records)}/{stocks.height}; ok={good}; "
                    f"failed={len(records)-good}; elapsed={time.monotonic()-started:.0f}s",
                    flush=True,
                )
    pl.DataFrame(records, infer_schema_length=None).sort("symbol").write_csv(
        cache / "download_status.csv"
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20140101", help="Warm-up start, YYYYMMDD")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--cache", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--provider", choices=["sina", "em", "tx"], default="sina")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass environment proxies for this process",
    )
    parser.add_argument(
        "--limit", type=int, help="Smoke test only; report must disclose a subset"
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    configure_requests(direct=args.direct)
    stocks = universe(args.cache)
    start_day = date.fromisoformat(
        f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]}"
    )
    end_day = date.fromisoformat(f"{args.end[:4]}-{args.end[4:6]}-{args.end[6:]}")
    stocks = stocks.filter(
        (pl.col("listing_date") <= end_day)
        & (pl.col("delisting_date").is_null() | (pl.col("delisting_date") >= start_day))
    )
    if args.limit:
        stocks = stocks.head(args.limit)
    market = benchmark(args.cache, args.start, args.end)
    print(
        f"CSI500: {market.height} rows; requested stocks: {stocks.height}", flush=True
    )
    download_prices(
        stocks, args.cache, args.start, args.end, args.workers, args.provider
    )


if __name__ == "__main__":
    main()
