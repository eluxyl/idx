"""Reusable MLS data cleaning pipeline.

This module standardizes raw MLS exports for analytics by:
1) parsing date columns,
2) removing redundant columns,
3) handling missing values,
4) enforcing numeric types,
5) flagging/removing invalid numeric values,
6) adding date consistency flags,
7) adding geographic quality flags,
and saving cleaned datasets with a data-quality report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import pandas as pd


DEFAULT_DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

DEFAULT_NUMERIC_COLUMNS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "Latitude",
    "Longitude",
]

DEFAULT_MISSING_FILL_MAP = {
    "PropertyType": "Unknown",
    "PropertySubType": "Unknown",
    "City": "Unknown",
    "CountyOrParish": "Unknown",
    "StateOrProvince": "Unknown",
}


@dataclass
class CleaningConfig:
    date_columns: List[str] = None
    numeric_columns: List[str] = None
    missing_fill_map: Dict[str, Any] = None
    drop_missing_column_threshold: Optional[float] = 0.98
    invalid_numeric_strategy: str = "remove"  # "remove" or "flag"
    extra_drop_columns: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.date_columns is None:
            self.date_columns = list(DEFAULT_DATE_COLUMNS)
        if self.numeric_columns is None:
            self.numeric_columns = list(DEFAULT_NUMERIC_COLUMNS)
        if self.missing_fill_map is None:
            self.missing_fill_map = dict(DEFAULT_MISSING_FILL_MAP)
        if self.extra_drop_columns is None:
            self.extra_drop_columns = []


def _existing_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def parse_date_columns(df: pd.DataFrame, date_columns: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    parse_failures: Dict[str, int] = {}

    for col in _existing_columns(out, date_columns):
        source_not_null = out[col].notna().sum()
        out[col] = pd.to_datetime(out[col], errors="coerce")
        parsed_not_null = out[col].notna().sum()
        parse_failures[col] = int(source_not_null - parsed_not_null)

    return out, parse_failures


def drop_redundant_columns(
    df: pd.DataFrame,
    drop_missing_column_threshold: Optional[float] = None,
    extra_drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = df.copy()

    dropped_duplicate_suffix: List[str] = []
    for col in list(out.columns):
        if not col.endswith(".1"):
            continue
        base = col[:-2]
        if base in out.columns:
            dropped_duplicate_suffix.append(col)

    if dropped_duplicate_suffix:
        out = out.drop(columns=dropped_duplicate_suffix)

    dropped_high_missing: List[str] = []
    if drop_missing_column_threshold is not None:
        missing_rates = out.isna().mean()
        dropped_high_missing = (
            missing_rates[missing_rates >= drop_missing_column_threshold]
            .index.tolist()
        )
        if dropped_high_missing:
            out = out.drop(columns=dropped_high_missing)

    dropped_extra: List[str] = []
    if extra_drop_columns:
        dropped_extra = _existing_columns(out, list(extra_drop_columns))
        if dropped_extra:
            out = out.drop(columns=dropped_extra)

    info = {
        "dropped_duplicate_suffix": dropped_duplicate_suffix,
        "dropped_high_missing": dropped_high_missing,
        "dropped_extra": dropped_extra,
    }
    return out, info


def fill_missing_values(df: pd.DataFrame, fill_map: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    filled_counts: Dict[str, int] = {}

    for col, fill_value in fill_map.items():
        if col not in out.columns:
            continue
        before = int(out[col].isna().sum())
        out[col] = out[col].fillna(fill_value)
        after = int(out[col].isna().sum())
        filled_counts[col] = before - after

    return out, filled_counts


def coerce_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    coercion_new_nulls: Dict[str, int] = {}

    for col in _existing_columns(out, numeric_columns):
        before_nulls = int(out[col].isna().sum())
        out[col] = pd.to_numeric(out[col], errors="coerce")
        after_nulls = int(out[col].isna().sum())
        coercion_new_nulls[col] = after_nulls - before_nulls

    return out, coercion_new_nulls


def add_invalid_numeric_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()

    def invalid_nonpositive(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return out[col].notna() & (out[col] <= 0)

    def invalid_negative(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return out[col].notna() & (out[col] < 0)

    out["closeprice_nonpositive_flag"] = invalid_nonpositive("ClosePrice")
    out["livingarea_nonpositive_flag"] = invalid_nonpositive("LivingArea")
    out["days_on_market_negative_flag"] = invalid_negative("DaysOnMarket")
    out["bedrooms_negative_flag"] = invalid_negative("BedroomsTotal")
    out["bathrooms_negative_flag"] = invalid_negative("BathroomsTotalInteger")

    out["invalid_numeric_flag"] = (
        out["closeprice_nonpositive_flag"]
        | out["livingarea_nonpositive_flag"]
        | out["days_on_market_negative_flag"]
        | out["bedrooms_negative_flag"]
        | out["bathrooms_negative_flag"]
    )

    counts = {
        "closeprice_nonpositive_flag": int(out["closeprice_nonpositive_flag"].sum()),
        "livingarea_nonpositive_flag": int(out["livingarea_nonpositive_flag"].sum()),
        "days_on_market_negative_flag": int(out["days_on_market_negative_flag"].sum()),
        "bedrooms_negative_flag": int(out["bedrooms_negative_flag"].sum()),
        "bathrooms_negative_flag": int(out["bathrooms_negative_flag"].sum()),
        "invalid_numeric_flag": int(out["invalid_numeric_flag"].sum()),
    }
    return out, counts


def apply_invalid_numeric_strategy(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, int]:
    if strategy not in {"remove", "flag"}:
        raise ValueError("invalid_numeric_strategy must be one of: 'remove', 'flag'")

    if strategy == "flag":
        return df, 0

    before = len(df)
    out = df.loc[~df["invalid_numeric_flag"]].copy()
    removed = before - len(out)
    return out, removed


def add_date_consistency_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()

    close = out["CloseDate"] if "CloseDate" in out.columns else pd.Series(pd.NaT, index=out.index)
    purchase = (
        out["PurchaseContractDate"]
        if "PurchaseContractDate" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    listing = (
        out["ListingContractDate"]
        if "ListingContractDate" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )

    out["listing_after_close_flag"] = listing.notna() & close.notna() & (listing > close)
    out["purchase_after_close_flag"] = purchase.notna() & close.notna() & (purchase > close)
    out["negative_timeline_flag"] = listing.notna() & purchase.notna() & (purchase < listing)

    counts = {
        "listing_after_close_flag": int(out["listing_after_close_flag"].sum()),
        "purchase_after_close_flag": int(out["purchase_after_close_flag"].sum()),
        "negative_timeline_flag": int(out["negative_timeline_flag"].sum()),
    }
    return out, counts


def add_geographic_quality_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()

    lat = out["Latitude"] if "Latitude" in out.columns else pd.Series(pd.NA, index=out.index)
    lon = out["Longitude"] if "Longitude" in out.columns else pd.Series(pd.NA, index=out.index)

    out["missing_coordinate_flag"] = lat.isna() | lon.isna()
    out["zero_coordinate_flag"] = (lat == 0) | (lon == 0)
    out["positive_longitude_flag"] = lon.notna() & (lon > 0)

    # Approximate California bounding box used for plausibility checks.
    out["implausible_coordinate_flag"] = (
        lat.notna()
        & lon.notna()
        & ((lat < 32.0) | (lat > 42.5) | (lon < -125.0) | (lon > -113.0))
    )

    out["invalid_geographic_flag"] = (
        out["missing_coordinate_flag"]
        | out["zero_coordinate_flag"]
        | out["positive_longitude_flag"]
        | out["implausible_coordinate_flag"]
    )

    counts = {
        "missing_coordinate_flag": int(out["missing_coordinate_flag"].sum()),
        "zero_coordinate_flag": int(out["zero_coordinate_flag"].sum()),
        "positive_longitude_flag": int(out["positive_longitude_flag"].sum()),
        "implausible_coordinate_flag": int(out["implausible_coordinate_flag"].sum()),
        "invalid_geographic_flag": int(out["invalid_geographic_flag"].sum()),
    }
    return out, counts


def _dtype_confirmation(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in columns:
        if col in df.columns:
            out[col] = str(df[col].dtype)
    return out


def clean_mls_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = config or CleaningConfig()
    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
    }

    work, parse_failures = parse_date_columns(df, cfg.date_columns)
    summary["date_parse_failures"] = parse_failures

    work, dropped_info = drop_redundant_columns(
        work,
        drop_missing_column_threshold=cfg.drop_missing_column_threshold,
        extra_drop_columns=cfg.extra_drop_columns,
    )
    summary["dropped_columns"] = dropped_info

    work, filled_counts = fill_missing_values(work, cfg.missing_fill_map)
    summary["missing_values_filled"] = filled_counts

    work, coercion_new_nulls = coerce_numeric_columns(work, cfg.numeric_columns)
    summary["numeric_coercion_new_nulls"] = coercion_new_nulls

    work, invalid_numeric_counts = add_invalid_numeric_flags(work)
    summary["invalid_numeric_counts"] = invalid_numeric_counts

    work, invalid_rows_removed = apply_invalid_numeric_strategy(
        work,
        strategy=cfg.invalid_numeric_strategy,
    )
    summary["invalid_numeric_strategy"] = cfg.invalid_numeric_strategy
    summary["rows_removed_for_invalid_numeric"] = int(invalid_rows_removed)

    work, date_flag_counts = add_date_consistency_flags(work)
    summary["date_consistency_flag_counts"] = date_flag_counts

    work, geo_flag_counts = add_geographic_quality_flags(work)
    summary["geographic_flag_counts"] = geo_flag_counts

    summary["rows_after"] = int(len(work))
    summary["columns_after"] = int(len(work.columns))
    summary["dtype_confirmation"] = _dtype_confirmation(
        work,
        cfg.date_columns + cfg.numeric_columns,
    )
    summary["remaining_null_counts_focus_fields"] = {
        col: int(work[col].isna().sum())
        for col in _existing_columns(work, cfg.date_columns + cfg.numeric_columns)
    }

    return work, summary


def process_mls_files(
    sold_input_path: str | Path,
    listings_input_path: str | Path,
    sold_output_path: str | Path,
    listings_output_path: str | Path,
    report_output_path: str | Path,
    config: Optional[CleaningConfig] = None,
) -> Dict[str, Any]:
    sold_raw = load_dataset(sold_input_path)
    listings_raw = load_dataset(listings_input_path)

    sold_clean, sold_summary = clean_mls_dataframe(
        sold_raw,
        dataset_name="sold",
        config=config,
    )
    listings_clean, listings_summary = clean_mls_dataframe(
        listings_raw,
        dataset_name="listings",
        config=config,
    )

    sold_clean.to_csv(sold_output_path, index=False)
    listings_clean.to_csv(listings_output_path, index=False)

    full_summary = {
        "sold": sold_summary,
        "listings": listings_summary,
        "outputs": {
            "sold_output_path": str(sold_output_path),
            "listings_output_path": str(listings_output_path),
            "report_output_path": str(report_output_path),
        },
    }

    Path(report_output_path).write_text(
        json.dumps(full_summary, indent=2, default=str),
        encoding="utf-8",
    )
    return full_summary


def summarize_for_console(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    for dataset in ["sold", "listings"]:
        s = summary[dataset]
        lines.append(f"[{dataset.upper()}]")
        lines.append(f"rows: {s['rows_before']} -> {s['rows_after']}")
        lines.append(f"cols: {s['columns_before']} -> {s['columns_after']}")
        lines.append(f"invalid numeric removed: {s['rows_removed_for_invalid_numeric']}")
        lines.append("dtype confirmation (focus fields):")
        for col, dtype in s["dtype_confirmation"].items():
            lines.append(f"  - {col}: {dtype}")
        lines.append("date consistency flags:")
        for k, v in s["date_consistency_flag_counts"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("geographic quality flags:")
        for k, v in s["geographic_flag_counts"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    return "\n".join(lines)
