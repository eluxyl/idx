# MLS Data Enrichment and Cleaning Pipeline

This folder now has a reusable, function-based pipeline for:

1. Enriching MLS sold/listings data with monthly mortgage rates from FRED
2. Cleaning and validating MLS data for analysis-ready output
3. Producing a data quality report with row counts, dtypes, and validation flags

## Files and Purpose

- `mortgage.py`
  - Fetches mortgage rates from FRED
  - Resamples to monthly averages
  - Merges rates onto sold/listings datasets by `year_month`
  - Validates that merged rate values are not null

- `mls_data_cleaning_pipeline.py`
  - Reusable cleaning functions
  - Date parsing, type coercion, missing value handling
  - Invalid numeric checks and date/geographic quality flags
  - Exports cleaned datasets and a JSON quality report

- `run_mls_cleaning.py`
  - Cleaning-only runner
  - Uses `sold_with_rates.csv` and `listings_with_rates.csv` if available
  - Falls back to `sold.csv` and `listings.csv`

- `run_full_mls_pipeline.py`
  - End-to-end runner
  - Runs mortgage enrichment first, then cleaning
  - Includes argparse options for plot generation, cleaning, and feature engineering

- `mls_feature_engineering.py`
  - Builds dashboard metrics from cleaned MLS data
  - Generates segment summaries by property, location, and office dimensions

## Quick Start

Run from this folder:

```bash
cd /accounts/masters/gongyaoxu/idx
```

### Option A: End-to-End Pipeline (recommended)

```bash
python run_full_mls_pipeline.py
```

Useful options (all have defaults):

```bash
python run_full_mls_pipeline.py \
  --generate-plots \
  --feature-engineering-output-dir . \
  --plot-output-dir . \
  --invalid-numeric-strategy remove \
  --drop-missing-column-threshold 0.5 \
  --high-cardinality-threshold 0.9
```

Property-type options:

```bash
# Default behavior: keep Residential only
python run_full_mls_pipeline.py

# Keep all property types
python run_full_mls_pipeline.py --include-all-property-types
```

Feature engineering runs by default and creates:

- `engineered_sold_analysis_ready.csv`
- `engineered_listings_analysis_ready.csv`
- `feature_engineering_report.json`
- `sold_property_segment_summary.csv` and `.json`
- `sold_location_segment_summary.csv` and `.json`
- `sold_office_segment_summary.csv` and `.json`
- `listings_property_segment_summary.csv` and `.json`
- `listings_location_segment_summary.csv` and `.json`
- `listings_office_segment_summary.csv` and `.json`
 - `*_with_outlier_flags.csv` and `*_iqr_filtered.csv` when IQR detection is enabled

### CLI Reference (run_full_mls_pipeline.py)

| Argument | Default | Purpose |
|---|---|---|
| `--mortgage-url` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US` | FRED mortgage CSV source |
| `--sold-files` | `CRMLSSold202602.csv CRMLSSold202603.csv` | Sold input CSV files to combine |
| `--listings-files` | `CRMLSListing202602.csv CRMLSListing202603.csv` | Listings input CSV files to combine |
| `--sold-date-column` | `CloseDate` | Sold date used to create `year_month` |
| `--listings-date-column` | `ListingContractDate` | Listings date used to create `year_month` |
| `--sold-with-rates-output` | `sold_with_rates.csv` | Enriched sold output path |
| `--listings-with-rates-output` | `listings_with_rates.csv` | Enriched listings output path |
| `--cleaned-sold-output` | `cleaned_sold_analysis_ready.csv` | Final cleaned sold output path |
| `--cleaned-listings-output` | `cleaned_listings_analysis_ready.csv` | Final cleaned listings output path |
| `--report-output` | `mls_cleaning_report.json` | Cleaning report JSON path |
| `--engineered-sold-output` | `engineered_sold_analysis_ready.csv` | Feature-engineered sold output path |
| `--engineered-listings-output` | `engineered_listings_analysis_ready.csv` | Feature-engineered listings output path |
| `--feature-engineering-report-output` | `feature_engineering_report.json` | Feature engineering summary JSON path |
| `--feature-engineering-output-dir` | `.` | Directory for segment summary CSV/JSON files |
| `--disable-feature-engineering` | `False` | Skip engineered outputs and segment summaries |
| `--invalid-numeric-strategy` | `remove` | Either `remove` or `flag` invalid numeric rows |
| `--drop-missing-column-threshold` | `0.5` | Drop columns with missing rate >= threshold |
| `--high-cardinality-threshold` | `0.9` | Drop object columns when unique-ratio exceeds threshold |
| `--disable-high-cardinality-drop` | `False` | Turn off high-cardinality dropping |
| `--only-residential` | `True` | Keep only residential rows (unless overridden below) |
| `--include-all-property-types` | `False` | Disable residential-only filtering |
| `--residential-property-types` | `Residential` | Allowed values when residential filter is enabled |
| `--extra-drop-columns` | *(empty)* | Additional columns to drop |
| `--generate-plots` | `False` | Generate histogram and boxplot PNGs |
| `--plot-output-dir` | `.` | Directory where plots are saved |
| `--plot-columns` | `ClosePrice ListPrice OriginalListPrice LivingArea LotSizeAcres BedroomsTotal BathroomsTotalInteger DaysOnMarket YearBuilt` | Columns used for optional plots |
| `--iqr-enable` | `False` | Enable IQR outlier detection and save flagged/filtered outputs |
| `--iqr-multiplier` | `1.5` | IQR multiplier used to compute lower/upper bounds |
| `--iqr-columns` | `ClosePrice LivingArea DaysOnMarket` | Columns to apply IQR detection to (defaults used if omitted) |

This will generate:

- `sold_with_rates.csv`
- `listings_with_rates.csv`
- `cleaned_sold_analysis_ready.csv`
- `cleaned_listings_analysis_ready.csv`
- `mls_cleaning_report.json`
- `engineered_sold_analysis_ready.csv`
- `engineered_listings_analysis_ready.csv`
- `feature_engineering_report.json`
- segment summary CSV/JSON files for property, location, and office groupings

### Option B: Cleaning Only

```bash
python run_mls_cleaning.py
```

## Function Usage Examples

### 1) Mortgage Enrichment Functions

```python
from mortgage import enrich_real_estate_data

sold_with_rates, listings_with_rates = enrich_real_estate_data(
    mortgage_url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
    sold_files=["CRMLSSold202602.csv", "CRMLSSold202603.csv"],
    listings_files=["CRMLSListing202602.csv", "CRMLSListing202603.csv"],
    sold_date_column="CloseDate",
    listings_date_column="ListingContractDate",
    sold_output="sold_with_rates.csv",
    listings_output="listings_with_rates.csv",
)
```

### 2) Cleaning Pipeline Functions

```python
from mls_data_cleaning_pipeline import CleaningConfig, process_mls_files

config = CleaningConfig(
    invalid_numeric_strategy="remove",   # or "flag"
    drop_missing_column_threshold=0.98,
)

summary = process_mls_files(
    sold_input_path="sold_with_rates.csv",
    listings_input_path="listings_with_rates.csv",
    sold_output_path="cleaned_sold_analysis_ready.csv",
    listings_output_path="cleaned_listings_analysis_ready.csv",
    report_output_path="mls_cleaning_report.json",
    config=config,
)

print(summary["sold"]["rows_before"], "->", summary["sold"]["rows_after"])
print(summary["listings"]["rows_before"], "->", summary["listings"]["rows_after"])
```

### 3) Cleaning a DataFrame Directly

```python
import pandas as pd
from mls_data_cleaning_pipeline import CleaningConfig, clean_mls_dataframe

raw_df = pd.read_csv("sold_with_rates.csv")
config = CleaningConfig(invalid_numeric_strategy="remove")
clean_df, clean_summary = clean_mls_dataframe(raw_df, dataset_name="sold", config=config)
```

## What Gets Validated During Cleaning

### Date conversion

- `CloseDate`
- `PurchaseContractDate`
- `ListingContractDate`
- `ContractStatusChangeDate`

### Numeric validity checks

- `ClosePrice <= 0`
- `LivingArea <= 0`
- `DaysOnMarket < 0`
- `BedroomsTotal < 0`
- `BathroomsTotalInteger < 0`

### Date consistency flags

- `listing_after_close_flag`
- `purchase_after_close_flag`
- `negative_timeline_flag`

### Geographic quality flags

- `missing_coordinate_flag`
- `zero_coordinate_flag`
- `positive_longitude_flag`
- `implausible_coordinate_flag`
- `invalid_geographic_flag`

## Report Contents

`mls_cleaning_report.json` includes:

- before/after row and column counts
- date parse failure counts
- columns dropped
- missing value fill counts
- numeric coercion null impact
- invalid numeric counts and removed rows
- date consistency flag counts
- geographic flag counts
- dtype confirmation for key date/numeric fields

## IQR Outlier Detection (optional)

When enabled via `--iqr-enable`, the cleaning pipeline computes IQR-based bounds for the configured columns (default: `ClosePrice`, `LivingArea`, `DaysOnMarket`) using the multiplier from `--iqr-multiplier` (default `1.5`). The pipeline will:

- Add boolean flag columns named `<Column>_outlier_flag` to indicate rows outside the IQR bounds.
- Save a flagged CSV alongside the normal cleaned output named `<cleaned_output>_with_outlier_flags.csv`.
- Save a filtered CSV removing any rows flagged as IQR outliers or previously-marked invalid numeric rows named `<cleaned_output>_iqr_filtered.csv`.
- Add an `iqr` section to `mls_cleaning_report.json` with per-column Q1/Q3/IQR/lower/upper, counts and percentages of outliers, medians before and after filtering, and paths to the saved flagged/filtered files.

Use example:

```bash
python run_full_mls_pipeline.py --iqr-enable --iqr-multiplier 1.5
```

## Notes

- `invalid_numeric_strategy="remove"` removes invalid rows after flagging them.
- `invalid_numeric_strategy="flag"` keeps all rows and only marks invalid records.
- The geographic plausibility check uses an approximate California bounding box.
