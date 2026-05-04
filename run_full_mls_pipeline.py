from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idx.mls_data_cleaning_pipeline import CleaningConfig, load_dataset, process_mls_files, summarize_for_console
from idx.mortgage import enrich_real_estate_data

engineer_and_summarize = importlib.import_module("mls_feature_engineering").engineer_and_summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full MLS pipeline: mortgage enrichment + configurable cleaning.",
    )

    parser.add_argument(
        "--mortgage-url",
        default="https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
        help="FRED mortgage CSV URL.",
    )
    parser.add_argument(
        "--sold-files",
        nargs="+",
        default=["CRMLSSold202602.csv", "CRMLSSold202603.csv"],
        help="Input sold CSV files to combine.",
    )
    parser.add_argument(
        "--listings-files",
        nargs="+",
        default=["CRMLSListing202602.csv", "CRMLSListing202603.csv"],
        help="Input listings CSV files to combine.",
    )
    parser.add_argument(
        "--sold-date-column",
        default="CloseDate",
        help="Date column used to derive year_month in sold data.",
    )
    parser.add_argument(
        "--listings-date-column",
        default="ListingContractDate",
        help="Date column used to derive year_month in listings data.",
    )

    parser.add_argument(
        "--sold-with-rates-output",
        default="sold_with_rates.csv",
        help="Output CSV path for sold data enriched with rates.",
    )
    parser.add_argument(
        "--listings-with-rates-output",
        default="listings_with_rates.csv",
        help="Output CSV path for listings data enriched with rates.",
    )
    parser.add_argument(
        "--cleaned-sold-output",
        default="cleaned_sold_analysis_ready.csv",
        help="Output CSV path for cleaned sold data.",
    )
    parser.add_argument(
        "--cleaned-listings-output",
        default="cleaned_listings_analysis_ready.csv",
        help="Output CSV path for cleaned listings data.",
    )
    parser.add_argument(
        "--report-output",
        default="mls_cleaning_report.json",
        help="Output JSON path for cleaning report.",
    )
    parser.add_argument(
        "--engineered-sold-output",
        default="engineered_sold_analysis_ready.csv",
        help="Output CSV path for feature-engineered sold data.",
    )
    parser.add_argument(
        "--engineered-listings-output",
        default="engineered_listings_analysis_ready.csv",
        help="Output CSV path for feature-engineered listings data.",
    )
    parser.add_argument(
        "--feature-engineering-report-output",
        default="feature_engineering_report.json",
        help="Output JSON path for feature engineering report.",
    )
    parser.add_argument(
        "--feature-engineering-output-dir",
        default=".",
        help="Directory for segment summary CSV/JSON files.",
    )
    parser.add_argument(
        "--disable-feature-engineering",
        action="store_true",
        help="Skip feature engineering and segment summaries.",
    )

    parser.add_argument(
        "--invalid-numeric-strategy",
        choices=["remove", "flag"],
        default="remove",
        help="How to handle invalid numeric rows.",
    )
    parser.add_argument(
        "--drop-missing-column-threshold",
        type=float,
        default=0.5,
        help="Drop columns with missing-rate >= this threshold.",
    )
    parser.add_argument(
        "--high-cardinality-threshold",
        type=float,
        default=0.9,
        help="Drop object columns with unique-ratio > this threshold.",
    )
    parser.add_argument(
        "--disable-high-cardinality-drop",
        action="store_true",
        help="Disable high-cardinality object column dropping.",
    )

    parser.add_argument(
        "--only-residential",
        action="store_true",
        default=True,
        help="Keep only rows where PropertyType is in residential types.",
    )
    parser.add_argument(
        "--include-all-property-types",
        action="store_true",
        help="Disable residential-only filtering.",
    )
    parser.add_argument(
        "--residential-property-types",
        nargs="+",
        default=["Residential"],
        help="Allowed property types when residential filter is enabled.",
    )
    parser.add_argument(
        "--extra-drop-columns",
        nargs="*",
        default=[],
        help="Optional extra columns to drop.",
    )

    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate histogram and boxplot image files.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=".",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--plot-columns",
        nargs="+",
        default=[
            "ClosePrice",
            "ListPrice",
            "OriginalListPrice",
            "LivingArea",
            "LotSizeAcres",
            "BedroomsTotal",
            "BathroomsTotalInteger",
            "DaysOnMarket",
            "YearBuilt",
        ],
        help="Columns used for optional distribution plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting full pipeline: mortgage enrichment + MLS cleaning\n")

    sold_with_rates, listings_with_rates = enrich_real_estate_data(
        mortgage_url=args.mortgage_url,
        sold_files=args.sold_files,
        listings_files=args.listings_files,
        sold_date_column=args.sold_date_column,
        listings_date_column=args.listings_date_column,
        sold_output=args.sold_with_rates_output,
        listings_output=args.listings_with_rates_output,
    )

    print("\nMortgage enrichment complete.")
    print(f"Sold rows with rates: {len(sold_with_rates)}")
    print(f"Listings rows with rates: {len(listings_with_rates)}")

    high_cardinality_threshold = args.high_cardinality_threshold
    if args.disable_high_cardinality_drop:
        high_cardinality_threshold = None

    only_residential = args.only_residential and (not args.include_all_property_types)

    config = CleaningConfig(
        invalid_numeric_strategy=args.invalid_numeric_strategy,
        drop_missing_column_threshold=args.drop_missing_column_threshold,
        extra_drop_columns=args.extra_drop_columns,
        only_residential=only_residential,
        residential_property_types=args.residential_property_types,
        high_cardinality_threshold=high_cardinality_threshold,
        generate_plots=args.generate_plots,
        plot_output_dir=args.plot_output_dir,
        plot_columns=args.plot_columns,
    )

    summary = process_mls_files(
        sold_input_path=args.sold_with_rates_output,
        listings_input_path=args.listings_with_rates_output,
        sold_output_path=args.cleaned_sold_output,
        listings_output_path=args.cleaned_listings_output,
        report_output_path=args.report_output,
        config=config,
    )

    print("\nCleaning complete.")
    print(summarize_for_console(summary))

    feature_engineering_summary = None
    if not args.disable_feature_engineering:
        cleaned_sold_df = load_dataset(args.cleaned_sold_output)
        cleaned_listings_df = load_dataset(args.cleaned_listings_output)

        engineered_sold, sold_feature_summary = engineer_and_summarize(
            cleaned_sold_df,
            output_dir=args.feature_engineering_output_dir,
            prefix="sold",
        )
        engineered_listings, listings_feature_summary = engineer_and_summarize(
            cleaned_listings_df,
            output_dir=args.feature_engineering_output_dir,
            prefix="listings",
        )

        engineered_sold.to_csv(args.engineered_sold_output, index=False)
        engineered_listings.to_csv(args.engineered_listings_output, index=False)

        feature_engineering_summary = {
            "sold": sold_feature_summary,
            "listings": listings_feature_summary,
            "outputs": {
                "engineered_sold_output": args.engineered_sold_output,
                "engineered_listings_output": args.engineered_listings_output,
            },
        }

        Path(args.feature_engineering_report_output).write_text(
            json.dumps(feature_engineering_summary, indent=2, default=str),
            encoding="utf-8",
        )

    print(f"Saved: {args.sold_with_rates_output}")
    print(f"Saved: {args.listings_with_rates_output}")
    print(f"Saved: {args.cleaned_sold_output}")
    print(f"Saved: {args.cleaned_listings_output}")
    print(f"Saved: {args.report_output}")
    if feature_engineering_summary is not None:
        print(f"Saved: {args.engineered_sold_output}")
        print(f"Saved: {args.engineered_listings_output}")
        print(f"Saved: {args.feature_engineering_report_output}")


if __name__ == "__main__":
    main()
