from __future__ import annotations

from mls_data_cleaning_pipeline import CleaningConfig, process_mls_files, summarize_for_console
from mortgage import enrich_real_estate_data


def main() -> None:
    print("Starting full pipeline: mortgage enrichment + MLS cleaning\n")

    sold_with_rates, listings_with_rates = enrich_real_estate_data(
        mortgage_url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
        sold_files=["CRMLSSold202602.csv", "CRMLSSold202603.csv"],
        listings_files=["CRMLSListing202602.csv", "CRMLSListing202603.csv"],
        sold_date_column="CloseDate",
        listings_date_column="ListingContractDate",
        sold_output="sold_with_rates.csv",
        listings_output="listings_with_rates.csv",
    )

    print("\nMortgage enrichment complete.")
    print(f"Sold rows with rates: {len(sold_with_rates)}")
    print(f"Listings rows with rates: {len(listings_with_rates)}")

    config = CleaningConfig(
        invalid_numeric_strategy="remove",
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

    print("\nCleaning complete.")
    print(summarize_for_console(summary))
    print("Saved: sold_with_rates.csv")
    print("Saved: listings_with_rates.csv")
    print("Saved: cleaned_sold_analysis_ready.csv")
    print("Saved: cleaned_listings_analysis_ready.csv")
    print("Saved: mls_cleaning_report.json")


if __name__ == "__main__":
    main()
