from __future__ import annotations

from pathlib import Path

from mls_data_cleaning_pipeline import CleaningConfig, process_mls_files, summarize_for_console


def choose_input(preferred: str, fallback: str) -> str:
    if Path(preferred).exists():
        return preferred
    return fallback


def main() -> None:
    sold_input = choose_input("sold_with_rates.csv", "sold.csv")
    listings_input = choose_input("listings_with_rates.csv", "listings.csv")

    config = CleaningConfig(
        invalid_numeric_strategy="remove",
        drop_missing_column_threshold=0.98,
    )

    summary = process_mls_files(
        sold_input_path=sold_input,
        listings_input_path=listings_input,
        sold_output_path="cleaned_sold_analysis_ready.csv",
        listings_output_path="cleaned_listings_analysis_ready.csv",
        report_output_path="mls_cleaning_report.json",
        config=config,
    )

    print("Cleaning complete.")
    print(summarize_for_console(summary))
    print("Saved: cleaned_sold_analysis_ready.csv")
    print("Saved: cleaned_listings_analysis_ready.csv")
    print("Saved: mls_cleaning_report.json")


if __name__ == "__main__":
    main()
