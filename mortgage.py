import pandas as pd
from typing import Tuple


def fetch_and_resample_mortgage_rates(url: str) -> pd.DataFrame:
    """
    Fetch mortgage rate data from FRED API and resample to monthly averages.
    
    Args:
        url: URL to the FRED mortgage data CSV
        
    Returns:
        DataFrame with year_month and rate_30yr_fixed columns
    """
    mortgage = pd.read_csv(url, parse_dates=['observation_date'])
    mortgage.columns = ['date', 'rate_30yr_fixed']
    mortgage['year_month'] = mortgage['date'].dt.to_period('M')
    
    mortgage_monthly = (
        mortgage.groupby('year_month')['rate_30yr_fixed']
        .mean()
        .reset_index()
    )
    
    print(f"✓ Fetched mortgage rates: {len(mortgage_monthly)} months of data")
    return mortgage_monthly


def load_and_combine_datasets(file_pairs: list) -> pd.DataFrame:
    """
    Load multiple CSV files and combine them into a single DataFrame.
    
    Args:
        file_pairs: List of file paths to load and concatenate
        
    Returns:
        Combined DataFrame
    """
    dataframes = [pd.read_csv(f) for f in file_pairs]
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"✓ Combined {len(file_pairs)} files -> {len(combined)} total rows")
    return combined


def add_year_month_column(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Add a year_month period column to a DataFrame.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column to convert
        
    Returns:
        DataFrame with added year_month column
    """
    df = df.copy()
    df['year_month'] = pd.to_datetime(df[date_column]).dt.to_period('M')
    return df


def merge_rates_with_validation(
    data_df: pd.DataFrame, 
    rates_df: pd.DataFrame,
    rate_column: str = 'rate_30yr_fixed',
    merge_key: str = 'year_month'
) -> pd.DataFrame:
    """
    Merge mortgage rates onto a dataset with validation for null values.
    
    Args:
        data_df: Dataset to enrich with rates
        rates_df: Mortgage rates DataFrame
        rate_column: Name of the rate column
        merge_key: Column to merge on
        
    Returns:
        Merged DataFrame with rates
        
    Raises:
        ValueError: If any null rates exist after merge
    """
    merged = data_df.merge(rates_df, on=merge_key, how='left')
    
    null_count = merged[rate_column].isnull().sum()
    print(f"  - Null rate values after merge: {null_count}")
    
    if null_count > 0:
        raise ValueError(
            f"VALIDATION FAILED: {null_count} null values found in {rate_column} after merge"
        )
    
    print(f"✓ Merge successful with no null rate values")
    return merged


def save_enriched_datasets(
    sold_df: pd.DataFrame,
    listings_df: pd.DataFrame,
    sold_output: str = 'sold_with_rates.csv',
    listings_output: str = 'listings_with_rates.csv'
) -> None:
    """
    Save enriched datasets to CSV files.
    
    Args:
        sold_df: Enriched sold dataset
        listings_df: Enriched listings dataset
        sold_output: Output file path for sold data
        listings_output: Output file path for listings data
    """
    sold_df.to_csv(sold_output, index=False)
    listings_df.to_csv(listings_output, index=False)
    print(f"✓ Saved {sold_output} ({len(sold_df)} rows)")
    print(f"✓ Saved {listings_output} ({len(listings_df)} rows)")


def enrich_real_estate_data(
    mortgage_url: str,
    sold_files: list,
    listings_files: list,
    sold_date_column: str = 'CloseDate',
    listings_date_column: str = 'ListingContractDate',
    sold_output: str = 'sold_with_rates.csv',
    listings_output: str = 'listings_with_rates.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main orchestration function: fetch mortgage rates, combine datasets, 
    add year_month, merge with validation, and save results.
    
    Args:
        mortgage_url: URL to FRED mortgage data
        sold_files: List of sold CSV file paths
        listings_files: List of listings CSV file paths
        sold_date_column: Date column name in sold data
        listings_date_column: Date column name in listings data
        sold_output: Output file path for enriched sold data
        listings_output: Output file path for enriched listings data
        
    Returns:
        Tuple of (enriched_sold_df, enriched_listings_df)
    """
    print("Starting real estate data enrichment pipeline...\n")
    
    # Mission 1 & 2: Fetch and resample to monthly averages
    print("(1) Fetching and resampling mortgage rates...")
    mortgage_monthly = fetch_and_resample_mortgage_rates(mortgage_url)
    
    # Load and combine datasets
    print("\nLoading sold and listings datasets...")
    sold = load_and_combine_datasets(sold_files)
    listings = load_and_combine_datasets(listings_files)
    
    # Add year_month columns
    print("\nAdding year_month columns...")
    sold = add_year_month_column(sold, sold_date_column)
    listings = add_year_month_column(listings, listings_date_column)
    print(f"✓ Added year_month to sold data")
    print(f"✓ Added year_month to listings data")
    
    # Mission 3: Merge onto both datasets
    print("\n(3) Merging mortgage rates onto datasets...")
    print("Sold dataset:")
    sold_with_rates = merge_rates_with_validation(sold, mortgage_monthly)
    
    print("Listings dataset:")
    listings_with_rates = merge_rates_with_validation(listings, mortgage_monthly)
    
    # Mission 4: Save enriched datasets
    print("\n(4) Saving enriched datasets...")
    save_enriched_datasets(sold_with_rates, listings_with_rates, sold_output, listings_output)
    
    print("\n✓ Pipeline complete!")
    return sold_with_rates, listings_with_rates


if __name__ == '__main__':
    # Execute the pipeline
    sold_with_rates, listings_with_rates = enrich_real_estate_data(
        mortgage_url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US',
        sold_files=['CRMLSSold202602.csv', 'CRMLSSold202603.csv'],
        listings_files=['CRMLSListing202602.csv', 'CRMLSListing202603.csv'],
    )
    
    # Preview
    print("\n--- Sold Data Preview ---")
    print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())
    
    print("\n--- Listings Data Preview ---")
    print(listings_with_rates[['ListingContractDate', 'year_month', 'rate_30yr_fixed']].head())

