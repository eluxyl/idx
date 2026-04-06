import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(file_path, only_residential=True):
    """
    Cleaning Sold/Listing dataset
    Input: file_path readable by pd.read_csv
    Output: cleaned DataFrame
            saved histograms and boxplots of selected values
    If you do not want to remove certain columns, comment out the drop lines
    """
    # Load the dataset
    sold = pd.read_csv(file_path)
    

    # Drop Columns with overlapping data
    cols_to_drop = [col for col in sold.columns if col.endswith('.1')]
    sold = sold.drop(columns=cols_to_drop)



    # Identify columns where more than 90% of values are unique strings
    high_cardinality = [col for col in sold.select_dtypes('object').columns 
                        if sold[col].nunique() / len(sold) > 0.9]
    print(f"Suggested drops: {high_cardinality}")

    sold=sold.drop(columns=high_cardinality)
    len(sold.columns)


    # Drop columns that are more than 50% NaN
    print("List of columns with more than half of NaN values:")
    print(sold.columns[sold.isna().mean() > 0.5].tolist())
    limit = len(sold) * 0.5
    sold = sold.dropna(thresh=limit, axis=1)

    len(sold.columns)


    print(f"Unique Property Types: {sold['PropertyType'].unique()}")

    if only_residential:
        # Filter to only include residential properties
        residential_types = ['Residential']
        sold = sold[sold['PropertyType'].isin(residential_types)]
        print(f"Filtered to residential properties. Remaining unique types: {sold['PropertyType'].unique()}")




    #Here we plot out all the stuff
    #Define your key columns--Change it if you need other columns
    target_cols = [
        'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
        'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger', 
        'DaysOnMarket', 'YearBuilt'
    ]

    # 1. Generate Percentile Distribution Table
    # We include 0.25 to 0.75 to catch outliers
    percentiles = [0.25,0.75]
    distribution_table = sold[target_cols].describe(percentiles=percentiles).T

    print("--- Percentile Distribution ---")
    print(distribution_table)


    # Set up the figure
    plt.style.use('seaborn-v0_8-muted') # A cleaner look for distributions
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        # sns.histplot provides both the bars and the smooth KDE line
        sns.histplot(data=sold, x=col, kde=True, ax=axes[i], color='teal', bins=30)
        
        axes[i].set_title(f'Histogram: {col}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel('')

        # Custom handling for highly skewed data (Prices and Lot Size)
        # This prevents one massive bar from squishing the rest of the graph
        if col in ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LotSizeAcres']:
            axes[i].set_yscale('log')
            axes[i].set_title(f'Histogram: {col} (Log Frequency)')

    plt.tight_layout()
    plt.savefig('real_estate_distributions_histogram.png')




    #For boxplots, we use ggplot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(target_cols):
        sns.boxplot(data=sold, x=col, ax=axes[i], color='#3498db', fliersize=3)
        axes[i].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        
        # Custom Tip: Using log scale for LotSizeAcres if outliers are too high
        if col == 'LotSizeAcres':
            axes[i].set_xscale('log')
            axes[i].set_title(f'Distribution: {col} (Log Scale)')

    plt.tight_layout()
    plt.savefig('real_estate_distributions_boxplot.png')

    sold.to_csv("cleaned_sold.csv", index=False)

if __name__ == "__main__":
    clean_data("sold.csv", only_residential=True)