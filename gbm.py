import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

# Suppress lightgbm warnings for cleaner output
warnings.filterwarnings('ignore')

def walk_forward_cross_validation(df, target_col='ClosePrice', initial_train_months=1):
    """
    Performs sequential walk-forward cross-validation on pre-engineered real estate data.
    Implements target log-transformation to stabilize RMSE against outliers.
    """
    print("\nStarting Walk-Forward Cross-Validation...")
    
    # Ensure chronological order
    if 'CloseDate' in df.columns:
        df['CloseDate'] = pd.to_datetime(df['CloseDate'])
        df = df.sort_values('CloseDate').reset_index(drop=True)
        
    # Sort months chronologically
    unique_months = sorted(df['year_month'].dropna().unique())
    
    if len(unique_months) <= initial_train_months:
        raise ValueError(f"Not enough months ({len(unique_months)}) for initial training threshold.")

    # Select predictive features (Matches the output of your new engineering script)
    features = [
        'OriginalListPrice', 'ListPrice', 'Latitude', 'Longitude', 'PropertyType', 
        'LivingArea', 'DaysOnMarket', 'FireplacesTotal', 'TaxAnnualAmount', 
        'YearBuilt', 'BathroomsTotalInteger', 'City', 'BedroomsTotal', 'PostalCode', 
        'rate_30yr_fixed', 'LotSizeAcres', 'mortgage_rate_mom_change', 'postal_prev_month_median'
    ]
    
    categorical_features = ['PostalCode', 'City', 'PropertyType']
    
    # Enforce categorical data types for LightGBM native handling
    # Explicitly cast to string first to prevent mixed-type errors
    for cat in categorical_features:
        if cat in df.columns:
            df[cat] = df[cat].astype(str).astype('category')

    results = []

    # Walk-forward loop
    for i in range(initial_train_months, len(unique_months)):
        train_months = unique_months[:i]
        test_month = unique_months[i]
        
        train_data = df[df['year_month'].isin(train_months)].dropna(subset=[target_col])
        test_data = df[df['year_month'] == test_month].dropna(subset=[target_col])
        
        # Skip iteration if either set is empty
        if len(train_data) == 0 or len(test_data) == 0:
            continue
            
        X_train, y_train = train_data[features], train_data[target_col]
        X_test, y_test = test_data[features], test_data[target_col]
        
        # --- SOTA LOG TRANSFORMATION EFFECT ---
        # Compresses the target variance so remaining outliers don't break the RMSE
        y_train_log = np.log1p(y_train)
        
        # Initialize LightGBM Regressor optimized for CPU execution
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(
            X_train, y_train_log,
            categorical_feature=[c for c in categorical_features if c in features]
        )
        
        # Predict outputs on log scale
        preds_log = model.predict(X_test)
        
        # Reverse the log-transform back to true dollar values before scoring
        preds = np.expm1(preds_log)
        
        # Evaluation metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"Tested on {test_month} | Train size: {len(X_train)} | Test size: {len(X_test)} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")
        
        results.append({'test_month': test_month, 'mae': mae, 'rmse': rmse})
        
    # Summary reporting
    if results:
        results_df = pd.DataFrame(results)
        print("\n--- Cross-Validation Summary ---")
        print(f"Average MAE:  ${results_df['mae'].mean():,.2f}")
        print(f"Average RMSE: ${results_df['rmse'].mean():,.2f}")
    else:
        print("No validation loops executed. Check your data range parameters.")
    
    return model, features

# --- Execution Entry Point ---
if __name__ == "__main__":
    try:
        # Load the fully processed file directly
        file_name = 'engineered_sold_analysis_ready_iqr_filtered.csv'
        print(f"Loading {file_name}...")
        df = pd.read_csv(file_name)
        
        # Pass directly to validation (Engineering is already complete)
        final_model, used_features = walk_forward_cross_validation(df)
        
    except FileNotFoundError:
        print("Error: 'engineered_sold_analysis_ready_iqr_filtered.csv' not found.")
        print("Please run your apply_iqr_and_engineer.py script first.")