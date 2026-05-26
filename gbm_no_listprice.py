import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

# Suppress lightgbm warnings for cleaner output
warnings.filterwarnings('ignore')

def walk_forward_cross_validation(df, target_col='ClosePrice', initial_train_months=1):
    """
    Walk-forward cross-validation variant that excludes the listed price (`ListPrice`) from predictors.
    """
    print("\nStarting Walk-Forward Cross-Validation (no ListPrice)...")
    
    # Ensure chronological order
    if 'CloseDate' in df.columns:
        df['CloseDate'] = pd.to_datetime(df['CloseDate'])
        df = df.sort_values('CloseDate').reset_index(drop=True)
        
    # Sort months chronologically
    unique_months = sorted(df['year_month'].dropna().unique())
    
    if len(unique_months) <= initial_train_months:
        raise ValueError(f"Not enough months ({len(unique_months)}) for initial training threshold.")

    # Select predictive features with `ListPrice` and `OriginalListPrice` removed
    features = [
        'Latitude', 'Longitude', 'PropertyType', 
        'LivingArea', 'DaysOnMarket', 'FireplacesTotal', 'TaxAnnualAmount', 
        'YearBuilt', 'BathroomsTotalInteger', 'City', 'BedroomsTotal', 'PostalCode', 
        'rate_30yr_fixed', 'LotSizeAcres', 'mortgage_rate_mom_change', 'postal_prev_month_median'
    ]
    
    categorical_features = ['PostalCode', 'City', 'PropertyType']
    
    # Enforce categorical data types for LightGBM native handling
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
        
        # Log-transform the target to stabilize RMSE
        y_train_log = np.log1p(y_train)
        
        # Initialize LightGBM Regressor
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
        
        # Predict and invert log-transform
        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)
        
        # Evaluation metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"Tested on {test_month} | Train size: {len(X_train)} | Test size: {len(X_test)} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")
        
        results.append({'test_month': test_month, 'mae': mae, 'rmse': rmse})
        
    # Summary reporting
    if results:
        results_df = pd.DataFrame(results)
        print("\n--- Cross-Validation Summary (no ListPrice) ---")
        print(f"Average MAE:  ${results_df['mae'].mean():,.2f}")
        print(f"Average RMSE: ${results_df['rmse'].mean():,.2f}")
    else:
        print("No validation loops executed. Check your data range parameters.")
    
    return model, features


if __name__ == "__main__":
    try:
        file_name = 'engineered_sold_analysis_ready_iqr_filtered.csv'
        print(f"Loading {file_name}...")
        df = pd.read_csv(file_name)
        
        final_model, used_features = walk_forward_cross_validation(df)
        
    except FileNotFoundError:
        print("Error: 'engineered_sold_analysis_ready_iqr_filtered.csv' not found.")
        print("Please run your apply_iqr_and_engineer.py script first.")
