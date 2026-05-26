import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

# Suppress LightGBM warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def walk_forward_cross_validation(df, target_col='ClosePrice', initial_train_months=1):
    """
    Performs sequential walk-forward cross-validation on pre-engineered data.
    Trains on log-transformed prices to stabilize RMSE and extracts Tableau-ready datasets.
    """
    print("\nStarting Walk-Forward Cross-Validation...")
    
    # 1. Ensure chronological order
    if 'CloseDate' in df.columns:
        df['CloseDate'] = pd.to_datetime(df['CloseDate'])
        df = df.sort_values('CloseDate').reset_index(drop=True)
        
    unique_months = sorted(df['year_month'].dropna().unique())
    
    if len(unique_months) <= initial_train_months:
        raise ValueError(f"Not enough months ({len(unique_months)}) for initial training threshold.")

    # 2. Define Features (Matches your pipeline's engineered schema)
    features = [
        'OriginalListPrice', 'ListPrice', 'Latitude', 'Longitude', 'PropertyType', 
        'LivingArea', 'DaysOnMarket', 'FireplacesTotal', 'TaxAnnualAmount', 
        'YearBuilt', 'BathroomsTotalInteger', 'City', 'BedroomsTotal', 'PostalCode', 
        'rate_30yr_fixed', 'LotSizeAcres', 'mortgage_rate_mom_change', 'postal_prev_month_median'
    ]
    
    categorical_features = ['PostalCode', 'City', 'PropertyType']
    
    # Clean categoricals for native LightGBM processing
    for cat in categorical_features:
        if cat in df.columns:
            df[cat] = df[cat].astype(str).astype('category')

    results = []
    all_predictions = [] 

    # 3. Walk-Forward Loop
    for i in range(initial_train_months, len(unique_months)):
        train_months = unique_months[:i]
        test_month = unique_months[i]
        
        train_data = df[df['year_month'].isin(train_months)].dropna(subset=[target_col])
        test_data = df[df['year_month'] == test_month].dropna(subset=[target_col])
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
            
        X_train, y_train = train_data[features], train_data[target_col]
        X_test, y_test = test_data[features], test_data[target_col]
        
        # Log-Transform Target
        y_train_log = np.log1p(y_train)
        
        # Initialize CPU-Optimized Model
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
        
        # Train
        model.fit(
            X_train, y_train_log,
            categorical_feature=[c for c in categorical_features if c in features]
        )
        
        # Predict & Reverse Log-Transform
        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)
        
        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"Tested on {test_month} | Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")
        
        results.append({'test_month': test_month, 'mae': mae, 'rmse': rmse})
        
        # 4. Extract Row-Level Data for Tableau
        tableau_batch = test_data.copy()
        tableau_batch['PredictedPrice'] = preds
        tableau_batch['Residual'] = preds - y_test
        tableau_batch['AbsoluteError'] = np.abs(preds - y_test)
        
        tableau_columns = [
            'ListingKey', 'CloseDate', 'year_month', 'PostalCode', 'City', 
            'PropertyType', 'ClosePrice', 'PredictedPrice', 'Residual', 'AbsoluteError',
            'Latitude', 'Longitude'
        ]
        
        available_columns = [col for col in tableau_columns if col in tableau_batch.columns]
        all_predictions.append(tableau_batch[available_columns])
        
    # 5. Summarize and Export
    if results:
        results_df = pd.DataFrame(results)
        print("\n--- Cross-Validation Summary ---")
        print(f"Average MAE:  ${results_df['mae'].mean():,.2f}")
        print(f"Average RMSE: ${results_df['rmse'].mean():,.2f}")
        
        print("\n--- Exporting Tableau Datasets ---")
        
        # Export Predictions & Residuals
        tableau_preds_df = pd.concat(all_predictions, ignore_index=True)
        tableau_preds_df.to_csv('tableau_predictions_tracker.csv', index=False)
        print("--> Saved 'tableau_predictions_tracker.csv' (Use for Geospatial & Actual vs. Predicted)")
        
        # Export Feature Importances
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_df.to_csv('tableau_feature_importance.csv', index=False)
        print("--> Saved 'tableau_feature_importance.csv' (Use for Driver Bar Charts)")
        
    else:
        print("No validation loops executed. Check your data range parameters.")
    
    return model, features

if __name__ == "__main__":
    file_name = 'engineered_sold_analysis_ready_iqr_filtered.csv'
    try:
        print(f"Loading dataset: {file_name}...")
        df = pd.read_csv(file_name)
        final_model, used_features = walk_forward_cross_validation(df)
        print("\nPipeline execution complete. Ready for visualization.")
    except FileNotFoundError:
        print(f"\nError: '{file_name}' not found.")
        print("Please ensure you have run the data engineering script first.")