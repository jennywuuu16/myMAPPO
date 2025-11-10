"""
Example script for preparing and using CSV data with the MAPPO supply chain system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_csv(file_path: str = './data/sales_data.csv', 
                     num_stores: int = 10,
                     num_items: int = 5,
                     num_days: int = 365):
    """
    Create a sample CSV file with sales data in the required format
    
    Columns: store_id, time, item_id, dep_id, cat_id, state_id, quantity, price
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    print(f"Creating sample sales data CSV...")
    
    # Generate data
    data = []
    base_date = datetime(2023, 1, 1)
    
    # Define stores across different states
    stores = [f'STORE_{i:03d}' for i in range(1, num_stores + 1)]
    items = [f'ITEM_{i:03d}' for i in range(1, num_items + 1)]
    
    # Map items to departments and categories
    item_dept_map = {f'ITEM_{i:03d}': f'DEPT_{(i-1)//2 + 1:02d}' for i in range(1, num_items + 1)}
    item_cat_map = {f'ITEM_{i:03d}': f'CAT_{(i-1)//3 + 1:02d}' for i in range(1, num_items + 1)}
    
    # Map stores to states
    store_state_map = {f'STORE_{i:03d}': f'STATE_{(i-1)//3 + 1:02d}' for i in range(1, num_stores + 1)}
    
    for day in range(num_days):
        current_date = base_date + timedelta(days=day)
        
        for store in stores:
            for item in items:
                # Generate realistic demand patterns
                base_demand = 50
                
                # Weekly seasonality (higher on weekends)
                day_of_week = current_date.weekday()
                weekend_factor = 1.3 if day_of_week >= 5 else 1.0
                
                # Monthly seasonality
                day_of_month = current_date.day
                monthly_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_month / 30)
                
                # Yearly seasonality (holiday effects)
                day_of_year = current_date.timetuple().tm_yday
                yearly_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
                
                # Store-specific factors
                store_idx = int(store.split('_')[1])
                store_size_factor = 0.5 + 1.5 * (store_idx / num_stores)
                
                # Item-specific factors
                item_idx = int(item.split('_')[1])
                item_popularity = 0.7 + 0.6 * np.random.random()
                
                # Random noise
                noise_factor = np.random.lognormal(0, 0.2)
                
                # Calculate final quantity
                quantity = int(
                    base_demand * 
                    weekend_factor * 
                    monthly_factor * 
                    yearly_factor * 
                    store_size_factor * 
                    item_popularity * 
                    noise_factor
                )
                quantity = max(0, quantity)
                
                # Price variations (small random changes)
                base_price = 10.0 + item_idx * 0.5
                price = base_price * (1 + np.random.normal(0, 0.05))
                price = round(max(5.0, price), 2)
                
                # Add to data
                data.append({
                    'store_id': store,
                    'time': current_date.strftime('%Y-%m-%d'),
                    'item_id': item,
                    'dep_id': item_dept_map[item],
                    'cat_id': item_cat_map[item],
                    'state_id': store_state_map[store],
                    'quantity': quantity,
                    'price': price
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    print(f"Sample data created successfully!")
    print(f"  - File: {file_path}")
    print(f"  - Records: {len(df):,}")
    print(f"  - Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  - Stores: {num_stores}")
    print(f"  - Items: {num_items}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  - Mean quantity: {df['quantity'].mean():.2f}")
    print(f"  - Std quantity: {df['quantity'].std():.2f}")
    print(f"  - Mean price: ${df['price'].mean():.2f}")
    
    return df

def validate_csv_format(file_path: str) -> bool:
    """
    Validate that the CSV file has the correct format
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['store_id', 'time', 'item_id', 'dep_id', 
                          'cat_id', 'state_id', 'quantity', 'price']
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            return False
        
        # Check data types
        try:
            pd.to_datetime(df['time'])
        except:
            print("Error: 'time' column cannot be converted to datetime")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['quantity']):
            print("Error: 'quantity' column must be numeric")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['price']):
            print("Error: 'price' column must be numeric")
            return False
        
        print(f"✓ CSV format is valid!")
        print(f"  - {len(df):,} records")
        print(f"  - {df['store_id'].nunique()} unique stores")
        print(f"  - {df['item_id'].nunique()} unique items")
        print(f"  - Date range: {df['time'].min()} to {df['time'].max()}")
        
        return True
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

def prepare_data_for_training(input_csv: str, output_dir: str = './data'):
    """
    Prepare and preprocess data for training
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_csv)
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values(['time', 'store_id', 'item_id'])
    
    # Split into train/validation/test
    unique_dates = df['time'].unique()
    n_dates = len(unique_dates)
    
    train_end_idx = int(n_dates * 0.7)
    val_end_idx = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    # Create splits
    train_df = df[df['time'].isin(train_dates)]
    val_df = df[df['time'].isin(val_dates)]
    test_df = df[df['time'].isin(test_dates)]
    
    # Save splits
    train_df.to_csv(f'{output_dir}/train_data.csv', index=False)
    val_df.to_csv(f'{output_dir}/val_data.csv', index=False)
    test_df.to_csv(f'{output_dir}/test_data.csv', index=False)
    
    print(f"Data splits created:")
    print(f"  - Train: {len(train_df):,} records ({train_dates[0]} to {train_dates[-1]})")
    print(f"  - Validation: {len(val_df):,} records ({val_dates[0]} to {val_dates[-1]})")
    print(f"  - Test: {len(test_df):,} records ({test_dates[0]} to {test_dates[-1]})")
    
    # Calculate and save statistics
    stats = {
        'mean_quantity_by_item': df.groupby('item_id')['quantity'].mean().to_dict(),
        'std_quantity_by_item': df.groupby('item_id')['quantity'].std().to_dict(),
        'mean_quantity_by_store': df.groupby('store_id')['quantity'].mean().to_dict(),
        'mean_price_by_item': df.groupby('item_id')['price'].mean().to_dict(),
        'total_records': len(df),
        'date_range': {
            'start': str(df['time'].min()),
            'end': str(df['time'].max())
        }
    }
    
    import json
    with open(f'{output_dir}/data_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nStatistics saved to {output_dir}/data_statistics.json")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Example 1: Create sample CSV data
    sample_file = './data/sales_data.csv'
    
    # Create sample data if it doesn't exist
    if not os.path.exists(sample_file):
        print("Creating sample sales data...")
        create_sample_csv(
            file_path=sample_file,
            num_stores=10,
            num_items=5,
            num_days=365
        )
    else:
        print(f"Using existing file: {sample_file}")
    
    # Example 2: Validate CSV format
    print("\n" + "="*50)
    print("Validating CSV format...")
    if validate_csv_format(sample_file):
        print("CSV is ready for training!")
    
    # Example 3: Prepare data splits
    print("\n" + "="*50)
    print("Preparing data splits for training...")
    prepare_data_for_training(sample_file)
    
    # Example 4: Show how to use with custom data
    print("\n" + "="*50)
    print("To use your own data:")
    print("1. Ensure your CSV has these columns:")
    print("   store_id, time, item_id, dep_id, cat_id, state_id, quantity, price")
    print("2. Update config.py:")
    print("   config.demand_data_path = 'path/to/your/data.csv'")
    print("3. Run training:")
    print("   python train.py")
