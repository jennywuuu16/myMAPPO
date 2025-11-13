"""
Data Loader for Supply Chain Management
Handles loading and preprocessing of historical sales data from CSV files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SalesDataLoader:
    """Load and preprocess sales data from CSV files"""
    
    def __init__(self, file_path: str, config):
        """
        Initialize data loader
        
        Args:
            file_path: Path to CSV file with sales data
            config: Configuration object
        """
        self.file_path = file_path
        self.config = config
        self.data = None
        self.processed_data = None
        self.store_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.dept_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        
        # Load and process data
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.data)} records from {self.file_path}")
            
            # Ensure required columns exist
            required_columns = ['store_id', 'time', 'item_id', 'dept_id', 
                              'cat_id', 'state_id', 'quantity', 'price']
            missing_columns = set(required_columns) - set(self.data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Convert time column to datetime
            self.data['time'] = pd.to_datetime(self.data['time'])
            
            # Sort by time and store
            self.data = self.data.sort_values(['store_id', 'time', 'item_id'])
            
        except FileNotFoundError:
            print(f"Warning: Data file {self.file_path} not found. Creating synthetic data.")
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        # Encode categorical variables
        self.data['store_idx'] = self.store_encoder.fit_transform(self.data['store_id'])
        self.data['item_idx'] = self.item_encoder.fit_transform(self.data['item_id'])
        self.data['dept_idx'] = self.dept_encoder.fit_transform(self.data['dept_id'])
        self.data['cat_idx'] = self.category_encoder.fit_transform(self.data['cat_id'])
        self.data['state_idx'] = self.state_encoder.fit_transform(self.data['state_id'])
        
        # Extract time features
        self.data['day_of_week'] = self.data['time'].dt.dayofweek
        self.data['day_of_month'] = self.data['time'].dt.day
        self.data['month'] = self.data['time'].dt.month
        self.data['week_of_year'] = self.data['time'].dt.isocalendar().week
        
        # Create pivot table for easy access: (date, store, item) -> quantity
        self.demand_matrix = self.data.pivot_table(
            index=['time', 'store_idx'],
            columns='item_idx',
            values='quantity',
            fill_value=0,
            aggfunc='sum'
        )
        
        # Create price matrix: (date, store, item) -> price
        self.price_matrix = self.data.pivot_table(
            index=['time', 'store_idx'],
            columns='item_idx',
            values='price',
            aggfunc='mean'
        )
        
        # Calculate average prices per product (for use when specific prices aren't available)
        self.avg_prices_by_product = self.data.groupby('item_idx')['price'].mean().to_dict()
        
        # Ensure we have prices for all products
        for i in range(max(self.avg_prices_by_product.keys()) + 1):
            if i not in self.avg_prices_by_product:
                self.avg_prices_by_product[i] = 10.0  # Default price
        
        # Get unique stores and items
        self.num_stores = len(self.data['store_id'].unique())
        self.num_items = len(self.data['item_id'].unique())
        self.num_days = len(self.data['time'].unique())
        
        print(f"Preprocessed data: {self.num_stores} stores, {self.num_items} items, {self.num_days} days")
        print(f"Average prices by product: {self.avg_prices_by_product}")
        
        # Update config if necessary
        if hasattr(self.config, 'num_retailers') and self.config.num_retailers != self.num_stores:
            print(f"Warning: Config has {self.config.num_retailers} retailers but data has {self.num_stores} stores")
            self.config.num_retailers = min(self.config.num_retailers, self.num_stores)
        
        if hasattr(self.config, 'num_products') and self.config.num_products != self.num_items:
            print(f"Warning: Config has {self.config.num_products} products but data has {self.num_items} items")
            self.config.num_products = min(self.config.num_products, self.num_items)
    
    def get_demand_sequence(self, start_date: Optional[datetime] = None, 
                           length: int = 100) -> Dict[str, np.ndarray]:
        """
        Get demand sequence for simulation
        
        Args:
            start_date: Starting date for sequence (random if None)
            length: Length of sequence in days
            
        Returns:
            Dictionary with demand data, price data for each store and metadata
        """
        available_dates = self.demand_matrix.index.get_level_values('time').unique()
        
        if start_date is None:
            # Random start date ensuring we have enough data
            max_start_idx = len(available_dates) - length
            if max_start_idx <= 0:
                start_idx = 0
                actual_length = len(available_dates)
            else:
                start_idx = np.random.randint(0, max_start_idx)
                actual_length = length
        else:
            start_idx = np.where(available_dates == start_date)[0][0]
            actual_length = min(length, len(available_dates) - start_idx)
        
        end_idx = start_idx + actual_length
        selected_dates = available_dates[start_idx:end_idx]
        
        # Extract demand and price for selected period
        demand_data = {}
        price_data = {}
        
        for store_idx in range(self.config.num_retailers):
            store_demand = []
            store_prices = []
            for date in selected_dates:
                if (date, store_idx) in self.demand_matrix.index:
                    day_demand = self.demand_matrix.loc[(date, store_idx)].values[:self.config.num_products]
                    day_prices = self.price_matrix.loc[(date, store_idx)].values[:self.config.num_products]
                    
                    # Fill NaN prices with average prices
                    for p in range(self.config.num_products):
                        if np.isnan(day_prices[p]):
                            day_prices[p] = self.avg_prices_by_product.get(p, 10.0)
                else:
                    # If no data for this store/date, use zeros for demand and avg prices
                    day_demand = np.zeros(self.config.num_products)
                    day_prices = np.array([self.avg_prices_by_product.get(p, 10.0) 
                                          for p in range(self.config.num_products)])
                
                store_demand.append(day_demand)
                store_prices.append(day_prices)
            
            demand_data[f'store_{store_idx}'] = np.array(store_demand)
            price_data[f'store_{store_idx}'] = np.array(store_prices)
        
        # Calculate average prices per product across all stores for this period
        all_prices = []
        for store_key in price_data:
            all_prices.append(price_data[store_key])
        
        if all_prices:
            avg_prices_period = np.mean(all_prices, axis=0)  # Average across stores
            retail_prices = np.mean(avg_prices_period, axis=0)  # Average across time
        else:
            retail_prices = np.array([self.avg_prices_by_product.get(p, 10.0) 
                                     for p in range(self.config.num_products)])
        
        # Get contextual features
        context_data = {
            'dates': selected_dates,
            'day_of_week': [d.dayofweek for d in selected_dates],
            'month': [d.month for d in selected_dates],
            'retail_prices': retail_prices,  # Average retail prices per product
            'wholesale_prices': retail_prices * self.config.wholesale_price_ratio,
            'supplier_prices': retail_prices * self.config.supplier_price_ratio
        }
        
        return {
            'demand': demand_data,
            'prices': price_data,
            'context': context_data,
            'start_date': selected_dates[0],
            'end_date': selected_dates[-1],
            'length': actual_length
        }
    
    def get_training_validation_split(self, split_ratio: float = 0.8) -> Tuple[Dict, Dict]:
        """
        Split data into training and validation sets
        
        Args:
            split_ratio: Ratio of training data
            
        Returns:
            Training and validation data dictionaries
        """
        all_dates = self.demand_matrix.index.get_level_values('time').unique()
        split_idx = int(len(all_dates) * split_ratio)
        
        train_dates = all_dates[:split_idx]
        val_dates = all_dates[split_idx:]
        
        # Create training data
        train_data = self.get_demand_sequence(
            start_date=train_dates[0],
            length=len(train_dates)
        )
        
        # Create validation data
        val_data = self.get_demand_sequence(
            start_date=val_dates[0],
            length=len(val_dates)
        )
        
        return train_data, val_data
    
    def get_demand_statistics(self) -> Dict:
        """Calculate and return demand statistics"""
        stats = {
            'mean_demand': self.data.groupby('item_idx')['quantity'].mean().to_dict(),
            'std_demand': self.data.groupby('item_idx')['quantity'].std().to_dict(),
            'max_demand': self.data.groupby('item_idx')['quantity'].max().to_dict(),
            'min_demand': self.data.groupby('item_idx')['quantity'].min().to_dict(),
            'mean_price': self.data.groupby('item_idx')['price'].mean().to_dict(),
            'total_sales': self.data.groupby('store_idx')['quantity'].sum().to_dict(),
        }
        
        # Add store-level statistics
        store_stats = self.data.groupby('store_idx')['quantity'].agg(['mean', 'std', 'sum'])
        stats['store_mean_demand'] = store_stats['mean'].to_dict()
        stats['store_std_demand'] = store_stats['std'].to_dict()
        stats['store_total_demand'] = store_stats['sum'].to_dict()
        
        return stats
    
    def get_item_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for items based on department and category"""
        item_features = self.data[['item_idx', 'dept_idx', 'cat_idx']].drop_duplicates()
        
        embeddings = {}
        for _, row in item_features.iterrows():
            item_idx = row['item_idx']
            # Simple one-hot encoding for departments and categories
            dept_vec = np.zeros(len(self.dept_encoder.classes_))
            dept_vec[row['dept_idx']] = 1
            
            cat_vec = np.zeros(len(self.category_encoder.classes_))
            cat_vec[row['cat_idx']] = 1
            
            embeddings[item_idx] = np.concatenate([dept_vec, cat_vec])
        
        return embeddings
    
    def get_store_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for stores based on state and other features"""
        store_features = self.data[['store_idx', 'state_idx']].drop_duplicates()
        
        embeddings = {}
        for _, row in store_features.iterrows():
            store_idx = row['store_idx']
            # One-hot encoding for state
            state_vec = np.zeros(len(self.state_encoder.classes_))
            state_vec[row['state_idx']] = 1
            
            # Add store size factor (based on historical sales)
            store_sales = self.data[self.data['store_idx'] == store_idx]['quantity'].sum()
            size_factor = store_sales / self.data.groupby('store_idx')['quantity'].sum().mean()
            
            embeddings[store_idx] = np.append(state_vec, size_factor)
        
        return embeddings