# MAPPO Supply Chain Management with External Data

## Overview
This implementation of MAPPO (Multi-Agent Proximal Policy Optimization) for supply chain management now supports loading historical sales data from CSV files instead of generating synthetic demand.

## Data Format Requirements

Your CSV file must contain the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| store_id | string | Unique store identifier | STORE_001 |
| time | date/datetime | Date of the sales record | 2023-01-15 |
| item_id | string | Unique product identifier | ITEM_001 |
| dep_id | string | Department identifier | DEPT_01 |
| cat_id | string | Category identifier | CAT_01 |
| state_id | string | State/region identifier | STATE_01 |
| quantity | numeric | Sales quantity | 50 |
| price | numeric | Unit price | 9.99 |

## Quick Start

### 1. Prepare Your Data

If you have your own sales data:
```bash
# Place your CSV file in the data directory
cp your_sales_data.csv ./data/sales_data.csv
```

Or create sample data:
```bash
python prepare_data.py
```

### 2. Configure the System

Edit `config.py` to point to your data file:
```python
# In config.py
demand_data_path = './data/sales_data.csv'
```

### 3. Train the Model

```bash
python train.py
```

## Data Preparation Script

The `prepare_data.py` script provides utilities for:

1. **Creating sample data**: Generates realistic sales data with patterns
2. **Validating CSV format**: Checks if your data meets requirements  
3. **Creating train/val/test splits**: Automatically splits data by time
4. **Computing statistics**: Calculates demand statistics for normalization

### Usage Examples

```python
from prepare_data import create_sample_csv, validate_csv_format

# Create sample data
create_sample_csv(
    file_path='./data/sales_data.csv',
    num_stores=10,
    num_items=5,
    num_days=365
)

# Validate your data
is_valid = validate_csv_format('./data/your_data.csv')

# Prepare splits
from prepare_data import prepare_data_for_training
train_df, val_df, test_df = prepare_data_for_training('./data/your_data.csv')
```

## Data Loader Features

The `SalesDataLoader` class provides:

- **Automatic encoding** of categorical variables (store, item, department, etc.)
- **Time feature extraction** (day of week, month, seasonality)
- **Demand statistics** calculation for normalization
- **Store and item embeddings** based on characteristics
- **Flexible sequence extraction** for training episodes
- **Automatic train/validation splitting**

## How Data is Used in Training

1. **Episode Start**: When an episode begins, the environment loads a random sequence of historical data
2. **Daily Demand**: Each step uses actual historical demand from the loaded sequence
3. **Context Features**: Time features and patterns are extracted from the data
4. **Predictor Training**: The predictor agent learns from actual historical patterns
5. **Counterfactual Analysis**: Optimal decisions are computed based on actual demand

## Key Differences from Synthetic Data

| Aspect | Synthetic | External CSV |
|--------|-----------|--------------|
| Demand Patterns | Generated with fixed seasonality | Learned from real patterns |
| Store Heterogeneity | Random factors | Based on actual store performance |
| Product Mix | Uniform distribution | Reflects actual product popularity |
| Seasonality | Simple sine waves | Complex real-world patterns |
| Price Variations | Random noise | Actual price changes |

## File Structure

```
.
├── config.py           # Configuration (updated with data path)
├── data_loader.py      # New data loading module
├── environment.py      # Updated to use external data
├── networks.py         # Neural networks (unchanged)
├── mappo.py           # MAPPO algorithm (unchanged)
├── train.py           # Updated training script
├── utils.py           # Utilities (unchanged)
├── prepare_data.py    # Data preparation utilities
└── data/
    ├── sales_data.csv      # Your sales data
    ├── train_data.csv      # Training split
    ├── val_data.csv        # Validation split
    └── test_data.csv       # Test split
```

## Troubleshooting

### No Data File Found
If the data file is not found, the system will automatically generate synthetic data for testing. Check the console output for warnings.

### Data Format Issues
Run validation before training:
```python
python prepare_data.py  # Will validate existing data
```

### Mismatch in Store/Product Count
The system will automatically adjust to use the minimum of:
- Number of stores/products in config
- Number of stores/products in data

## Advanced Usage

### Custom Context Features

You can extend context features in `data_loader.py`:
```python
def get_context_features(self):
    # Add custom features like holidays, promotions, etc.
    features = [
        self.is_holiday(),
        self.promotion_active(),
        self.competitor_price_index(),
        # ... more features
    ]
    return np.array(features)
```

### Different Data Sources

The data loader can be extended to support other formats:
- Multiple CSV files
- Database connections
- API endpoints
- Parquet files

## Performance Considerations

- **Memory**: Large datasets are processed in chunks
- **Speed**: Data is cached during episodes for fast access
- **Scaling**: Supports datasets with millions of records

## Contact & Support

For questions or issues with data integration, please refer to the documentation or create an issue in the repository.
