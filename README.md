# Multi-Agent Proximal Policy Optimization (MAPPO) for Supply Chain Management

## Model Framework

### Overview
This project implements a Multi-Agent Reinforcement Learning system using MAPPO to optimize inventory management in a multi-echelon supply chain. The system consists of three types of agents:
1. **Retailer Agents** - Make ordering decisions to meet customer demand
2. **Warehouse Agent** - Manages inventory and fulfills retailer orders
3. **Predictor Agent** - Forecasts future demand to guide decisions

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Supply Chain Environment                     │
│  ┌─────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │  Retailers  │ ←──→ │  Warehouse   │ ←──→ │  Suppliers   │   │
│  │  (Agents)   │      │   (Agent)    │      │              │   │
│  └─────────────┘      └──────────────┘      └──────────────┘   │
│         ↑                     ↑                                  │
│         │                     │                                  │
│         └─────────────────────┴──────────────────────────────┐  │
│                                                                │  │
│                    ┌──────────────────────┐                   │  │
│                    │  Predictor Agent     │                   │  │
│                    │  (Demand Forecast)   │                   │  │
│                    └──────────────────────┘                   │  │
│                              ↓                                 ↓  │
│                    ┌─────────────────────────────────────────────┤
│                    │         Historical Data & Context          │
│                    └─────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘

                              ↓ States ↓

┌─────────────────────────────────────────────────────────────────┐
│                         MAPPO Algorithm                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Shared Critic (Global Value Function)                   │   │
│  │  - Input: Global state (all agents' observations)        │   │
│  │  - Output: Value estimate V(s_global)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐    │
│  │ Retailer   │  │ Warehouse  │  │    Predictor Actor     │    │
│  │   Actor    │  │   Actor    │  │  (Demand Forecasting)  │    │
│  │  (Policy)  │  │  (Policy)  │  │  - LSTM + Attention    │    │
│  └────────────┘  └────────────┘  └────────────────────────┘    │
│        ↓               ↓                     ↓                    │
└────────┼───────────────┼─────────────────────┼───────────────────┘
         ↓               ↓                     ↓

      Actions         Actions            Demand Predictions
```

### Agent Details

#### 1. Retailer Agent
**State Space:**
- Inventory levels (normalized by capacity)
- Demand history encoding (mean, std)
- Day of week encoding (cyclical)
- Predicted next-day demand from Predictor

**Action Space:**
- Order quantity for each product (continuous)

**Reward Function:**
```python
profit = revenue - ordering_cost - holding_cost - stockout_cost

where:
  revenue = sales × retail_prices
  ordering_cost = orders × wholesale_prices  # Payment to warehouse
  holding_cost = inventory × h_i
  stockout_cost = stockout × b_i
```

**Network Architecture:**
- Input: State features
- Hidden layers: 3 × 256 units (ReLU)
- Output: Action distribution (Gaussian)

#### 2. Warehouse Agent
**State Space:**
- Warehouse inventory (normalized)
- Aggregated demand prediction (4-day forecast from Predictor)
- Ordering cost and lead time
- Downstream inventory summary
- Order day indicator

**Action Space:**
- Order quantity to suppliers (continuous, only on order days)

**Reward Function:**
```python
profit = revenue - ordering_cost - holding_cost - stockout_cost

where:
  revenue = deliveries × wholesale_prices
  ordering_cost = orders × supplier_prices  # Payment to suppliers
  holding_cost = inventory × h_w
  stockout_cost = unmet_demand × b_w
```

**Network Architecture:**
- Input: State features
- Hidden layers: 3 × 256 units (ReLU)
- Output: Action distribution (Gaussian)

#### 3. Predictor Agent
**State Space:**
- Historical sales data (30-day window)
- Context features (day of week, month, seasonality)
- Time progress in episode

**Action Space:**
- Demand forecast for next H days (H=4)
- Output shape: (num_retailers, H, num_products)

**Reward Function (Decision Error):**
```python
# Calculate profit gap between actual and optimal
predicted_profit = warehouse.profit + sum(retailer.profit)
optimal_profit = calculate_counterfactual_profit(env, actual_demand)

profit_gap = |predicted_profit - optimal_profit['total']| / max(|optimal_profit['total']|, 1e-5)
decision_error = -profit_gap  # Negative to minimize gap

Returns:
{
  'error': float,              # Decision error (reward signal)
  'predicted_profit': float,   # Actual profit achieved
  'optimal_profit': {          # Theoretical optimal breakdown
    'retailer': float,
    'warehouse': float,
    'total': float
  },
  'profit_gap': float         # Normalized gap
}
```

**Network Architecture:**
- LSTM encoder for historical data (128 hidden units)
- Attention mechanism over time steps
- Context fusion layer
- Decoder: 3 × 256 units (ReLU)
- Output: Multi-horizon demand forecast

### Cost Structure

| Component | Retailer | Warehouse | Description |
|-----------|----------|-----------|-------------|
| **Ordering Cost** | `order × wholesale_prices` | `order × supplier_prices` | Payment to upstream entity |
| **Holding Cost** | `inventory × 0.1` | `inventory × 0.05` | Cost per unit held |
| **Stockout Cost** | `stockout × 2.0` | `stockout × 1.5` | Penalty for unmet demand |

### Price Structure

Given retail price `p_retail`:
- **Retail price**: `p_retail` (e.g., ~$14-15)
- **Wholesale price**: `p_retail × 0.6` (e.g., ~$9)
- **Supplier price**: `p_retail × 0.3` (e.g., ~$4.5)

**Expected Margins:**
- Retailer: `(p_retail - wholesale) ≈ 40%`
- Warehouse: `(wholesale - supplier) ≈ 50%`

### MAPPO Training Algorithm

```
For each episode:
  1. Reset environment and get initial observations

  For each time step:
    2. Prepare states for all agents
       - Warehouse state: inventory, demand forecast, order cycle
       - Retailer states: inventory, demand history, predictions
       - Predictor state: historical sales, context features

    3. Select actions using current policies
       - Retailers: Order quantities
       - Warehouse: Replenishment order
       - Predictor: Demand forecast

    4. Execute actions in environment
       - Update inventories
       - Calculate sales and stockouts
       - Compute profits

    5. Calculate rewards
       - Retailer/Warehouse: Immediate profit
       - Predictor: Decision error (optimal gap)

    6. Compute global value using shared critic

    7. Store transition in replay buffer

  8. Update all policies and critic using PPO
     - Compute GAE advantages
     - Update actors (policy gradient)
     - Update critic (value loss)
     - Use entropy regularization
```

### Key Hyperparameters

```python
# Environment
num_retailers = 1
num_products = 2
episode_length = 100
warehouse_order_cycle = 4  # Days between warehouse orders
lead_time = 3              # Days for warehouse orders to arrive

# MAPPO
learning_rate = 3e-4
gamma = 0.99               # Discount factor
gae_lambda = 0.95          # GAE parameter
eps_clip = 0.2             # PPO clip range
value_loss_coef = 0.5
entropy_coef = 0.01
batch_size = 64
ppo_epochs = 4
update_frequency = 10      # Episodes between updates

# Networks
hidden_size = 256
lstm_hidden_size = 128
num_layers = 3

# Prediction
prediction_horizon = 4     # Days to forecast
history_length = 30        # Historical window
```

### Data Requirements

The system uses real sales data with the following schema:

```
Required columns:
- store_id: Store identifier
- time: Date timestamp
- item_id: Product identifier
- dept_id: Department ID
- cat_id: Category ID
- state_id: State ID
- quantity: Sales quantity (demand)
- price: Selling price
```

Expected data statistics (for stable training):
- Daily sales: ~1-8 units per product
- Prices: ~$14-15 per unit
- Multiple days of history (recommended: 100+ days)

### File Structure

```
myMAPPO/
├── config.py           # Configuration parameters
├── environment.py      # Supply chain environment
├── networks.py         # Neural network architectures
├── mappo.py           # MAPPO algorithm implementation
├── utils.py           # Utility functions (optimal profit, metrics)
├── data_loader.py     # Sales data loading and preprocessing
├── train.py           # Main training script
└── README.md          # This file
```

### Usage

```bash
# Train the model
python train.py

# The script will:
# 1. Load sales data from config.demand_data_path
# 2. Initialize environment and agents
# 3. Train for config.num_episodes episodes
# 4. Save models to config.model_save_path
# 5. Generate training curves and logs
```

### Evaluation Metrics

The system tracks:
- **Average Reward**: Sum of all agent rewards
- **System Profit**: Total profit (retailers + warehouse)
- **Predictor Error**: Decision error magnitude
- **Service Level**: Fill rate, stockout frequency
- **Inventory Metrics**: Average inventory, turnover
- **Bullwhip Effect**: Demand amplification ratio

### Recent Fixes (Nov 2024)

**Profit Calculation Correction:**
- Fixed retailer and warehouse to use wholesale/supplier prices as ordering costs
- Previously used fixed transaction fees (0.3, 0.1) instead of actual prices
- This caused items to appear "cheap", leading to over-ordering and negative profits
- Now profits are realistic and positive with proper margins

**Optimal Profit Calculation:**
- Changed return type from `float` to `Dict` for better breakdown
- Returns: `{'retailer': float, 'warehouse': float, 'total': float}`
- Updated decision error calculation to use `optimal_profit['total']`
- Added safe division to prevent divide-by-zero errors

### References

- MAPPO Paper: [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- Supply Chain RL: Multi-echelon inventory optimization
- Demand Forecasting: LSTM with attention mechanism
