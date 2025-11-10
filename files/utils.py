"""
Utility Functions for MAPPO Supply Chain Management
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linprog
from config import Config

def calculate_counterfactual_profit(env, actual_demand: np.ndarray, config: Config) -> float:
    """
    Calculate the theoretical optimal profit with perfect demand knowledge
    This is used to compute the decision error for the predictor agent
    
    Args:
        env: Supply chain environment
        actual_demand: Actual demand that occurred
        config: Configuration object
    
    Returns:
        Theoretical optimal profit
    """
    
    # Simplified calculation - in practice, this would solve an optimization problem
    # Here we use a heuristic approach
    
    optimal_profit = 0.0
    
    # For each retailer, calculate optimal order quantity given perfect demand knowledge
    for i, retailer in enumerate(env.retailers):
        demand_i = actual_demand[i]
        current_inventory = retailer.inventory
        
        # Optimal order is to meet exactly the demand minus current inventory
        optimal_order = np.maximum(demand_i - current_inventory, 0)
        
        # Calculate profit with optimal order
        sales = np.minimum(current_inventory + optimal_order, demand_i)
        revenue = np.sum(sales * config.retail_price)
        ordering_cost = np.sum(optimal_order * config.ordering_cost_retailer)
        holding_cost = np.sum((current_inventory + optimal_order - sales) * config.holding_cost_retailer)
        
        optimal_profit += revenue - ordering_cost - holding_cost
    
    # For warehouse, calculate optimal order given aggregated retailer demand
    total_retailer_demand = np.sum(actual_demand, axis=0)
    warehouse_inventory = env.warehouse.inventory
    
    # Warehouse optimal order (considering lead time)
    if env.warehouse.is_order_day():
        # Estimate demand for lead time period
        future_demand_estimate = total_retailer_demand * config.lead_time
        optimal_warehouse_order = np.maximum(future_demand_estimate - warehouse_inventory, 0)
        
        # Calculate warehouse profit component
        fulfilled = np.minimum(warehouse_inventory, total_retailer_demand)
        revenue = np.sum(fulfilled * config.wholesale_price)
        ordering_cost = np.sum(optimal_warehouse_order * config.ordering_cost_warehouse)
        holding_cost = np.sum(warehouse_inventory * config.holding_cost_warehouse)
        supplier_cost = np.sum(optimal_warehouse_order * config.supplier_price)
        
        optimal_profit += revenue - ordering_cost - holding_cost - supplier_cost
    
    return optimal_profit

def solve_newsvendor_problem(demand_mean: float, demand_std: float, 
                            holding_cost: float, stockout_cost: float,
                            ordering_cost: float, selling_price: float) -> float:
    """
    Solve the classical newsvendor problem for optimal order quantity
    
    Args:
        demand_mean: Mean demand
        demand_std: Standard deviation of demand
        holding_cost: Cost per unit of holding inventory
        stockout_cost: Cost per unit of stockout
        ordering_cost: Cost per unit ordered
        selling_price: Revenue per unit sold
    
    Returns:
        Optimal order quantity
    """
    from scipy.stats import norm
    
    # Critical ratio
    cu = selling_price - ordering_cost  # Underage cost
    co = ordering_cost - holding_cost   # Overage cost
    
    if cu + co <= 0:
        return 0
    
    critical_ratio = cu / (cu + co)
    
    # Find order quantity using inverse normal distribution
    z_score = norm.ppf(critical_ratio)
    optimal_quantity = demand_mean + z_score * demand_std
    
    return max(0, optimal_quantity)

def compute_supply_chain_metrics(env, episode_data: Dict) -> Dict:
    """
    Compute comprehensive supply chain performance metrics
    
    Args:
        env: Supply chain environment
        episode_data: Data collected during an episode
    
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Service level metrics
    if 'demands' in episode_data and 'sales' in episode_data:
        total_demand = np.sum(episode_data['demands'])
        total_sales = np.sum(episode_data['sales'])
        metrics['fill_rate'] = total_sales / total_demand if total_demand > 0 else 0
        
        # Stockout frequency
        stockout_periods = np.sum(episode_data['sales'] < episode_data['demands'])
        total_periods = len(episode_data['demands'])
        metrics['stockout_frequency'] = stockout_periods / total_periods if total_periods > 0 else 0
    
    # Inventory metrics
    if 'inventories' in episode_data:
        inventories = np.array(episode_data['inventories'])
        metrics['avg_inventory'] = np.mean(inventories)
        metrics['inventory_turnover'] = total_sales / metrics['avg_inventory'] if metrics['avg_inventory'] > 0 else 0
    
    # Cost metrics
    if 'costs' in episode_data:
        metrics['total_cost'] = np.sum(episode_data['costs'])
        metrics['avg_cost_per_period'] = metrics['total_cost'] / total_periods if total_periods > 0 else 0
    
    # Profit metrics
    if 'profits' in episode_data:
        metrics['total_profit'] = np.sum(episode_data['profits'])
        metrics['profit_margin'] = metrics['total_profit'] / (total_sales * 10) if total_sales > 0 else 0  # Assuming price of 10
    
    # Bullwhip effect (demand amplification)
    if 'retailer_orders' in episode_data and 'warehouse_orders' in episode_data:
        retailer_orders = np.array(episode_data['retailer_orders'])
        warehouse_orders = np.array(episode_data['warehouse_orders'])
        
        if len(retailer_orders) > 1 and len(warehouse_orders) > 1:
            retailer_cv = np.std(retailer_orders) / np.mean(retailer_orders) if np.mean(retailer_orders) > 0 else 0
            warehouse_cv = np.std(warehouse_orders) / np.mean(warehouse_orders) if np.mean(warehouse_orders) > 0 else 0
            metrics['bullwhip_effect'] = warehouse_cv / retailer_cv if retailer_cv > 0 else 1
    
    return metrics

def create_demand_forecast(historical_demand: np.ndarray, 
                          horizon: int,
                          method: str = 'moving_average') -> np.ndarray:
    """
    Create demand forecast using various methods
    
    Args:
        historical_demand: Historical demand data (time_steps, products)
        horizon: Forecast horizon
        method: Forecasting method ('moving_average', 'exponential_smoothing', 'linear_trend')
    
    Returns:
        Forecasted demand for the horizon
    """
    if len(historical_demand) == 0:
        return np.zeros((horizon, historical_demand.shape[1] if len(historical_demand.shape) > 1 else 1))
    
    if method == 'moving_average':
        # Simple moving average
        window_size = min(7, len(historical_demand))
        if len(historical_demand.shape) == 1:
            historical_demand = historical_demand.reshape(-1, 1)
        
        forecast = np.mean(historical_demand[-window_size:], axis=0)
        return np.tile(forecast, (horizon, 1))
    
    elif method == 'exponential_smoothing':
        # Exponential smoothing
        alpha = 0.3
        if len(historical_demand.shape) == 1:
            historical_demand = historical_demand.reshape(-1, 1)
        
        forecast = historical_demand[-1].copy()
        for i in range(len(historical_demand) - 2, -1, -1):
            forecast = alpha * historical_demand[i] + (1 - alpha) * forecast
        
        return np.tile(forecast, (horizon, 1))
    
    elif method == 'linear_trend':
        # Linear trend extrapolation
        if len(historical_demand.shape) == 1:
            historical_demand = historical_demand.reshape(-1, 1)
        
        time_steps = len(historical_demand)
        x = np.arange(time_steps)
        
        forecasts = []
        for product in range(historical_demand.shape[1]):
            y = historical_demand[:, product]
            
            # Fit linear trend
            coefficients = np.polyfit(x, y, 1)
            
            # Extrapolate
            future_x = np.arange(time_steps, time_steps + horizon)
            future_y = np.polyval(coefficients, future_x)
            forecasts.append(future_y)
        
        return np.array(forecasts).T
    
    else:
        # Default to moving average
        return create_demand_forecast(historical_demand, horizon, 'moving_average')

def calculate_safety_stock(demand_mean: float, demand_std: float,
                          lead_time: int, service_level: float = 0.95) -> float:
    """
    Calculate safety stock based on demand variability and desired service level
    
    Args:
        demand_mean: Mean demand per period
        demand_std: Standard deviation of demand per period
        lead_time: Lead time in periods
        service_level: Desired service level (probability of not stocking out)
    
    Returns:
        Safety stock quantity
    """
    from scipy.stats import norm
    
    # Z-score for desired service level
    z_score = norm.ppf(service_level)
    
    # Safety stock formula
    safety_stock = z_score * demand_std * np.sqrt(lead_time)
    
    return safety_stock

def compute_reorder_point(demand_mean: float, lead_time: int, 
                         safety_stock: float) -> float:
    """
    Calculate reorder point for inventory management
    
    Args:
        demand_mean: Mean demand per period
        lead_time: Lead time in periods
        safety_stock: Safety stock quantity
    
    Returns:
        Reorder point
    """
    return demand_mean * lead_time + safety_stock

class ExperienceReplay:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, transition: Dict):
        """Add a transition to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of transitions"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

def normalize_observations(observations: np.ndarray, 
                          mean: Optional[np.ndarray] = None,
                          std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize observations using running statistics
    
    Args:
        observations: Observations to normalize
        mean: Running mean (will be computed if None)
        std: Running standard deviation (will be computed if None)
    
    Returns:
        Normalized observations, updated mean, updated std
    """
    if mean is None:
        mean = np.mean(observations, axis=0)
    if std is None:
        std = np.std(observations, axis=0)
        std[std < 1e-6] = 1.0  # Prevent division by zero
    
    normalized = (observations - mean) / std
    
    return normalized, mean, std

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, 
                next_values: torch.Tensor, dones: torch.Tensor,
                gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Rewards tensor
        values: Value estimates
        next_values: Next value estimates
        dones: Done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages, returns
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns

def save_checkpoint(state: Dict, filepath: str):
    """Save training checkpoint"""
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, device: torch.device) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
