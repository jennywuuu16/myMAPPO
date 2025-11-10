"""
Supply Chain Environment for MAPPO
Simulates a three-tier supply chain with suppliers, warehouse, and retailers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import Config
from data_loader import SalesDataLoader

@dataclass
class Inventory:
    """Inventory state for entities"""
    current: np.ndarray  # Current inventory levels
    capacity: float  # Maximum capacity
    
class RetailerAgent:
    """Retailer agent in the supply chain"""
    
    def __init__(self, agent_id: int, config: Config):
        self.id = agent_id
        self.config = config
        self.inventory = np.zeros(config.num_products)
        self.capacity = config.retailer_capacity
        self.demand_history = []
        self.order_history = []
        self.profit = 0.0
        
    def get_state(self) -> np.ndarray:
        """Get current state of the retailer"""
        # State includes: inventory, demand history encoding, day of week
        state = np.concatenate([
            self.inventory / self.capacity,  # Normalized inventory
            self._get_demand_encoding(),
            [self.get_day_encoding()]
        ])
        return state
    
    def _get_demand_encoding(self) -> np.ndarray:
        """Encode historical demand using simple statistics"""
        if len(self.demand_history) < 7:
            return np.zeros(self.config.num_products * 3)  # mean, std, trend for each product
        
        history = np.array(self.demand_history[-30:])  # Last 30 days
        encoding = []
        for p in range(self.config.num_products):
            product_history = history[:, p] if history.ndim > 1 else history
            encoding.extend([
                np.mean(product_history),
                np.std(product_history),
                np.polyfit(range(len(product_history)), product_history, 1)[0]  # Trend
            ])
        return np.array(encoding) / 100.0  # Normalize
    
    def get_day_encoding(self) -> float:
        """Encode cyclical time information"""
        return np.sin(2 * np.pi * len(self.demand_history) / 7) 
    
    def step(self, order: np.ndarray, received: np.ndarray, demand: np.ndarray) -> float:
        """Execute one step for the retailer"""
        # Update inventory with received goods
        self.inventory += received
        
        # Calculate actual sales (limited by inventory)
        sales = np.minimum(self.inventory, demand)
        self.inventory -= sales
        
        # Calculate stockout
        stockout = np.maximum(demand - sales, 0)
        
        # Calculate reward (daily profit)
        revenue = np.sum(sales * self.config.retail_price)
        ordering_cost = np.sum(order * self.config.ordering_cost_retailer)
        holding_cost = np.sum(self.inventory * self.config.holding_cost_retailer)
        stockout_cost = np.sum(stockout * self.config.stockout_cost_retailer)
        
        daily_profit = revenue - ordering_cost - holding_cost - stockout_cost
        self.profit += daily_profit
        
        # Update history
        self.demand_history.append(demand)
        self.order_history.append(order)
        
        return daily_profit

class WarehouseAgent:
    """Warehouse agent in the supply chain"""
    
    def __init__(self, config: Config):
        self.config = config
        self.inventory = np.ones(config.num_products) * config.warehouse_initial_inventory
        self.capacity = config.warehouse_capacity
        self.pending_orders = []  # Orders in transit from suppliers
        self.retailer_demand_history = []
        self.order_history = []
        self.profit = 0.0
        self.day_counter = 0
        
    def get_state(self) -> np.ndarray:
        """Get current state of the warehouse"""
        # State includes: inventory, predicted demand, ordering cost, lead time, 
        # downstream inventory, cycle position
        state = np.concatenate([
            self.inventory / self.capacity,  # Normalized inventory
            self._get_demand_prediction(),
            [self.config.ordering_cost_warehouse / 10.0],  # Normalized ordering cost
            [self.config.lead_time / 10.0],  # Normalized lead time
            self._get_downstream_inventory_summary(),
            [self.is_order_day()]
        ])
        return state
    
    def _get_demand_prediction(self) -> np.ndarray:
        """Get predicted downstream demand"""
        if len(self.retailer_demand_history) < 7:
            return np.ones(self.config.num_products) * 0.5
        
        # Simple moving average prediction
        recent_demand = np.array(self.retailer_demand_history[-14:])
        return np.mean(recent_demand, axis=0) / 1000.0  # Normalize
    
    def _get_downstream_inventory_summary(self) -> np.ndarray:
        """Get summary of downstream retailer inventories"""
        # This would be provided by the environment
        return np.zeros(self.config.num_products)  # Placeholder
    
    def is_order_day(self) -> float:
        """Check if today is an ordering day for the warehouse"""
        return 1.0 if self.day_counter % self.config.warehouse_order_cycle == 0 else 0.0
    
    def step(self, order_to_suppliers: np.ndarray, 
             retailer_orders: np.ndarray, 
             retailer_deliveries: np.ndarray) -> float:
        """Execute one step for the warehouse"""
        self.day_counter += 1
        
        # Receive pending orders from suppliers (with lead time)
        if len(self.pending_orders) > self.config.lead_time:
            received = self.pending_orders.pop(0)
            self.inventory += received
            self.inventory = np.minimum(self.inventory, self.capacity)
        
        # Add new order to pending (will arrive after lead time)
        if self.is_order_day():
            self.pending_orders.append(order_to_suppliers)
        
        # Update inventory after sending to retailers
        self.inventory -= retailer_deliveries
        self.inventory = np.maximum(self.inventory, 0)
        
        # Calculate warehouse profit
        revenue = np.sum(retailer_deliveries * self.config.wholesale_price)
        ordering_cost = np.sum(order_to_suppliers * self.config.ordering_cost_warehouse)
        holding_cost = np.sum(self.inventory * self.config.holding_cost_warehouse)
        supplier_cost = np.sum(order_to_suppliers * self.config.supplier_price)
        
        # Calculate stockout cost (unfulfilled retailer demand)
        total_demand = np.sum(retailer_orders, axis=0)
        fulfilled = np.sum(retailer_deliveries, axis=0)
        stockout = np.maximum(total_demand - fulfilled, 0)
        stockout_cost = np.sum(stockout * self.config.stockout_cost_warehouse)
        
        daily_profit = revenue - ordering_cost - holding_cost - supplier_cost - stockout_cost
        self.profit += daily_profit
        
        # Update history
        self.retailer_demand_history.append(total_demand)
        self.order_history.append(order_to_suppliers)
        
        return daily_profit

class SupplyChainEnv:
    """Supply Chain Environment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.retailers = [RetailerAgent(i, config) for i in range(config.num_retailers)]
        self.warehouse = WarehouseAgent(config)
        self.current_step = 0
        self.episode_profits = []
        
        # Load sales data
        self.data_loader = SalesDataLoader(config.demand_data_path, config)
        self.demand_sequence = None
        self.current_demand_idx = 0
        
        # Get demand statistics for normalization
        self.demand_stats = self.data_loader.get_demand_statistics()
        
        # Store embeddings for context
        self.item_embeddings = self.data_loader.get_item_embeddings()
        self.store_embeddings = self.data_loader.get_store_embeddings()
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        # Reset all agents
        for retailer in self.retailers:
            retailer.inventory = np.ones(self.config.num_products) * self.config.retailer_initial_inventory
            retailer.demand_history = []
            retailer.order_history = []
            retailer.profit = 0.0
            
        self.warehouse = WarehouseAgent(self.config)
        self.current_step = 0
        
        # Load new demand sequence for this episode
        self.demand_sequence = self.data_loader.get_demand_sequence(
            start_date=None,  # Random start date
            length=self.config.episode_length
        )
        self.current_demand_idx = 0
        
        # Return initial observations
        observations = {
            'warehouse': self.warehouse.get_state(),
            'retailers': np.array([r.get_state() for r in self.retailers]),
            'global': self._get_global_state()
        }
        
        return observations
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one environment step
        actions: Dictionary containing 'warehouse' and 'retailers' actions
        """
        self.current_step += 1
        
        # Parse actions
        warehouse_action = actions['warehouse']
        retailer_actions = actions['retailers']  # Shape: (num_retailers, num_products)
        predicted_demand = actions.get('predicted_demand', self._get_current_demand())
        
        # Get actual demand from loaded data
        actual_demand = self._get_current_demand()
        self.current_demand_idx += 1
        
        # Process retailer orders
        total_retailer_orders = np.sum(retailer_actions, axis=0)
        
        # Warehouse fulfillment (prioritize by order size)
        retailer_deliveries = self._fulfill_retailer_orders(retailer_actions)
        
        # Execute warehouse step
        warehouse_reward = self.warehouse.step(
            warehouse_action, 
            retailer_actions,
            retailer_deliveries
        )
        
        # Execute retailer steps
        retailer_rewards = []
        for i, retailer in enumerate(self.retailers):
            reward = retailer.step(
                retailer_actions[i],
                retailer_deliveries[i],
                actual_demand[i]
            )
            retailer_rewards.append(reward)
        
        # Get new observations
        observations = {
            'warehouse': self.warehouse.get_state(),
            'retailers': np.array([r.get_state() for r in self.retailers]),
            'global': self._get_global_state()
        }
        
        # Compute rewards
        rewards = {
            'warehouse': warehouse_reward,
            'retailers': np.array(retailer_rewards),
            'system': warehouse_reward + np.sum(retailer_rewards)
        }
        
        # Check if episode is done
        done = self.current_step >= self.config.episode_length
        
        # Additional info
        info = {
            'actual_demand': actual_demand,
            'predicted_demand': predicted_demand,
            'stockout_retailers': self._calculate_stockouts(),
            'inventory_levels': self._get_inventory_levels()
        }
        
        return observations, rewards, done, info
    
    def _get_current_demand(self) -> np.ndarray:
        """Get current demand from loaded data sequence"""
        if self.demand_sequence is None or self.current_demand_idx >= self.demand_sequence['length']:
            # If no data or reached end, return zeros
            return np.zeros((self.config.num_retailers, self.config.num_products))
        
        # Collect demand for all retailers at current time step
        demand = []
        for retailer_idx in range(self.config.num_retailers):
            store_key = f'store_{retailer_idx}'
            if store_key in self.demand_sequence['demand']:
                store_demand = self.demand_sequence['demand'][store_key]
                if self.current_demand_idx < len(store_demand):
                    demand.append(store_demand[self.current_demand_idx])
                else:
                    demand.append(np.zeros(self.config.num_products))
            else:
                demand.append(np.zeros(self.config.num_products))
        
        return np.array(demand, dtype=np.float32)
    
    def _fulfill_retailer_orders(self, retailer_orders: np.ndarray) -> np.ndarray:
        """
        Fulfill retailer orders based on warehouse inventory
        Prioritize retailers with larger orders
        """
        deliveries = np.zeros_like(retailer_orders)
        
        for p in range(self.config.num_products):
            available = self.warehouse.inventory[p]
            product_orders = retailer_orders[:, p]
            
            # Sort retailers by order size (descending)
            sorted_indices = np.argsort(product_orders)[::-1]
            
            for idx in sorted_indices:
                if available > 0:
                    delivered = min(product_orders[idx], available)
                    deliveries[idx, p] = delivered
                    available -= delivered
                    
        return deliveries
    
    def _get_global_state(self) -> np.ndarray:
        """Get global state for centralized critic"""
        # Combine all agent states
        warehouse_state = self.warehouse.get_state()
        retailer_states = np.concatenate([r.get_state() for r in self.retailers])
        
        # Add system-level features
        total_inventory = np.sum([r.inventory for r in self.retailers], axis=0)
        total_inventory = np.append(total_inventory, self.warehouse.inventory)
        
        global_state = np.concatenate([
            warehouse_state,
            retailer_states,
            total_inventory / 1000.0,  # Normalized
            [self.current_step / self.config.episode_length]  # Time progress
        ])
        
        return global_state
    
    def _calculate_stockouts(self) -> np.ndarray:
        """Calculate stockout information"""
        stockouts = []
        for retailer in self.retailers:
            if len(retailer.demand_history) > 0 and len(retailer.order_history) > 0:
                last_demand = retailer.demand_history[-1]
                last_inventory = retailer.inventory
                stockout = np.maximum(last_demand - last_inventory, 0)
                stockouts.append(np.sum(stockout))
            else:
                stockouts.append(0)
        return np.array(stockouts)
    
    def _get_inventory_levels(self) -> Dict:
        """Get current inventory levels"""
        return {
            'warehouse': self.warehouse.inventory.copy(),
            'retailers': np.array([r.inventory.copy() for r in self.retailers])
        }
    
    def get_context_features(self) -> np.ndarray:
        """Get context features from loaded data (prices, seasonality, etc.)"""
        if self.demand_sequence and 'context' in self.demand_sequence:
            context = self.demand_sequence['context']
            if self.current_demand_idx < len(context['day_of_week']):
                features = [
                    context['day_of_week'][self.current_demand_idx] / 7.0,  # Normalized day of week
                    context['month'][self.current_demand_idx] / 12.0,  # Normalized month
                    np.sin(2 * np.pi * self.current_demand_idx / 7),  # Weekly cycle
                    np.cos(2 * np.pi * self.current_demand_idx / 7),
                    np.sin(2 * np.pi * self.current_demand_idx / 30),  # Monthly cycle
                    np.cos(2 * np.pi * self.current_demand_idx / 30),
                ]
                return np.array(features)
        
        # Return default context if no data
        return np.zeros(6)
