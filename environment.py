"""
Supply Chain Environment for MAPPO
Enhanced with PredictorAgent integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import Config
from data_loader import SalesDataLoader


@dataclass
class Inventory:
    """Inventory state for entities"""
    current: np.ndarray
    capacity: float


class RetailerAgent:
    """Retailer agent in the supply chain"""

    def __init__(self, agent_id: int, config: Config):
        self.id = agent_id
        self.config = config
        self.inventory = np.atleast_1d(np.zeros(config.num_products))
        self.capacity = config.retailer_capacity
        self.demand_history = []
        self.order_history = []
        self.profit = 0.0
        # 新增：存储 Predictor 预测的下一天需求
        self.predicted_next_day = np.zeros(config.num_products)

    def get_state(self) -> np.ndarray:
        """Get current state of the retailer"""
        inventory_normalized = np.atleast_1d(self.inventory / self.capacity)
        demand_encoding = np.atleast_1d(self._get_demand_encoding())
        day_encoding = np.array([self.get_day_encoding()])
        # 新增：加入下一天的预测需求（归一化）
        pred_demand_norm = np.atleast_1d(self.predicted_next_day / 100.0)

        state = np.concatenate([
            inventory_normalized,
            demand_encoding,
            day_encoding,
            pred_demand_norm
        ])
        return state

    def _get_demand_encoding(self) -> np.ndarray:
        """Encode historical demand using simple statistics"""
        if len(self.demand_history) < 7:
            return np.zeros(self.config.num_products * 2)
        history = np.array(self.demand_history[-30:])
        encoding = []
        for p in range(self.config.num_products):
            product_history = history[:, p] if history.ndim > 1 else history
            encoding.extend([
                np.mean(product_history),
                np.std(product_history),
                #np.polyfit(range(len(product_history)), product_history, 1)[0]
            ])
        return np.array(encoding) / 100.0

    def get_day_encoding(self) -> float:
        """Encode cyclical time information"""
        return np.sin(2 * np.pi * len(self.demand_history) / 7)

    def step(self, order: np.ndarray, received: np.ndarray, demand: np.ndarray,
             retail_prices: np.ndarray, wholesale_prices: np.ndarray) -> float:
        """Execute one step for the retailer"""
        self.inventory += received
        sales = np.minimum(self.inventory, demand)
        self.inventory -= sales
        stockout = np.maximum(demand - sales, 0)

        revenue = np.sum(sales * retail_prices)
        ordering_cost = np.sum(order * wholesale_prices)  # Payment to warehouse at wholesale prices
        holding_cost = np.sum(self.inventory * self.config.holding_cost_retailer)
        stockout_cost = np.sum(stockout * self.config.stockout_cost_retailer)
        daily_profit = revenue - ordering_cost - holding_cost - stockout_cost
        self.profit += daily_profit

        self.demand_history.append(demand)
        self.order_history.append(order)
        return daily_profit


class WarehouseAgent:
    """Warehouse agent in the supply chain"""

    def __init__(self, config: Config):
        self.config = config
        self.inventory = np.atleast_1d(np.ones(config.num_products) * config.warehouse_initial_inventory)
        self.capacity = config.warehouse_capacity
        self.pending_orders = []
        self.retailer_demand_history = []
        self.order_history = []
        self.profit = 0.0
        self.day_counter = 0
        # 新增：存储预测的未来4天需求
        self.predicted_demand = np.zeros((config.prediction_horizon, config.num_products))

    def get_state(self) -> np.ndarray:
        """Get current state of the warehouse"""
        inventory_normalized = np.atleast_1d(self.inventory / self.capacity)
        demand_pred = np.atleast_1d(self._get_demand_prediction())
        ordering_cost = np.array([self.config.ordering_cost_warehouse / 10.0])
        lead_time = np.array([self.config.lead_time / 10.0])
        downstream_inv = np.atleast_1d(self._get_downstream_inventory_summary())
        order_day = np.array([self.is_order_day()])

        state = np.concatenate([
            inventory_normalized,
            demand_pred,
            ordering_cost,
            lead_time,
            downstream_inv,
            order_day
        ])
        return state

    def _get_demand_prediction(self) -> np.ndarray:
        """Use PredictorAgent’s 4-day forecast"""
        if hasattr(self, "predicted_demand") and np.any(self.predicted_demand):
            # 平均未来4天预测需求
            avg_forecast = np.mean(self.predicted_demand, axis=0)
            return avg_forecast / 1000.0
        else:
            # fallback
            if len(self.retailer_demand_history) < 4:
                return np.atleast_1d(np.ones(self.config.num_products) * 0.5)
            recent_demand = np.array(self.retailer_demand_history[-14:])
            if recent_demand.ndim == 1:
                return np.atleast_1d(np.mean(recent_demand) / 1000.0)
            else:
                return np.atleast_1d(np.mean(recent_demand, axis=0) / 1000.0)

    def _get_downstream_inventory_summary(self) -> np.ndarray:
        """Placeholder for downstream inventory aggregation"""
        return np.atleast_1d(np.zeros(self.config.num_products))

    def is_order_day(self) -> float:
        return 1.0 if self.day_counter % self.config.warehouse_order_cycle == 0 else 0.0

    def step(self, order_to_suppliers: np.ndarray,
             retailer_orders: np.ndarray,
             retailer_deliveries: np.ndarray,
             wholesale_prices: np.ndarray,
             supplier_prices: np.ndarray) -> float:
        """Execute one step for the warehouse"""
        self.day_counter += 1

        if len(self.pending_orders) > self.config.lead_time:
            received = self.pending_orders.pop(0)
            self.inventory += received
            self.inventory = np.minimum(self.inventory, self.capacity)

        if self.is_order_day():
            self.pending_orders.append(order_to_suppliers)

        self.inventory -= np.array(retailer_deliveries).reshape(-1)
        self.inventory = np.maximum(self.inventory, 0)

        revenue = np.sum(retailer_deliveries * wholesale_prices)
        ordering_cost = np.sum(order_to_suppliers * supplier_prices)  # Payment to suppliers
        holding_cost = np.sum(self.inventory * self.config.holding_cost_warehouse)

        total_demand = np.sum(retailer_orders, axis=0)
        fulfilled = np.sum(retailer_deliveries, axis=0)
        stockout = np.maximum(total_demand - fulfilled, 0)
        stockout_cost = np.sum(stockout * self.config.stockout_cost_warehouse)

        daily_profit = revenue - ordering_cost - holding_cost - stockout_cost
        self.profit += daily_profit

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

        self.data_loader = SalesDataLoader(config.demand_data_path, config)
        self.demand_sequence = None
        self.current_demand_idx = 0

        self.demand_stats = self.data_loader.get_demand_statistics()
        self.item_embeddings = self.data_loader.get_item_embeddings()
        self.store_embeddings = self.data_loader.get_store_embeddings()

        self.retail_prices = None
        self.wholesale_prices = None
        self.supplier_prices = None

    def reset(self) -> Dict[str, np.ndarray]:
        for retailer in self.retailers:
            retailer.inventory = np.atleast_1d(np.ones(self.config.num_products) * self.config.retailer_initial_inventory)
            retailer.demand_history = []
            retailer.order_history = []
            retailer.profit = 0.0
            retailer.predicted_next_day = np.zeros(self.config.num_products)

        self.warehouse = WarehouseAgent(self.config)
        self.current_step = 0

        self.demand_sequence = self.data_loader.get_demand_sequence(
            start_date=None, length=self.config.episode_length)
        self.current_demand_idx = 0

        if 'context' in self.demand_sequence:
            self.retail_prices = np.atleast_1d(self.demand_sequence['context']['retail_prices'])
            self.wholesale_prices = np.atleast_1d(self.demand_sequence['context']['wholesale_prices'])
            self.supplier_prices = np.atleast_1d(self.demand_sequence['context']['supplier_prices'])
        else:
            self.retail_prices = np.ones(self.config.num_products) * 10.0
            self.wholesale_prices = self.retail_prices * self.config.wholesale_price_ratio
            self.supplier_prices = self.retail_prices * self.config.supplier_price_ratio

        observations = {
            'warehouse': self.warehouse.get_state(),
            'retailers': np.array([r.get_state() for r in self.retailers]),
            'global': self._get_global_state()
        }
        return observations

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, bool, Dict]:
        self.current_step += 1
        warehouse_action = actions['warehouse']
        retailer_actions = actions['retailers']
        #predicted_demand = actions.get('predicted_demand', self._get_current_demand())
        predicted_demand = actions.get('predicted_demand', None)
        if predicted_demand is None:
        # 如果没有 Predictor 输出，用当前实际需求重复预测4天
            current_demand = self._get_current_demand()
            predicted_demand = np.repeat(current_demand[:, np.newaxis, :], self.config.prediction_horizon, axis=1)
        elif predicted_demand.ndim == 2:
        # 如果维度是 (num_retailers, num_products)，补成4天
            predicted_demand = np.repeat(predicted_demand[:, np.newaxis, :], self.config.prediction_horizon, axis=1)

        actual_demand = self._get_current_demand()
        self.current_demand_idx += 1

        # === 新增逻辑：将预测结果分发给各 Agent ===
        if predicted_demand is not None:
            # shape: (num_retailers, horizon, num_products)
            for i, retailer in enumerate(self.retailers):
                retailer.predicted_next_day = predicted_demand[i, 0]  # 下一天
            self.warehouse.predicted_demand = np.sum(predicted_demand, axis=0)  # 聚合未来4天
        # ==============================================

        total_retailer_orders = np.sum(retailer_actions, axis=0)
        retailer_deliveries = self._fulfill_retailer_orders(retailer_actions)

        warehouse_reward = self.warehouse.step(
            warehouse_action, retailer_actions, retailer_deliveries,
            self.wholesale_prices, self.supplier_prices
        )

        retailer_rewards = []
        for i, retailer in enumerate(self.retailers):
            reward = retailer.step(
                retailer_actions[i], retailer_deliveries[i],
                actual_demand[i], self.retail_prices, self.wholesale_prices
            )
            retailer_rewards.append(reward)

        observations = {
            'warehouse': self.warehouse.get_state(),
            'retailers': np.array([r.get_state() for r in self.retailers]),
            'global': self._get_global_state()
        }

        rewards = {
            'warehouse': warehouse_reward,
            'retailers': np.array(retailer_rewards),
            'system': warehouse_reward + np.sum(retailer_rewards)
        }

        done = self.current_step >= self.config.episode_length
        info = {
            'actual_demand': actual_demand,
            'predicted_demand': predicted_demand,
            'stockout_retailers': self._calculate_stockouts(),
            'inventory_levels': self._get_inventory_levels()
        }

        return observations, rewards, done, info

    # 以下函数不变
    def _get_current_demand(self) -> np.ndarray:
        if self.demand_sequence is None or self.current_demand_idx >= self.demand_sequence['length']:
            return np.zeros((self.config.num_retailers, self.config.num_products))
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
        deliveries = np.zeros_like(retailer_orders)
        for p in range(self.config.num_products):
            available = self.warehouse.inventory[p]
            product_orders = retailer_orders[:, p]
            sorted_indices = np.argsort(product_orders)[::-1]
            for idx in sorted_indices:
                if available > 0:
                    delivered = min(product_orders[idx], available)
                    deliveries[idx, p] = delivered
                    available -= delivered
        return deliveries

    def _get_global_state(self) -> np.ndarray:
        warehouse_state = self.warehouse.get_state()
        retailer_states = np.concatenate([r.get_state() for r in self.retailers])
        retailer_inventories = [np.atleast_1d(r.inventory) for r in self.retailers]
        if retailer_inventories:
            total_retailer_inventory = np.sum(retailer_inventories, axis=0)
        else:
            total_retailer_inventory = np.zeros(self.config.num_products)
        total_inventory = np.concatenate([
            np.atleast_1d(total_retailer_inventory),
            np.atleast_1d(self.warehouse.inventory)
        ])
        global_state = np.concatenate([
            np.atleast_1d(warehouse_state),
            np.atleast_1d(retailer_states),
            np.atleast_1d(total_inventory / 1000.0),
            np.array([self.current_step / self.config.episode_length])
        ])
        return global_state

    def _calculate_stockouts(self) -> np.ndarray:
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
        return {
            'warehouse': self.warehouse.inventory.copy(),
            'retailers': np.array([r.inventory.copy() for r in self.retailers])
        }

    def get_context_features(self) -> np.ndarray:
        if self.demand_sequence and 'context' in self.demand_sequence:
            context = self.demand_sequence['context']
            if self.current_demand_idx < len(context['day_of_week']):
                features = [
                    context['day_of_week'][self.current_demand_idx] / 7.0,
                    context['month'][self.current_demand_idx] / 12.0,
                    np.sin(2 * np.pi * self.current_demand_idx / 7),
                    np.cos(2 * np.pi * self.current_demand_idx / 7),
                    np.sin(2 * np.pi * self.current_demand_idx / 30),
                    np.cos(2 * np.pi * self.current_demand_idx / 30),
                ]
                return np.array(features)
        return np.zeros(6)
