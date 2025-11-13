"""
Configuration file for MAPPO Supply Chain Management
"""

import numpy as np
import torch

class Config:
    """Configuration class for supply chain MAPPO parameters"""
    
    # Environment parameters
    num_retailers = 1  # Number of retail stores
    num_products = 2    # Number of products
    num_suppliers = 1   # Number of suppliers
    num_episodes = 1000
    episode_length = 100  # T periods
    
    # Warehouse parameters
    warehouse_order_cycle = 4  # Warehouse orders every M days
    lead_time = 3  # L days lead time for warehouse orders
    warehouse_capacity = 10000
    warehouse_initial_inventory = 500
    
    # Retailer parameters
    retailer_capacity = 100
    retailer_initial_inventory = 50
    retailer_order_cycle = 1  # Daily ordering
    
    # Cost parameters
    holding_cost_retailer = 0.1  # h_i
    holding_cost_warehouse = 0.05  # h_w
    stockout_cost_retailer = 2.0  # b_i
    stockout_cost_warehouse = 1.5  # b_w
    ordering_cost_retailer = 0.3  # c_i
    ordering_cost_warehouse = 0.1  # c_w

    wholesale_price_ratio=0.6  # 
    supplier_price_ratio=0.3  #
    
    # Data parameters
    demand_data_path = '/home/ubuntu/Desktop/paper4_data/store_CA_1_product2.csv'  # Path to historical sales data
    train_test_split = 0.8  # Ratio for train/test split
    sequence_length = 30  # Days of history to use for prediction
    
    # MAPPO parameters
    learning_rate = 3e-4
    gamma = 0.99  # discount factor
    gae_lambda = 0.95
    eps_clip = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    
    # Training parameters
    batch_size = 64
    mini_batch_size = 32
    ppo_epochs = 4
    update_frequency = 10  # Update every N episodes
    
    # Network parameters
    hidden_size = 256
    lstm_hidden_size = 128
    num_layers = 3
    activation = 'relu'
    
    # Prediction parameters
    prediction_horizon = 4  # H days forecast
    history_length = 30  # Historical data for prediction
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_interval = 100
    save_interval = 1000
    model_save_path = './models/'
    log_path = './logs/'
