"""
Neural Network Models for MAPPO Supply Chain
Includes Actor, Critic, and Predictor networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional
from config import Config

class MLPNetwork(nn.Module):
    """Basic MLP network building block"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, activation: str = 'relu'):
        super(MLPNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class RetailerActor(nn.Module):
    """Actor network for retailer agents"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        super(RetailerActor, self).__init__()
        self.config = config
        
        # Main network
        self.network = MLPNetwork(
            state_dim, 
            config.hidden_size, 
            config.hidden_size,
            config.num_layers
        )
        
        # Output layers for mean and std
        self.mean_layer = nn.Linear(config.hidden_size, action_dim)
        self.log_std_layer = nn.Linear(config.hidden_size, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution parameters"""
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, min=-2, max=2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
        
        # Calculate log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing and adjust log_prob
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Scale action to valid range (0 to max order quantity)
        # Improved scaling: use retailer capacity as upper bound for more appropriate actions
        max_order = self.config.retailer_capacity  # Use capacity from config
        action_scaled = (action_tanh + 1) * (max_order / 2.0)  # Scale to [0, capacity]

        return action_scaled, log_prob, dist.entropy().sum(dim=-1, keepdim=True)

class WarehouseActor(nn.Module):
    """Actor network for warehouse agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        super(WarehouseActor, self).__init__()
        self.config = config
        
        # Main network with additional capacity for warehouse complexity
        self.network = MLPNetwork(
            state_dim,
            config.hidden_size * 2,  # Larger network for warehouse
            config.hidden_size * 2,
            config.num_layers
        )
        
        # Output layers
        self.mean_layer = nn.Linear(config.hidden_size * 2, action_dim)
        self.log_std_layer = nn.Linear(config.hidden_size * 2, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp for stability
        log_std = torch.clamp(log_std, min=-2, max=2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, is_order_day: bool, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        if not is_order_day:
            # No ordering on non-order days
            batch_size = state.shape[0] if len(state.shape) > 1 else 1
            action_dim = self.config.num_products
            zeros = torch.zeros((batch_size, action_dim), device=state.device)
            return zeros, torch.zeros((batch_size, 1), device=state.device), torch.zeros((batch_size, 1), device=state.device)
        
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
        
        # Calculate log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Scale to valid range (larger scale for warehouse)
        # Improved scaling: use warehouse capacity as reference
        max_order = self.config.warehouse_capacity * 0.3  # Max order as 30% of capacity per cycle
        action_scaled = (action_tanh + 1) * (max_order / 2.0)

        return action_scaled, log_prob, dist.entropy().sum(dim=-1, keepdim=True)

class CentralizedCritic(nn.Module):
    """Centralized critic for MAPPO"""
    
    def __init__(self, global_state_dim: int, config: Config):
        super(CentralizedCritic, self).__init__()
        
        self.network = MLPNetwork(
            global_state_dim,
            config.hidden_size * 2,
            1,  # Value output
            config.num_layers + 1  # Deeper network for critic
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.network(global_state)

class DemandPredictor(nn.Module):
    """LSTM-based demand predictor agent"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 config: Config):
        super(DemandPredictor, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        
        # LSTM encoder for historical data
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Context encoder (promotions, seasonality, price)
        self.context_encoder = MLPNetwork(
            input_dim=10,  # Context features dimension
            hidden_dim=64,
            output_dim=32,
            num_layers=2
        )
        
        # Prediction network
        self.prediction_network = MLPNetwork(
            input_dim=hidden_dim + 32,  # LSTM output + context
            hidden_dim=config.hidden_size,
            output_dim=output_dim * config.prediction_horizon,  # Predict H days
            num_layers=3
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, historical_sales: torch.Tensor, 
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        historical_sales: (batch, sequence_length, num_products)
        context: (batch, context_dim)
        Returns: predictions, lstm_features
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(historical_sales)
        
        # Use last hidden state
        lstm_features = h_n[-1]  # (batch, hidden_dim)
        
        # Context encoding
        context_features = self.context_encoder(context)
        
        # Combine features
        combined = torch.cat([lstm_features, context_features], dim=-1)
        
        # Predict future demand
        predictions = self.prediction_network(combined)
        
        # Reshape to (batch, horizon, num_products)
        batch_size = predictions.shape[0]
        predictions = predictions.view(batch_size, self.config.prediction_horizon, -1)
        
        # Apply ReLU to ensure non-negative predictions
        predictions = F.relu(predictions)
        
        return predictions, lstm_features
    
    def get_mse_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss for auxiliary training"""
        return F.mse_loss(predictions, targets)

class PredictorActor(nn.Module):
    """Actor network for predictor agent that outputs demand predictions as actions"""
    
    def __init__(self, state_dim: int, config: Config):
        super(PredictorActor, self).__init__()
        self.config = config
        
        # Use the demand predictor as the core
        self.predictor = DemandPredictor(
            input_dim=config.num_products,
            hidden_dim=config.lstm_hidden_size,
            output_dim=config.num_products,
            config=config
        )
        
        # Additional network for policy learning
        self.policy_network = MLPNetwork(
            input_dim=config.lstm_hidden_size,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            num_layers=2
        )
        
        # Output mean and std for each prediction
        output_size = config.num_products * config.prediction_horizon
        self.mean_layer = nn.Linear(config.hidden_size, output_size)
        self.log_std_layer = nn.Linear(config.hidden_size, output_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0)
    
    def forward(self, historical_sales: torch.Tensor, 
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning prediction action"""
        predictions, lstm_features = self.predictor(historical_sales, context)
        
        # Policy network processing
        policy_features = self.policy_network(lstm_features)
        
        # Get distribution parameters
        mean = self.mean_layer(policy_features)
        log_std = self.log_std_layer(policy_features)
        log_std = torch.clamp(log_std, min=-2, max=2)
        
        # Reshape
        batch_size = mean.shape[0]
        mean = mean.view(batch_size, self.config.prediction_horizon, self.config.num_products)
        log_std = log_std.view(batch_size, self.config.prediction_horizon, self.config.num_products)
        
        return mean, log_std, predictions
    
    def get_action(self, historical_sales: torch.Tensor, context: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prediction action"""
        mean, log_std, base_predictions = self.forward(historical_sales, context)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
        
        # Ensure non-negative predictions
        action = F.relu(action)
        
        # Calculate log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=[-2, -1], keepdim=True)
        entropy = dist.entropy().sum(dim=[-2, -1], keepdim=True)
        
        return action, log_prob, entropy
