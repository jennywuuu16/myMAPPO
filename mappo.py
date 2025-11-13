"""
MAPPO Algorithm Implementation for Supply Chain Management
Multi-Agent Proximal Policy Optimization with Centralized Training and Decentralized Execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import Config
from networks import (
    RetailerActor, WarehouseActor, CentralizedCritic,
    PredictorActor
)

@dataclass
class Transition:
    """Single transition in replay buffer"""
    state: Dict[str, torch.Tensor]
    action: Dict[str, torch.Tensor]
    reward: Dict[str, torch.Tensor]
    next_state: Dict[str, torch.Tensor]
    done: torch.Tensor
    log_prob: Dict[str, torch.Tensor]
    value: torch.Tensor
    global_state: torch.Tensor

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, config: Config):
        self.device = torch.device(config.device)
        self.buffer_size = buffer_size
        self.config = config
        self.clear()
        
    def clear(self):
        """Clear the buffer"""
        self.states = {'warehouse': [], 'retailers': [], 'predictor': []}
        self.actions = {'warehouse': [], 'retailers': [], 'predictor': []}
        self.rewards = {'warehouse': [], 'retailers': [], 'predictor': []}
        self.next_states = {'warehouse': [], 'retailers': [], 'predictor': []}
        self.dones = []
        self.log_probs = {'warehouse': [], 'retailers': [], 'predictor': []}
        self.values = []
        self.global_states = []
        self.advantages = {'warehouse': None, 'retailers': None, 'predictor': None}
        self.returns = None
        
    def add(self, transition: Transition):
        """Add a transition to the buffer"""
        for key in self.states.keys():
            if key in transition.state:
                self.states[key].append(transition.state[key])
                self.actions[key].append(transition.action[key])
                self.rewards[key].append(transition.reward[key])
                self.next_states[key].append(transition.next_state[key])
                self.log_probs[key].append(transition.log_prob[key])
                
        self.dones.append(transition.done)
        self.values.append(transition.value)
        self.global_states.append(transition.global_state)
        
    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute returns and GAE advantages"""
        device = last_value.device
        
        # Convert lists to tensors
        values = torch.cat(self.values).to(device)
        rewards_dict = {}
        for key in self.rewards.keys():
            if len(self.rewards[key]) > 0:
                rewards_dict[key] = torch.stack(self.rewards[key]).to(device)
        
        dones = torch.stack(self.dones).to(device)
        
        # Compute returns and advantages for each agent type
        self.advantages = {}
        self.returns = {}
        
        for key in rewards_dict.keys():
            rewards = rewards_dict[key]
            
            # Initialize
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            
            # Compute GAE
            last_gae_lambda = 0
            T = len(rewards)
            
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
                last_gae_lambda = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lambda
                advantages[t] = last_gae_lambda
                returns[t] = advantages[t] + values[t]
            
            self.advantages[key] = advantages
            self.returns[key] = returns
    
    def get_batch(self, batch_size: int) -> Dict:
        """Get a batch of data for training"""
        # Flatten data
        # mappo.py  RolloutBuffer.get_batch
        data = {}
        for key in self.states.keys():
            if len(self.states[key]) == 0:
                continue

            if key == 'predictor':
        # predictor state 是 dict，需要把每个字段分别堆叠
                hist = torch.stack([
                    torch.as_tensor(d['historical'], dtype=torch.float32, device=self.device)
                    for d in self.states[key]
                ])
                ctx = torch.stack([
                    torch.as_tensor(d['context'], dtype=torch.float32, device=self.device)
                    for d in self.states[key]
                ])
                data['predictor_historical'] = hist
                data['predictor_context'] = ctx
        # 下面三个张量本身就不是 dict，直接堆
                data['predictor_actions'] = torch.stack(self.actions[key])
                data['predictor_log_probs'] = torch.stack(self.log_probs[key])
                data['predictor_advantages'] = self.advantages[key]
                data['predictor_returns'] = self.returns[key]
            else:
        # 原有逻辑
                data[f'{key}_states'] = torch.stack([
                    x if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(dtype=torch.float32, device=self.device)
                    for x in self.states[key]
                ])
                data[f'{key}_actions'] = torch.stack(self.actions[key])
                data[f'{key}_log_probs'] = torch.stack(self.log_probs[key])
                data[f'{key}_advantages'] = self.advantages[key]
                data[f'{key}_returns'] = self.returns[key]
# 公共部分不变
                data['global_states'] = torch.stack(self.global_states).to(self.device)
                data['values'] = torch.cat(self.values)

                '''# 在 get_batch 末尾加
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                return batch'''
        
        # Random sampling
        num_samples = len(self.global_states)
        indices = np.random.choice(num_samples, min(batch_size, num_samples), replace=False)
        
        batch = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                batch[key] = value[indices]
        
        return batch

class MAPPO:
    """Multi-Agent PPO Algorithm"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize buffer
        self.buffer = RolloutBuffer(config.episode_length, config)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'predictor_loss': [],
            'episode_rewards': []
        }
        
    def _init_networks(self):
        """Initialize all actor and critic networks"""
        # Create a dummy environment to get actual state dimensions
        from environment import SupplyChainEnv
        dummy_env = SupplyChainEnv(self.config)
        dummy_obs = dummy_env.reset()
        
        # Get actual dimensions from the environment
        warehouse_state_dim = dummy_obs['warehouse'].shape[0] if len(dummy_obs['warehouse'].shape) > 0 else len(dummy_obs['warehouse'])
        retailer_state_dim = dummy_obs['retailers'][0].shape[0] if len(dummy_obs['retailers'][0].shape) > 0 else len(dummy_obs['retailers'][0])
        global_state_dim = dummy_obs['global'].shape[0] if len(dummy_obs['global'].shape) > 0 else len(dummy_obs['global'])
        
        print(f"Detected state dimensions:")
        print(f"  Warehouse state: {warehouse_state_dim}")
        print(f"  Retailer state: {retailer_state_dim}")
        print(f"  Global state: {global_state_dim}")
        
        # Retailer actors (one for each retailer)
        self.retailer_actors = nn.ModuleList([
            RetailerActor(retailer_state_dim, self.config.num_products, self.config)
            for _ in range(self.config.num_retailers)
        ]).to(self.device)
        
        # Warehouse actor
        self.warehouse_actor = WarehouseActor(
            warehouse_state_dim, self.config.num_products, self.config
        ).to(self.device)
        
        # Predictor actor
        self.predictor_actor = PredictorActor(
            retailer_state_dim, self.config
        ).to(self.device)
        
        # Centralized critic
        self.critic = CentralizedCritic(global_state_dim, self.config).to(self.device)
        
        # Clean up dummy environment
        del dummy_env
        
    def _init_optimizers(self):
        """Initialize optimizers for all networks"""
        # Collect all actor parameters
        actor_params = []
        for actor in self.retailer_actors:
            actor_params.extend(actor.parameters())
        actor_params.extend(self.warehouse_actor.parameters())
        actor_params.extend(self.predictor_actor.parameters())
        
        # Actor optimizer
        self.actor_optimizer = optim.Adam(
            actor_params,
            lr=self.config.learning_rate
        )
        
        # Critic optimizer
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )
    
    def select_actions(self, states: Dict, deterministic: bool = False) -> Dict:
        """Select actions for all agents"""
        actions = {}
        log_probs = {}
        
        # Retailer actions
        retailer_states = states['retailers']
        retailer_actions = []
        retailer_log_probs = []
        
        for i, actor in enumerate(self.retailer_actors):
            state_tensor = torch.FloatTensor(retailer_states[i]).unsqueeze(0).to(self.device)
            action, log_prob, _ = actor.get_action(state_tensor, deterministic)
            retailer_actions.append(action.detach().cpu().numpy().squeeze())
            retailer_log_probs.append(log_prob)
        
        actions['retailers'] = np.array(retailer_actions)
        log_probs['retailers'] = torch.cat(retailer_log_probs)
        
        # Warehouse action
        warehouse_state = torch.FloatTensor(states['warehouse']).unsqueeze(0).to(self.device)
        is_order_day = warehouse_state[0, -1].item() > 0.5  # Check cycle position
        
        warehouse_action, warehouse_log_prob, _ = self.warehouse_actor.get_action(
            warehouse_state, is_order_day, deterministic
        )
        actions['warehouse'] = warehouse_action.detach().cpu().numpy().squeeze()
        log_probs['warehouse'] = warehouse_log_prob
        
        # Predictor action (demand prediction)
        if 'predictor' in states:
            # Prepare historical data and context
            historical_sales = torch.FloatTensor(states['predictor']['historical']).unsqueeze(0).to(self.device)
            context = torch.FloatTensor(states['predictor']['context']).unsqueeze(0).to(self.device)
            
            pred_action, pred_log_prob, _ = self.predictor_actor.get_action(
                historical_sales, context, deterministic
            )
            actions['predicted_demand'] = pred_action.detach().cpu().numpy().squeeze()
            log_probs['predictor'] = pred_log_prob
        
        return actions, log_probs
    
    def compute_value(self, global_state: np.ndarray) -> torch.Tensor:
        """Compute state value using centralized critic"""
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value
    
    def update(self):
        """Update all networks using PPO"""
        # Compute returns and advantages
        last_global_state = self.buffer.global_states[-1]
        last_value = self.compute_value(last_global_state)
        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        
        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Get batch
            batch = self.buffer.get_batch(self.config.batch_size)
            
            # Update actors
            actor_loss = self._update_actors(batch)
            
            # Update critic
            critic_loss = self._update_critic(batch)
            
            # Update predictor with decision error
            predictor_loss = self._update_predictor(batch)
            
            # Store statistics
            self.training_stats['actor_loss'].append(actor_loss)
            self.training_stats['critic_loss'].append(critic_loss)
            self.training_stats['predictor_loss'].append(predictor_loss)
    
    def _update_actors(self, batch: Dict) -> float:
        """Update all actor networks"""
        total_loss = 0
        
        # Update retailer actors
        if 'retailers_states' in batch:
            retailer_states = batch['retailers_states']
            retailer_actions = batch['retailers_actions']
            retailer_log_probs_old = batch['retailers_log_probs']
            retailer_advantages = batch['retailers_advantages']
            
            for i, actor in enumerate(self.retailer_actors):
                # Get current log probs
                states_i = retailer_states[:, i].to(self.device)
                actions_i = retailer_actions[:, i].to(self.device)
                
                mean, log_std = actor(states_i)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions_i).sum(dim=-1, keepdim=True)
                
                # Compute ratio
                ratio = torch.exp(log_probs - retailer_log_probs_old[:, i])
                
                # Compute surrogate losses
                advantages_i = retailer_advantages[:, i]
                surr1 = ratio * advantages_i
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages_i
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                actor_loss -= self.config.entropy_coef * entropy
                
                total_loss += actor_loss.item()
        
        # Update warehouse actor
        if 'warehouse_states' in batch:
            warehouse_states = batch['warehouse_states'].to(self.device)
            warehouse_actions = batch['warehouse_actions'].to(self.device)
            warehouse_log_probs_old = batch['warehouse_log_probs'].to(self.device)
            warehouse_advantages = batch['warehouse_advantages'].to(self.device)
            
            # Check if order day
            is_order_day = warehouse_states[:, -1] > 0.5
            
            if is_order_day.any():
                # Filter to order days only
                order_day_indices = is_order_day.nonzero(as_tuple=True)[0]
                states_filtered = warehouse_states[order_day_indices]
                actions_filtered = warehouse_actions[order_day_indices]
                log_probs_old_filtered = warehouse_log_probs_old[order_day_indices]
                advantages_filtered = warehouse_advantages[order_day_indices]
                
                # Get current log probs
                mean, log_std = self.warehouse_actor(states_filtered)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions_filtered).sum(dim=-1, keepdim=True)
                
                # Compute ratio and surrogate loss
                ratio = torch.exp(log_probs - log_probs_old_filtered)
                surr1 = ratio * advantages_filtered
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages_filtered
                
                warehouse_loss = -torch.min(surr1, surr2).mean()
                warehouse_loss -= self.config.entropy_coef * dist.entropy().mean()
                
                total_loss += warehouse_loss.item()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        if total_loss > 0:
            # Create computational graph for backward pass
            loss_tensor = torch.tensor(total_loss, requires_grad=True)
            loss_tensor.backward()
            nn.utils.clip_grad_norm_(
                self.actor_optimizer.param_groups[0]['params'],
                self.config.max_grad_norm
            )
            self.actor_optimizer.step()
        
        return total_loss
    
    def _update_critic(self, batch: Dict) -> float:
        """Update centralized critic"""
        global_states = batch['global_states']
        returns = batch.get('retailers_returns', batch.get('warehouse_returns'))
        
        if returns is None:
            return 0.0
        
        # Flatten returns if needed
        if len(returns.shape) > 2:
            returns = returns.view(-1, 1)
        
        # Get value predictions
        values = self.critic(global_states)
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Optimize
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        return value_loss.item()
    
    def _update_predictor(self, batch: Dict) -> float:
        """Update predictor using decision error as reward"""
        if 'predictor_states' not in batch:
            return 0.0
        
        # Get predictor data
        predictor_advantages = batch['predictor_advantages']
        
        # Predictor loss is negative decision error
        # This encourages predictions that lead to better ordering decisions
        predictor_loss = -predictor_advantages.mean()
        
        # Add auxiliary MSE loss if available
        if 'predictor_mse' in batch:
            mse_loss = batch['predictor_mse'].mean()
            predictor_loss += 0.1 * mse_loss  # Small weight for MSE
        
        # Optimize predictor
        predictor_params = list(self.predictor_actor.parameters())
        predictor_optimizer = optim.Adam(predictor_params, lr=self.config.learning_rate)
        
        predictor_optimizer.zero_grad()
        predictor_loss.backward()
        nn.utils.clip_grad_norm_(predictor_params, self.config.max_grad_norm)
        predictor_optimizer.step()
        
        return predictor_loss.item()
    
    def save_models(self, path: str):
        """Save all models"""
        torch.save({
            'retailer_actors': self.retailer_actors.state_dict(),
            'warehouse_actor': self.warehouse_actor.state_dict(),
            'predictor_actor': self.predictor_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.retailer_actors.load_state_dict(checkpoint['retailer_actors'])
        self.warehouse_actor.load_state_dict(checkpoint['warehouse_actor'])
        self.predictor_actor.load_state_dict(checkpoint['predictor_actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)