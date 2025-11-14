"""
Main Training Script for MAPPO Supply Chain Management
"""
import torch
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict,List
from config import Config
#from environment import SupplyChainEnv
from mappo import MAPPO, Transition
from utils import calculate_counterfactual_profit
from data_loader import SalesDataLoader

import environment
import os
import importlib
import environment
importlib.reload(environment)
from environment import SupplyChainEnv, RetailerAgent

print(">>> Environment module loaded from:", environment.__file__)

print("Retailer state check:", len(RetailerAgent(0, Config()).get_state()))

def prepare_predictor_state(env: SupplyChainEnv, config: Config) -> Dict:
    """Prepare state for predictor agent using actual historical data"""
    # Get historical sales data from environment's data loader
    if len(env.retailers[0].demand_history) >= config.history_length:
        # Use actual historical data from retailers
        historical_sales = []
        for retailer in env.retailers:
            recent_history = retailer.demand_history[-config.history_length:]
            historical_sales.append(recent_history)
        
        historical_sales = np.array(historical_sales)  # Shape: (num_retailers, history_length, num_products)
    else:
        # For initial steps, use data from the loader
        if env.demand_sequence and env.current_demand_idx > 0:
            # Get historical data from loaded sequence
            start_idx = max(0, env.current_demand_idx - config.history_length)
            historical_sales = []
            
            for retailer_idx in range(config.num_retailers):
                store_key = f'store_{retailer_idx}'
                if store_key in env.demand_sequence['demand']:
                    store_data = env.demand_sequence['demand'][store_key]
                    history = store_data[start_idx:env.current_demand_idx]
                    
                    # Pad if necessary
                    if len(history) < config.history_length:
                        padding = np.zeros((config.history_length - len(history), config.num_products))
                        history = np.vstack([padding, history])
                    
                    historical_sales.append(history)
                else:
                    historical_sales.append(np.zeros((config.history_length, config.num_products)))
            
            historical_sales = np.array(historical_sales)
        else:
            # No data available yet, use zeros
            historical_sales = np.zeros((config.num_retailers, config.history_length, config.num_products))
    
    # Get context features from environment
    context_features = env.get_context_features()
    
    # Add additional context
    context = np.concatenate([
        context_features,
        [env.current_step / config.episode_length],  # Time progress
        [float(env.warehouse.is_order_day())],  # Warehouse order day indicator
        [0.0, 0.0]  # Padding to reach 10 dimensions
    ])[:10]  # Ensure exactly 10 dimensions
    
    return {
        'historical': historical_sales.mean(axis=0),  # Average across retailers
        'context': context
    }

def calculate_decision_error(env: SupplyChainEnv,
                            predicted_demand: np.ndarray,
                            actual_demand: np.ndarray,
                            system_reward: float,
                            config: Config,
                            alpha: float=0.7) -> float:
    """
    Calculate predictor reward based on System performance
    Predictor gets rewardd when its forcasts lead to profitable decisions
    Calculate decision error for predictor agent
    """
    #prediction_reward = system_reward * 0.1

    # Calculate profit achieved with predicted demand (already computed in environment step)
    predicted_profit = env.warehouse.profit + sum(r.profit for r in env.retailers)
    
    # Calculate theoretical optimal profit with perfect demand knowledge
    optimal_profit = calculate_counterfactual_profit(env, actual_demand, config)
    
    # Decision error (negative because we want to minimize the gap)
    profit_gap = abs(predicted_profit - optimal_profit)/(optimal_profit, 1e-5)
    
    # Calculate prediction accuracy using negative MSE (higher is better)
    if predicted_demand.ndim == 3:
        # Shape: (num_retailers, horizon, num_products)
        # Use only next-day predictions for reward
        predicted_next_day = predicted_demand[:, 0, :]
    else:
        predicted_next_day = predicted_demand

    # Calculate MSE between prediction and actual
    mse = np.mean((predicted_next_day - actual_demand) ** 2)
    
    total_reward = -(alpha* profit_gap + (1 - alpha) * mse)
    print(f"optimal profit:{optimal_profit},predicted profit:{predicted_profit}")

    return total_reward

def train_mappo(config: Config):
    """Main training function"""
    
    # First validate that data is available
    print(f"Loading sales data from: {config.demand_data_path}")
    data_loader = SalesDataLoader(config.demand_data_path, config)
    
    # Print data statistics
    stats = data_loader.get_demand_statistics()
    print(f"\nData Statistics:")
    print(f"  - Number of stores: {data_loader.num_stores}")
    print(f"  - Number of items: {data_loader.num_items}")
    print(f"  - Number of days: {data_loader.num_days}")
    print(f"  - Mean demand per item: {np.mean(list(stats['mean_demand'].values())):.2f}")
    
    # Create environment and algorithm
    env = SupplyChainEnv(config)
    mappo = MAPPO(config)

    # ============================
# Load pretrained Predictor model
# ============================
    predictor_path = os.path.join(config.model_save_path, 'predictor_pretrained.pt')
    if os.path.exists(predictor_path):
        try:
            mappo.predictor_actor.load_state_dict(
                torch.load(predictor_path, map_location=config.device)
            )
            print(f"✅ Loaded pretrained predictor weights from {predictor_path}")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained predictor model: {e}")
    else:
        print("⚠️ No pretrained predictor model found — training from scratch.")

    
    # Training metrics
    episode_rewards = []
    system_profits = []
    predictor_errors = []
    
    # Create directories for saving
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)
    
    # Training loop
    for episode in tqdm(range(config.num_episodes), desc="Training MAPPO"):
        # Reset environment
        observations = env.reset()
        episode_reward = 0
        episode_profit = 0
        episode_predictor_error = 0
        
        # Episode loop
        for step in range(config.episode_length):
            # Prepare states for all agents
            states = {
                'warehouse': observations['warehouse'],
                'retailers': observations['retailers'],
                'predictor': prepare_predictor_state(env, config)
            }
            
            # Select actions
            actions, log_probs = mappo.select_actions(states, deterministic=False)
            
            # Step environment
            next_observations, rewards, done, info = env.step(actions)
            
            # Calculate predictor reward (decision error)
            if 'predicted_demand' in actions:
                decision_error = calculate_decision_error(
                    env, 
                    actions['predicted_demand'],
                    info['actual_demand'],
                    rewards['system'],
                    config
                )
                rewards['predictor'] = decision_error
                episode_predictor_error += decision_error
            
            # Store transition
            transition = Transition(
                state=states,
                action={
                    'warehouse': torch.FloatTensor(actions['warehouse']),
                    'retailers': torch.FloatTensor(actions['retailers']),
                    'predictor': torch.FloatTensor(actions.get('predicted_demand', np.zeros((config.prediction_horizon, config.num_products))))
                },
                reward={
                    'warehouse': torch.FloatTensor([rewards['warehouse']]),
                    'retailers': torch.FloatTensor(rewards['retailers']),
                    'predictor': torch.FloatTensor([rewards.get('predictor', 0.0)])
                },
                next_state={
                    'warehouse': next_observations['warehouse'],
                    'retailers': next_observations['retailers'],
                    'predictor': prepare_predictor_state(env, config)
                },
                done=torch.FloatTensor([float(done)]),
                log_prob=log_probs,
                value=mappo.compute_value(observations['global']),
                global_state=torch.FloatTensor(observations['global'])
            )
            
            mappo.buffer.add(transition)
            
            # Update observations
            observations = next_observations
            
            # Accumulate rewards
            episode_reward += rewards['system']
            episode_profit += rewards['warehouse'] + np.sum(rewards['retailers'])
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        system_profits.append(episode_profit)
        predictor_errors.append(episode_predictor_error)
        
        # Update networks
        if (episode + 1) % config.update_frequency == 0:
            mappo.update()
            mappo.buffer.clear()
        
        # Logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_profit = np.mean(system_profits[-config.log_interval:])
            avg_pred_error = np.mean(predictor_errors[-config.log_interval:])
            
            print(f"\nEpisode {episode + 1}/{config.num_episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average System Profit: {avg_profit:.2f}")
            print(f"Average Predictor Error: {avg_pred_error:.2f}")
            
            # Save metrics
            metrics = {
                'episode': episode + 1,
                'avg_reward': float(avg_reward),
                'avg_profit': float(avg_profit),
                'avg_predictor_error': float(avg_pred_error),
                'actor_loss': float(np.mean(mappo.training_stats['actor_loss'][-100:])) if mappo.training_stats['actor_loss'] else 0,
                'critic_loss': float(np.mean(mappo.training_stats['critic_loss'][-100:])) if mappo.training_stats['critic_loss'] else 0
            }
            
            with open(os.path.join(config.log_path, f'metrics_{episode+1}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save models
        if (episode + 1) % config.save_interval == 0:
            model_path = os.path.join(config.model_save_path, f'mappo_episode_{episode+1}.pt')
            mappo.save_models(model_path)
            print(f"Models saved to {model_path}")
    
    # Final save
    final_model_path = os.path.join(config.model_save_path, 'mappo_final.pt')
    mappo.save_models(final_model_path)
    
    # Plot training curves
    plot_training_curves(episode_rewards, system_profits, predictor_errors, config)
    
    return mappo, episode_rewards

def plot_training_curves(episode_rewards: List[float], 
                        system_profits: List[float],
                        predictor_errors: List[float],
                        config: Config):
    """Plot training curves"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Smooth curves using moving average
    window_size = 100
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Raw')
    if len(episode_rewards) > window_size:
        axes[0].plot(range(window_size-1, len(episode_rewards)), 
                    moving_average(episode_rewards, window_size), 
                    label=f'MA({window_size})', linewidth=2)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # System profits
    axes[1].plot(system_profits, alpha=0.3, label='Raw')
    if len(system_profits) > window_size:
        axes[1].plot(range(window_size-1, len(system_profits)), 
                    moving_average(system_profits, window_size), 
                    label=f'MA({window_size})', linewidth=2)
    axes[1].set_title('System Profits')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Profit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Predictor errors
    axes[2].plot(predictor_errors, alpha=0.3, label='Raw')
    if len(predictor_errors) > window_size:
        axes[2].plot(range(window_size-1, len(predictor_errors)), 
                    moving_average(predictor_errors, window_size), 
                    label=f'MA({window_size})', linewidth=2)
    axes[2].set_title('Predictor Decision Errors')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Decision Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(config.log_path, 'training_curves.jpg')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    
    plt.show()

def evaluate_policy(mappo: MAPPO, config: Config, num_episodes: int = 10):
    """Evaluate trained policy"""
    env = SupplyChainEnv(config)
    
    total_rewards = []
    total_profits = []
    retailer_stockouts = []
    warehouse_stockouts = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = 0
        episode_profit = 0
        episode_retailer_stockout = 0
        episode_warehouse_stockout = 0
        
        for step in range(config.episode_length):
            # Prepare states
            states = {
                'warehouse': observations['warehouse'],
                'retailers': observations['retailers'],
                'predictor': prepare_predictor_state(env, config)
            }
            
            # Select actions (deterministic for evaluation)
            actions, _ = mappo.select_actions(states, deterministic=True)
            
            # Step environment
            observations, rewards, done, info = env.step(actions)
            
            # Accumulate metrics
            episode_reward += rewards['system']
            episode_profit += rewards['warehouse'] + np.sum(rewards['retailers'])
            episode_retailer_stockout += np.sum(info['stockout_retailers'])
            
            # Check warehouse stockout
            total_demand = np.sum([r.demand_history[-1] if len(r.demand_history) > 0 else 0 
                                  for r in env.retailers], axis=0)
            warehouse_available = env.warehouse.inventory
            episode_warehouse_stockout += np.sum(np.maximum(total_demand - warehouse_available, 0))
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_profits.append(episode_profit)
        retailer_stockouts.append(episode_retailer_stockout)
        warehouse_stockouts.append(episode_warehouse_stockout)
    
    # Print evaluation results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average System Profit: {np.mean(total_profits):.2f} ± {np.std(total_profits):.2f}")
    print(f"Average Retailer Stockouts: {np.mean(retailer_stockouts):.2f} ± {np.std(retailer_stockouts):.2f}")
    print(f"Average Warehouse Stockouts: {np.mean(warehouse_stockouts):.2f} ± {np.std(warehouse_stockouts):.2f}")
    print("="*50)
    
    return {
        'rewards': total_rewards,
        'profits': total_profits,
        'retailer_stockouts': retailer_stockouts,
        'warehouse_stockouts': warehouse_stockouts
    }

if __name__ == "__main__":
    # Load configuration
    config = Config()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Starting MAPPO Training for Supply Chain Management")
    print(f"Configuration:")
    print(f"  - Retailers: {config.num_retailers}")
    print(f"  - Products: {config.num_products}")
    print(f"  - Episodes: {config.num_episodes}")
    print(f"  - Episode Length: {config.episode_length}")
    print(f"  - Device: {config.device}")
    print()
    
    # Train MAPPO
    trained_mappo, rewards = train_mappo(config)
    # Evaluate trained policy
    print("\nEvaluating trained policy...")
    evaluation_results = evaluate_policy(trained_mappo, config, num_episodes=20)
    
    # Save final results
    final_results = {
        'training_rewards': [float(r) for r in rewards[-1000:]],  # Last 1000 episodes
        'evaluation': {
            'mean_reward': float(np.mean(evaluation_results['rewards'])),
            'std_reward': float(np.std(evaluation_results['rewards'])),
            'mean_profit': float(np.mean(evaluation_results['profits'])),
            'std_profit': float(np.std(evaluation_results['profits']))
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(config.log_path, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nTraining complete!")
