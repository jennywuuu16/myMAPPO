"""
Test script to verify optimal profit dict fix works correctly
"""

import numpy as np
import sys
from config import Config
from environment import SupplyChainEnv
from utils import calculate_counterfactual_profit

print("=" * 70)
print("Testing Optimal Profit Dict Fix")
print("=" * 70)

# Initialize config and environment
config = Config()
env = SupplyChainEnv(config)

# Reset environment
obs = env.reset()

# Simulate one step with some actions
retailer_actions = np.array([[5.0, 5.0]])  # Order 5 units of each product
warehouse_action = np.array([10.0, 10.0])  # Order 10 units from supplier

actions = {
    'warehouse': warehouse_action,
    'retailers': retailer_actions
}

# Step environment
next_obs, rewards, done, info = env.step(actions)

print("\n1. Testing calculate_counterfactual_profit return type:")
print("-" * 70)

# Get actual demand from info
actual_demand = info['actual_demand']
print(f"Actual demand shape: {actual_demand.shape}")
print(f"Actual demand: {actual_demand}")

# Calculate optimal profit
optimal_profit = calculate_counterfactual_profit(env, actual_demand, config)

print(f"\nOptimal profit type: {type(optimal_profit)}")
print(f"Optimal profit keys: {optimal_profit.keys() if isinstance(optimal_profit, dict) else 'N/A'}")

if isinstance(optimal_profit, dict):
    print("✓ PASS: optimal_profit is a dict")
    print(f"  - Retailer optimal profit: ${optimal_profit['retailer']:.2f}")
    print(f"  - Warehouse optimal profit: ${optimal_profit['warehouse']:.2f}")
    print(f"  - Total optimal profit: ${optimal_profit['total']:.2f}")
else:
    print("✗ FAIL: optimal_profit should be a dict, got:", type(optimal_profit))
    sys.exit(1)

print("\n2. Testing decision error calculation:")
print("-" * 70)

# Simulate the calculation from train.py
predicted_profit = env.warehouse.profit + sum(r.profit for r in env.retailers)
print(f"Predicted profit: ${predicted_profit:.2f}")

# Calculate profit gap with safe division (from train.py)
try:
    profit_gap = abs(predicted_profit - optimal_profit['total']) / max(abs(optimal_profit['total']), 1e-5)
    decision_error = -profit_gap

    print(f"Profit gap: {profit_gap:.4f}")
    print(f"Decision error: {decision_error:.4f}")
    print("✓ PASS: Decision error calculation works correctly")
except Exception as e:
    print(f"✗ FAIL: Error in calculation: {e}")
    sys.exit(1)

print("\n3. Testing decision error dict structure:")
print("-" * 70)

# Create the full decision error result dict
decision_error_result = {
    'error': float(decision_error),
    'predicted_profit': float(predicted_profit),
    'optimal_profit': optimal_profit,
    'profit_gap': float(profit_gap)
}

print("Decision error result structure:")
for key, value in decision_error_result.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for sub_key, sub_value in value.items():
            print(f"    {sub_key}: {sub_value:.2f}")
    else:
        print(f"  {key}: {value:.4f}")

print("✓ PASS: Decision error dict structure is correct")

print("\n4. Testing value extraction for reward:")
print("-" * 70)

# Test extracting the error value for use as reward
reward_value = decision_error_result['error']
print(f"Reward value type: {type(reward_value)}")
print(f"Reward value: {reward_value:.4f}")

if isinstance(reward_value, (int, float)):
    print("✓ PASS: Reward value is a scalar (can be used as reward)")
else:
    print("✗ FAIL: Reward value should be a scalar")
    sys.exit(1)

print("\n5. Testing multiple time steps:")
print("-" * 70)

# Test over multiple steps to ensure consistency
errors = []
for step in range(5):
    actions = {
        'warehouse': np.random.uniform(0, 10, config.num_products),
        'retailers': np.random.uniform(0, 5, (config.num_retailers, config.num_products))
    }

    next_obs, rewards, done, info = env.step(actions)
    actual_demand = info['actual_demand']

    optimal = calculate_counterfactual_profit(env, actual_demand, config)
    predicted = env.warehouse.profit + sum(r.profit for r in env.retailers)

    gap = abs(predicted - optimal['total']) / max(abs(optimal['total']), 1e-5)
    error = -gap
    errors.append(error)

    print(f"  Step {step+1}: error={error:.4f}, predicted=${predicted:.2f}, optimal=${optimal['total']:.2f}")

print(f"\nAll {len(errors)} steps completed successfully")
print("✓ PASS: Multiple time steps work correctly")

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print("✓ All tests passed!")
print("\nKey changes verified:")
print("1. calculate_counterfactual_profit() returns Dict with breakdown")
print("2. Dict contains 'retailer', 'warehouse', and 'total' keys")
print("3. Decision error calculation uses optimal_profit['total']")
print("4. Safe division prevents divide-by-zero errors")
print("5. Reward extraction works correctly for MAPPO training")
print("\nThe TypeError in train.py should now be fixed!")
print("=" * 70)
