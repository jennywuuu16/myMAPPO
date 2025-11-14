# Profit Calculation Analysis

## Current Problem

### Retailer (INCORRECT):
```python
revenue = sales * retail_prices  # ~15 per unit
ordering_cost = order * 0.3      # WRONG: Should be ~9 (wholesale price)
holding_cost = inventory * 0.1
stockout_cost = stockout * 2.0
profit = revenue - ordering_cost - holding_cost - stockout_cost
```

**Issue**: Retailer pays 0.3 per unit instead of 9 (wholesale price)
- Apparent margin: 15 - 0.3 = 14.7 (98%!) → Agents over-order
- Actual margin should be: 15 - 9 = 6 (40%)

### Warehouse (HAS TWO COSTS):
```python
revenue = deliveries * wholesale_prices  # ~9 per unit
ordering_cost = orders * 0.1             # Transaction fee?
supplier_cost = orders * supplier_prices # ~4.5 (payment to suppliers)
profit = revenue - ordering_cost - holding_cost - supplier_cost - stockout_cost
```

**Question**: Should warehouse have BOTH ordering_cost (0.1) AND supplier_cost (4.5)?
- If ordering_cost is transaction fee: Keep both
- If ordering_cost should BE the payment: Remove ordering_cost, keep supplier_cost

## Expected Values (for products price ~14-15, sales 1-8/day)

### Retailer:
- Retail price: 15
- Wholesale price: 9 (60%)
- Margin per unit: 6
- Daily sales: ~5 units
- Gross profit: 5 × 6 = 30
- Costs: holding (~0.5) + stockout (varies)
- **Expected net profit: 20-30 per day**

### Warehouse:
- Wholesale price: 9
- Supplier price: 4.5 (30% of retail)
- Margin per unit: 4.5
- Daily deliveries: ~5 units
- Gross profit: 5 × 4.5 = 22.5
- Costs: holding (~2.5) + stockout (varies) + transaction (0.5)
- **Expected net profit: 15-20 per day**

## Recommended Fix

### Option 1: ordering_cost = payment (user's description)
**Retailer:**
```python
ordering_cost = order * wholesale_prices  # Use wholesale prices, not config value
profit = revenue - ordering_cost - holding_cost - stockout_cost
```

**Warehouse:**
```python
# Remove ordering_cost, keep supplier_cost as the payment
profit = revenue - supplier_cost - holding_cost - stockout_cost
```

### Option 2: Separate transaction fee + payment
**Retailer:**
```python
ordering_fee = order * 0.3  # Small transaction cost
wholesale_payment = received * wholesale_prices  # Actual payment
profit = revenue - wholesale_payment - ordering_fee - holding_cost - stockout_cost
```

**Warehouse:**
```python
ordering_fee = order * 0.1
supplier_payment = order * supplier_prices
profit = revenue - supplier_payment - ordering_fee - holding_cost - stockout_cost
```

## User's Specification
"the retailer's payment to warehouse is the ordering cost"

This indicates **Option 1**: ordering_cost should BE the payment (no separate transaction fee).
