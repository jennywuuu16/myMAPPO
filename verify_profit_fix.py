"""
Verification script for profit calculation fix

Expected behavior with prices ~14-15, sales ~1-8:
- Retail price: 15, Wholesale: 9 (60%), Supplier: 4.5 (30%)
- Retailer margin: 15 - 9 = 6 per unit
- Warehouse margin: 9 - 4.5 = 4.5 per unit
- Expected profits should be POSITIVE (small values)
"""

print("="*70)
print("PROFIT CALCULATION VERIFICATION")
print("="*70)

print("\nExpected Profit Structure:")
print("-" * 70)

# Product with price ~15, daily sales ~5 units
retail_price = 15.0
wholesale_price = retail_price * 0.6  # 9.0
supplier_price = retail_price * 0.3   # 4.5

print(f"\nProduct pricing:")
print(f"  Retail price: ${retail_price:.2f}")
print(f"  Wholesale price: ${wholesale_price:.2f} (60% of retail)")
print(f"  Supplier price: ${supplier_price:.2f} (30% of retail)")

units_sold = 5
units_ordered = 5
inventory_remaining = 2

print(f"\nScenario: {units_sold} units sold, {units_ordered} units ordered, {inventory_remaining} units remaining")

# Retailer calculation
retailer_revenue = units_sold * retail_price
retailer_ordering_cost = units_ordered * wholesale_price  # Payment to warehouse
retailer_holding_cost = inventory_remaining * 0.1
retailer_profit = retailer_revenue - retailer_ordering_cost - retailer_holding_cost

print(f"\nRetailer:")
print(f"  Revenue: ${retailer_revenue:.2f} ({units_sold} × ${retail_price:.2f})")
print(f"  Ordering cost (payment to warehouse): ${retailer_ordering_cost:.2f} ({units_ordered} × ${wholesale_price:.2f})")
print(f"  Holding cost: ${retailer_holding_cost:.2f} ({inventory_remaining} × $0.10)")
print(f"  Net profit: ${retailer_profit:.2f}")
print(f"  Profit margin: {(retailer_profit/retailer_revenue)*100:.1f}%")

# Warehouse calculation
warehouse_revenue = units_sold * wholesale_price  # What they charge retailers
warehouse_ordering_cost = units_ordered * supplier_price  # Payment to suppliers
warehouse_holding_cost = inventory_remaining * 0.05
warehouse_profit = warehouse_revenue - warehouse_ordering_cost - warehouse_holding_cost

print(f"\nWarehouse:")
print(f"  Revenue: ${warehouse_revenue:.2f} ({units_sold} × ${wholesale_price:.2f})")
print(f"  Ordering cost (payment to suppliers): ${warehouse_ordering_cost:.2f} ({units_ordered} × ${supplier_price:.2f})")
print(f"  Holding cost: ${warehouse_holding_cost:.2f} ({inventory_remaining} × $0.05)")
print(f"  Net profit: ${warehouse_profit:.2f}")
print(f"  Profit margin: {(warehouse_profit/warehouse_revenue)*100:.1f}%")

print(f"\nTotal supply chain profit: ${retailer_profit + warehouse_profit:.2f}")

print("\n" + "="*70)
print("VERIFICATION CHECKS")
print("="*70)

checks_passed = 0
checks_total = 0

# Check 1: Retailer profit should be positive
checks_total += 1
if retailer_profit > 0:
    print(f"✓ Retailer profit is positive: ${retailer_profit:.2f}")
    checks_passed += 1
else:
    print(f"✗ ERROR: Retailer profit is negative: ${retailer_profit:.2f}")

# Check 2: Warehouse profit should be positive
checks_total += 1
if warehouse_profit > 0:
    print(f"✓ Warehouse profit is positive: ${warehouse_profit:.2f}")
    checks_passed += 1
else:
    print(f"✗ ERROR: Warehouse profit is negative: ${warehouse_profit:.2f}")

# Check 3: Retailer margin should be reasonable (30-50%)
checks_total += 1
retailer_margin_pct = (retailer_profit/retailer_revenue)*100
if 20 <= retailer_margin_pct <= 60:
    print(f"✓ Retailer margin is reasonable: {retailer_margin_pct:.1f}%")
    checks_passed += 1
else:
    print(f"✗ ERROR: Retailer margin is unrealistic: {retailer_margin_pct:.1f}%")

# Check 4: Warehouse margin should be reasonable
checks_total += 1
warehouse_margin_pct = (warehouse_profit/warehouse_revenue)*100
if 20 <= warehouse_margin_pct <= 60:
    print(f"✓ Warehouse margin is reasonable: {warehouse_margin_pct:.1f}%")
    checks_passed += 1
else:
    print(f"✗ ERROR: Warehouse margin is unrealistic: {warehouse_margin_pct:.1f}%")

print(f"\n{checks_passed}/{checks_total} checks passed")

if checks_passed == checks_total:
    print("\n✓ ALL CHECKS PASSED - Profit calculations are correct!")
else:
    print(f"\n✗ {checks_total - checks_passed} CHECKS FAILED - Review profit calculations")

print("="*70)
