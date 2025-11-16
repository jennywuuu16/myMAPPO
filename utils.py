import numpy as np

# -----------------------
# 辅助仿真与搜索函数
# -----------------------

def simulate_warehouse_cycle_profit(start_inventory, pending_orders, supplier_prices,
                                    wholesale_prices, daily_retailer_demands, Q_order, config):
    """
    Simulate the warehouse over the cycle given candidate Q_order.
    pending_orders: list of tuples (day_issued, qty_array) as in env
    daily_retailer_demands: np.array shape (cycle_len, num_products) aggregated retailer demand per day
    Q_order: np.array shape (num_products,) - the order placed at cycle start (continuous allowed)
    Returns: (total_profit, daily_profits_list, capacity_violated_flag)
    """
    cycle_len = daily_retailer_demands.shape[0]
    P = daily_retailer_demands.shape[1]

    # clone state
    inv = start_inventory.copy().astype(float)
    # build a local pipeline copy: compute arrivals indexed by simulation day offset
    # pending_orders entries are (day_issued, qty_array), and arrival happens when (sim_day_counter - day_issued) >= lead_time
    # For simulation we'll treat day_issued relative to current global day; we'll map arrivals to offsets.
    # To simplify we create arrival_map: offset_day -> sum of qty arrays arriving at that offset
    arrival_map = {}
    # assume current global day_counter provided via pending_orders day_issued; but in this simulation we consider offsets relative to current day
    # map any pending with day_issued >=0 appropriately: arrival_offset = (day_issued_offset) such that arrival occurs during our cycle
    # We don't know global day_counter here, so we interpret pending_orders as they will arrive with some offsets; instead, to be robust:
    # We'll convert pending_orders to offsets by: if pending was (day_issued, qty) and env.day_counter when called is D, arrival_offset = day_issued + lead_time - D
    # But here, caller should pass pending_orders adjusted relative to current day: to keep general we accept pending_orders as list of tuples where qty arrival day offsets are unknown.
    # To avoid complexity, we expect pending_orders passed in already in the form of (day_issued, qty) using env.warehouse.day_counter semantics.
    # So here we build arrival_map using same semantics: arrival_day_global = day_issued + config.lead_time; we will compute arrival offset = arrival_day_global - current_day_global.
    # But we don't have current_day_global here; to produce correct offsets, caller of this function (compute_warehouse_counterfactual_cycle) will translate pending_orders to offsets for us.
    # For safety, interpret pending_orders as list of (offset, qty) where offset=days until arrival (0 means arrives today). If not, compute in caller.
    arrival_map = {}
    for off, qty in pending_orders:
        # off is interpreted as offset days until arrival (0 means arrives next simulated day 0)
        off_int = int(off)
        if off_int not in arrival_map:
            arrival_map[off_int] = np.zeros(P, dtype=float)
        arrival_map[off_int] += np.array(qty, dtype=float)

    # we assume Q_order will arrive after lead_time offset L (i.e. offset = L-1 arrives before day 0? we'll define Q arrives at offset = config.lead_time - 1 and we consider arrival at morning)
    L = config.lead_time
    # treat Q arrival offset as L (i.e., arrives at start of day L of the simulation loop)
    q_arrival_offset = L
    if q_arrival_offset not in arrival_map:
        arrival_map[q_arrival_offset] = np.zeros(P, dtype=float)
    arrival_map[q_arrival_offset] += Q_order.astype(float)

    daily_profits = []
    total_profit = 0.0

    # For capacity check: if at any time inventory after applying arrivals exceeds capacity -> capacity violation
    capacity_violated = False

    # simulate day by day
    for day in range(cycle_len):
        # arrivals that occur at this day offset
        if day in arrival_map:
            inv += arrival_map[day]
            # environment clamps inventory at capacity when actually adding; here we enforce capacity constraint
            if np.any(inv > config.warehouse_capacity + 1e-8):
                # capacity would be exceeded at arrival -> mark violation and stop simulation
                capacity_violated = True
                # We still can choose to clip, but user requested capacity must be satisfied -> treat violation as infeasible
                break

        # fulfill demand (lost-sales)
        demand = daily_retailer_demands[day]
        sales = np.minimum(inv, demand)
        inv -= sales
        stockout = np.maximum(demand - sales, 0.0)

        revenue = np.sum(sales * wholesale_prices)
        holding_cost = np.sum(inv * config.holding_cost_warehouse)
        stockout_cost = np.sum(stockout * config.stockout_cost_warehouse)
        ordering_cost = 0.0
        # attribute supplier cost at time of order (day 0)
        if day == 0:
            ordering_cost = np.sum(Q_order * supplier_prices)

        daily_profit = revenue - ordering_cost - holding_cost - stockout_cost
        daily_profits.append(daily_profit)
        total_profit += daily_profit

    return total_profit, daily_profits, capacity_violated


def compute_warehouse_counterfactual_cycle(env, demands_4d, config,
                                           coarse_ratio=10, refine_range=5, refine_step=0.1):
    """
    High-precision counterfactual solver for warehouse over the 4-day cycle.
    - demands_4d: shape (cycle_len, num_products)
    Returns: final_profit, best_Q (num_products,), daily_profits (list)
    """
    P = config.num_products
    cycle_len = demands_4d.shape[0]

    start_inventory = env.warehouse.inventory.copy().astype(float)
    # Build pending_orders offsets relative to "today": convert env.pending_orders (day_issued, qty) to offsets
    # arrival_offset = (day_issued + config.lead_time) - current_day_global
    # current_day_global is env.warehouse.day_counter (we assume caller calls this after the day's env.day_counter update)
    current_global_day = env.warehouse.day_counter
    pending_offsets = []
    for (day_issued, qty) in getattr(env.warehouse, "pending_orders", []):
        arrival_global = day_issued + config.lead_time
        offset = int(arrival_global - current_global_day)
        # only keep arrivals that will happen within cycle window (and also keep those beyond; they will be outside arrival_map)
        pending_offsets.append((offset, np.array(qty, dtype=float)))

    # wholesale & supplier price vectors
    wholesale_prices = env.wholesale_prices if env.wholesale_prices is not None else np.ones(P) * (env.retail_prices.mean() * 0.6 if env.retail_prices is not None else 6.0)
    supplier_prices = env.supplier_prices if env.supplier_prices is not None else wholesale_prices * 0.5

    # compute total_needed baseline
    total_needed = np.sum(demands_4d, axis=0)  # (P,)
    # max_Q: conservative bound: cannot order more than capacity remaining after accounting for arrivals and expected shipments
    # compute arrivals within cycle (from pending_offsets)
    future_arrivals = np.zeros(P, dtype=float)
    for off, qty in pending_offsets:
        if 0 <= off <= cycle_len + config.lead_time + 5:
            future_arrivals += qty
    capacity_remaining = config.warehouse_capacity - start_inventory - future_arrivals
    # allow some buffer; we will enforce capacity violation check in simulate
    # set max_Q to max(total_needed - start_inventory, 0) + buffer
    max_Q = np.maximum(total_needed - start_inventory, 0).astype(int) + 10
    # but limit by capacity_remaining (don't allow ordering that will definitely exceed capacity at arrival)
    # for safety, set an upper bound per product:
    max_Q = np.minimum(max_Q, np.maximum(config.warehouse_capacity - start_inventory, 0).astype(int) + 10)

    # coarse search per-product (independent)
    best_Q = np.zeros(P, dtype=float)
    best_profit = -1e18

    # coarse step based on max across products
    max_overall = max(1, int(np.max(max_Q)))
    coarse_step = max(1, max(1, int(np.ceil(max_overall / max(1, coarse_ratio)))))

    for p in range(P):
        best_q_p = 0.0
        best_profit_p = -1e18
        max_qp = int(max(0, max_Q[p]))
        # coarse loop
        q_vals = list(range(0, max_qp + 1, coarse_step))
        if q_vals[-1] != max_qp:
            q_vals.append(max_qp)
        for q in q_vals:
            cand_Q = best_Q.copy()
            cand_Q[p] = q
            tot_profit, daily_profs, violated = simulate_warehouse_cycle_profit(
                start_inventory=start_inventory,
                pending_orders=pending_offsets,
                supplier_prices=supplier_prices,
                wholesale_prices=wholesale_prices,
                daily_retailer_demands=demands_4d,
                Q_order=cand_Q,
                config=config
            )
            if violated:
                continue
            if tot_profit > best_profit_p:
                best_profit_p = tot_profit
                best_q_p = q
        best_Q[p] = best_q_p

    # refine: joint local search around best_Q
    # build search ranges
    search_grids = []
    for p in range(P):
        low = max(0, best_Q[p] - refine_range)
        high = best_Q[p] + refine_range
        # create float grid with refine_step
        grid = np.arange(low, high + 1e-9, refine_step)
        search_grids.append(grid)

    # Cartesian product joint search (for small P). If P > 3, this may explode; but user said small dim.
    from itertools import product
    best_joint_Q = best_Q.copy()
    best_joint_profit = -1e18
    for combo in product(*search_grids):
        cand_Q = np.array(combo, dtype=float)
        # quick capacity pre-check: if start_inventory + future_arrivals + cand_Q > capacity at arrival -> may violate; we rely on simulate to check
        tot_profit, daily_profs, violated = simulate_warehouse_cycle_profit(
            start_inventory=start_inventory,
            pending_orders=pending_offsets,
            supplier_prices=supplier_prices,
            wholesale_prices=wholesale_prices,
            daily_retailer_demands=demands_4d,
            Q_order=cand_Q,
            config=config
        )
        if violated:
            continue
        if tot_profit > best_joint_profit:
            best_joint_profit = tot_profit
            best_joint_Q = cand_Q.copy()

    # final simulate
    final_profit, final_daily, violated = simulate_warehouse_cycle_profit(
        start_inventory=start_inventory,
        pending_orders=pending_offsets,
        supplier_prices=supplier_prices,
        wholesale_prices=wholesale_prices,
        daily_retailer_demands=demands_4d,
        Q_order=best_joint_Q,
        config=config
    )

    if violated:
        # if best caused violation (rare), return a fallback: no-order profit
        no_order_profit, no_daily, _ = simulate_warehouse_cycle_profit(
            start_inventory=start_inventory,
            pending_orders=pending_offsets,
            supplier_prices=supplier_prices,
            wholesale_prices=wholesale_prices,
            daily_retailer_demands=demands_4d,
            Q_order=np.zeros(P),
            config=config
        )
        return no_order_profit, np.zeros(P), no_daily

    return final_profit, best_joint_Q, final_daily


def distribute_cycle_profit_to_days(cycle_profit, demands_4d):
    """
    Distribute cycle_profit to each day proportionally to day demand (sum across products).
    Returns list length cycle_len of allocated floats.
    """
    day_totals = np.sum(demands_4d, axis=1)  # shape (cycle_len,)
    total = float(np.sum(day_totals))
    cycle_len = demands_4d.shape[0]
    if total <= 0:
        return [cycle_profit / cycle_len] * cycle_len
    else:
        return list((day_totals / total) * cycle_profit)

# -----------------------
# 主函数：替换 calculate_counterfactual_profit
# -----------------------

def calculate_counterfactual_profit(env, actual_demand: np.ndarray, config):
    """
    Compute counterfactual (theoretical optimal) profit.
    Returns a dict with:
        'total_daily_opt_profit' : float (sum of retailer daily optimal profits + warehouse allocated today if cycle end)
        'retailer_profits' : np.array (num_retailers,) retailer optimal profits for today
        'warehouse_cycle' : dict or None; if computed includes keys:
            'cycle_profit', 'allocations'(list), 'best_Q', 'daily_profits'
    """
    P = config.num_products
    R = config.num_retailers

    # Prices
    retail_prices = env.retail_prices if env.retail_prices is not None else np.ones(P) * 10.0
    wholesale_prices = env.wholesale_prices if env.wholesale_prices is not None else retail_prices * 0.6
    supplier_prices = env.supplier_prices if env.supplier_prices is not None else retail_prices * 0.3

    result = {
        'retailer_profits': np.zeros(R, dtype=float),
        'warehouse_cycle': None,
        'total_daily_opt_profit': 0.0
    }

    # ----------------
    # Retailer optimal (daily) - can be computed with perfect knowledge of today's demand
    # actual_demand shape: (num_retailers, num_products)
    # ----------------
    for i, retailer in enumerate(env.retailers):
        demand_i = actual_demand[i].astype(float)
        current_inventory = retailer.inventory.copy().astype(float)
        # optimal to exactly meet today's demand (perfect foresight)
        optimal_order = np.maximum(demand_i - current_inventory, 0.0)
        # capacity cap: inventory + order <= retailer.capacity
        cap_space = retailer.capacity - current_inventory
        optimal_order = np.minimum(optimal_order, np.maximum(cap_space, 0.0))

        sales = np.minimum(current_inventory + optimal_order, demand_i)
        stockout = np.maximum(demand_i - sales, 0.0)

        revenue = np.sum(sales * retail_prices)
        ordering_cost = np.sum(optimal_order * config.ordering_cost_retailer)
        holding_cost = np.sum((current_inventory + optimal_order - sales) * config.holding_cost_retailer)
        stockout_cost = np.sum(stockout * config.stockout_cost_retailer)

        daily_opt_profit = revenue - ordering_cost - holding_cost - stockout_cost
        result['retailer_profits'][i] = daily_opt_profit
        result['total_daily_opt_profit'] += daily_opt_profit

    # ----------------
    # Warehouse optimal: only compute when a cycle completes (we need the 4-day true demands)
    # We'll determine current global day index using env.current_demand_idx (which points to next to-read index),
    # so the last observed day index is current_idx - 1
    # ----------------
    # compute current day index oracle-safe
    try:
        current_idx = env.current_demand_idx - 1  # last realized day index
    except Exception:
        current_idx = None

    cycle_len = config.warehouse_order_cycle
    do_warehouse_cycle = False
    if current_idx is not None:
        # when (current_idx + 1) % cycle_len == 0 -> cycle ended at this day (0-index)
        if ((current_idx + 1) % cycle_len) == 0:
            do_warehouse_cycle = True

    if do_warehouse_cycle:
        # retrieve last `cycle_len` days demands from env.demand_sequence if available
        demands_4d = []
        for d_offset in range(cycle_len):
            idx = current_idx - (cycle_len - 1) + d_offset
            # build per-retailer demand for this day
            day_dem = np.zeros((R, P), dtype=float)
            if hasattr(env, "demand_sequence") and env.demand_sequence is not None and 'demand' in env.demand_sequence:
                for r in range(R):
                    store_key = f'store_{r}'
                    store_series = env.demand_sequence['demand'].get(store_key, [])
                    if 0 <= idx < len(store_series):
                        day_dem[r, :] = np.array(store_series[idx], dtype=float)
                    else:
                        day_dem[r, :] = np.zeros(P)
            else:
                # fallback: replicate today's actual_demand
                day_dem[:, :] = actual_demand.copy()

            # sum across retailers -> get aggregated demand per product for that day
            agg = np.sum(day_dem, axis=0)
            demands_4d.append(agg)
        demands_4d = np.vstack(demands_4d)  # shape (cycle_len, P)

        # compute high-precision optimal for warehouse cycle
        cycle_profit, best_Q, daily_profits = compute_warehouse_counterfactual_cycle(env, demands_4d, config)

        # allocate cycle profit back to days proportionally
        allocations = distribute_cycle_profit_to_days(cycle_profit, demands_4d)  # list length cycle_len

        # find which day in cycle is 'today' index
        day_in_cycle = cycle_len - 1  # since we assembled from oldest->today, today is last element
        allocated_today = allocations[day_in_cycle]

        # add to result
        result['warehouse_cycle'] = {
            'cycle_profit': float(cycle_profit),
            'allocations': allocations,
            'best_Q': best_Q.tolist(),
            'daily_profits': daily_profits
        }
        result['total_daily_opt_profit'] += allocated_today
    else:
        # No warehouse cycle computed today; return retailers only
        result['warehouse_cycle'] = None

    return result
