# myMAPPO
test
## Overview
This implementation of MAPPO (Multi-Agent Proximal Policy Optimization) for supply chain management now supports loading historical sales data from CSV files.
Specifically, A PredictorAgent is used for guiding the prediction model, a minimized deicison error is expected rathere than simple prediction error (such as MSE)
We want the Warehouse and Retailers Agents make good decisions based on the PredictionAgent Action.The alogrithm basic framework is as follows:


PredictorAgent → predict future demands → action_pred 形状 [warehouse order period, num_products]
       ↓next
RetailerAgent：只应读取“下一天预测值”(because retailers replenish everyday)
       ↓next
WarehouseAgent：应读取完整的未来预测值(covering its order period)
