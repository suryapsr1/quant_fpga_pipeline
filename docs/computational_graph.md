# Computational Graph Specification: Hybrid VAR(2) + Residual FNN Pipeline


## 1. End-to-End Pipeline Architecture


Incoming Tick: x_t = [Buy_Orders, Sell_Orders]^T  ∈ ℝ^2
       │
       ├──────────────────────────────────────────┐
       ▼                                          │
[ 4-Element Shift Register ]                      │
  s_t = [x_{t-1}, x_{t-2}] ∈ ℝ^4                  │
       │                                          │
       ▼                                          │
[ Stage 1: Linear VAR(2) GEMV ]                   │
  y_VAR = A · s_t + c   ∈ ℝ^2                     │
       │                                          │
       ├──────────────────────────────────────────┘
       ▼
[ Element-Wise Subtraction ]
  ε_t = x_t - y_VAR     ∈ ℝ^2  (Market shock / residual)
       │
       ▼
[ Stage 2: Residual FNN (32-16-2) ]
  h_1   = ReLU(W_1 · ε_t + b_1)     ∈ ℝ^32
  h_2   = ReLU(W_2 · h_1 + b_2)     ∈ ℝ^16
  y_res = W_3 · h_2 + b_3           ∈ ℝ^2
       │
       ▼
[ Recombination & Signal Logic ]
  y_final = y_VAR + y_res           ∈ ℝ^2
  OFI     = (y_final[0] - y_final[1]) / (y_final[0] + y_final[1])
  Signal  = BUY (OFI > 0.05) | SELL (OFI < -0.05) | HOLD