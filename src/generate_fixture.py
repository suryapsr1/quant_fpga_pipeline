import os
import numpy as np

np.random.seed(42)

#From paper 

M = 2          # Dimensionality: [Buy Orders, Sell Orders]
p = 2          # Optimal lag order p = 2
H1 = 32        # FNN Hidden Layer 1 units
H2 = 16        # FNN Hidden Layer 2 units
N_ticks = 100  # Number of streaming ticks for test fixture

# Intercept c for [buy_orders, sell_orders] (mean baseline)
c_var = np.array([29.35, 28.50], dtype=np.float32)

#cocatenating A1 and A2 into a single matrix

A_var= np.array([[0.0037,  0.0111, -0.0005,  0.0167],
    [0.0105, -0.0021,  0.0142, -0.0011]
], dtype= np.float32)


#FNN parameters 

# layer 1: (32,2)
W1= (np.random.randn(H1,M)*np.sqrt(2.0/M)).astype(np.float32)
b1= np.zeros(H1, dtype= np.float32)

#layer 2: (16,32) 
W2= (np.random.randn(H2, H1)* np.sqrt(2.0/H1)).astype(np.float32)
b2= np.zeros(H2, dtype= np.float32)

#Output layer

W3= (np.random.randn(M, H2)* np.sqrt(2.0/H2)).astype(np.float32)
b3= np.zeros(M, dtype= np.float32)

#synthetic Buy and sell stream 
base = 30.0 + np.sin(np.linspace(0, 10, N_ticks))
buy_stream= np.maximum(1.0, base + np.random.normal(0,5, N_ticks)).astype(np.float32)
sell_stream= np.maximum(1.0, base + np.random.normal(0, 5, N_ticks)).astype(np.float32)
#cocatenate
X_stream= np.column_stack([buy_stream, sell_stream]).astype(np.float32)


#Saving the variables for consistent results

os.makedirs("fixtures", exist_ok=True)
out_path= os.path.join("fixtures", "model_and_trace.npz")
np.savez_compressed(
    out_path,
    c_var=c_var,
    A_var= A_var,
    W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3,
    X_stream= X_stream, 
    p=p, M=M, H1=H1, H2=H2
    )
print(f"fixture created at {out_path} with {N_ticks} ticks")