import os
import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def compute_ofi(buy_orders: float, sell_orders: float)-> float:
    denom= buy_orders + sell_orders

    #division by zero error
    if abs(denom) < 1e-7:
        return 0.0
    return float((buy_orders - sell_orders) / denom)

def classify_signal(ofi: float, threshold: float = 0.05) ->str:
    if ofi > threshold:
        return "BUY"
    elif ofi < -threshold:
        return "SELL"
    else: 
        return "HOLD"



def run_cpu_reference():
    # laoding the deterministic fixture 
    fixture_path= os.path.join("fixtures", "model_and_trace.npz")

    if not os.path.exists(fixture_path):
        raise FileNotFoundError(f"Fixture missing: {fixture_path}: Run the file first")

    data= np.load(fixture_path)
    c_var = data["c_var"]      # Shape: (2,)
    A_var = data["A_var"]      # Shape: (2, 4)
    W1, b1 = data["W1"], data["b1"]  # W1: (32, 2), b1: (32,)
    W2, b2 = data["W2"], data["b2"]  # W2: (16, 32), b2: (16,)
    W3, b3 = data["W3"], data["b3"]  # W3: (2, 16), b3: (2,)
    X_stream = data["X_stream"]      # Shape: (100, 2)
    p = int(data["p"])               # p = 2
    M = int(data["M"])               # M = 2
    N_ticks = len(X_stream)

    #initialise output arrays
    y_var_trace = np.zeros((N_ticks, M), dtype=np.float32)
    residual_trace = np.zeros((N_ticks, M), dtype=np.float32)
    y_res_trace = np.zeros((N_ticks, M), dtype=np.float32)
    y_final_trace = np.zeros((N_ticks, M), dtype=np.float32)
    ofi_trace = np.zeros(N_ticks, dtype=np.float32)
    signals = []

    #initialise lag variable

    state_buffer= np.zeros(p*M , dtype= np.float32)
    state_buffer[:M]= X_stream[0]
    state_buffer[M: 2*M]= X_stream[0]



#tick by tick processing

    for t in range(N_ticks):
        x_t= X_stream[t]    # Current incoming market tick: [Buy_Orders, Sell_Orders]

        # Formula: ŷ_VAR = A @ [Y_{t-1}, Y_{t-2}]^T + c
        y_var = A_var @ state_buffer + c_var
        y_var_trace[t] = y_var


        #Residual Formula: ε̂_t = Y_t - ŷ_VAR

        eps_t = x_t - y_var
        residual_trace[t] = eps_t

        #Residual FNN Forward Pass

        h1 = relu(W1 @ eps_t + b1)      # Layer 1: (32, 2) @ (2,) + (32,) -> (32,)
        h2 = relu(W2 @ h1 + b2)          # Layer 2: (16, 32) @ (32,) + (16,) -> (16,)
        y_res = W3 @ h2 + b3             # Output: (2, 16) @ (16,) + (2,) -> (2,)
        y_res_trace[t] = y_res


        # Recombination and trading signal generation

        y_final = y_var + y_res
        y_final_trace[t] = y_final

        ofi = compute_ofi(y_final[0], y_final[1])
        ofi_trace[t] = ofi
        sig = classify_signal(ofi, threshold=0.05)
        signals.append(sig)

        #shift the register state update

        state_buffer[M: 2*M]= state_buffer[:M]
        state_buffer[:M]= x_t


        golden_path= os.path.join("fixtures", "golden_trace.npz")

        np.savez_compressed(
            golden_path,
            inputs= X_stream,
            y_var= y_var_trace,
            residuals= residual_trace,
            y_res= y_res_trace,
            y_final= y_final_trace,
            ofi= ofi_trace,
            signals= np.array(signals)
        )
        

        print(f"Path saved at : {golden_path}")
        print("\nFirst 5 Processed Ticks:")
    for i in range(5):
        actual_buy = float(X_stream[i, 0])
        actual_sell = float(X_stream[i, 1])
        pred_buy = float(y_final_trace[i, 0])
        pred_sell = float(y_final_trace[i, 1])
        ofi_val = float(ofi_trace[i])
        sig = str(signals[i])
        print(f"  Tick {i:02d} | Actual: ({actual_buy:.1f}, {actual_sell:.1f}) | "
              f"Final Pred: ({pred_buy:.1f}, {pred_sell:.1f}) | "
              f"OFI: {ofi_val:+.3f} -> Signal: {sig}")

        
if __name__ == "__main__":
    run_cpu_reference()
       












