import os
import numpy as np

def test_pipeline():

    golden_path = os.path.join("fixtures", "golden_trace.npz")

    assert os.path.exists(golden_path), f"Missing {golden_path}. Run cpu_reference.py first."

    data = np.load(golden_path)
    inputs    = data["inputs"]      # Shape: (100, 2)
    y_var     = data["y_var"]       # VAR linear predictions
    residuals = data["residuals"]   # Market shocks (epsilon)
    y_res     = data["y_res"]       # Neural network corrections
    y_final   = data["y_final"]     # Combined predictions
    ofi       = data["ofi"]         # Imbalance score
    signals   = data["signals"]     # BUY, SELL, or HOLD

    # Check 1: Did we get 100 ticks of 2 numbers each?
    assert inputs.shape == (100, 2), "Input shape is wrong!"
    assert y_final.shape == (100, 2), "Output shape is wrong!"

    # Check 2: Error must equal (Actual - VAR)
    assert np.allclose(residuals, inputs - y_var, atol=1e-5), "Residual equation failed!"

    # Check 3: Final must equal (VAR + Neural Net)
    assert np.allclose(y_final, y_var + y_res, atol=1e-5), "Recombination equation failed!"

    # Check 4: OFI fraction must stay between -1.0 and +1.0
    assert np.all(ofi >= -1.0) and np.all(ofi <= 1.0), "OFI went out of bounds!"
    # Check 5: Did the BUY / SELL / HOLD rules work correctly?
    for i in range(len(ofi)):
        score = ofi[i]
        sig = signals[i]
        if score > 0.05:
            assert sig == "BUY", f"Tick {i} should be BUY"
        elif score < -0.05:
            assert sig == "SELL", f"Tick {i} should be SELL"
        else:
            assert sig == "HOLD", f"Tick {i} should be HOLD"

    print("ALL 5 CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_pipeline()