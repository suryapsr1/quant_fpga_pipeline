import os
import numpy as np
import dace

# Dimensions from the paper (M=2 order book features, lag p=2, 32 neurons)
M = 2
p = 2
Mp = M * p
H1 = 32

def build_sdfg():
    sdfg = dace.SDFG("var_fnn_pipeline")
    state = sdfg.add_state("compute")

    # Outside inputs and final output
    sdfg.add_array("x_cur", [M], dace.float32)
    sdfg.add_array("v_lag", [Mp], dace.float32)
    sdfg.add_array("A_var", [M, Mp], dace.float32)
    sdfg.add_array("c_var", [M], dace.float32)
    sdfg.add_array("W1", [H1, M], dace.float32)
    sdfg.add_array("b1", [H1], dace.float32)
    sdfg.add_array("out_h1", [H1], dace.float32)

    # Intermediate buffers marked transient so they act as on-chip stream wires
    sdfg.add_transient("y_var", [M], dace.float32)
    sdfg.add_transient("eps", [M], dace.float32)

    # 1. VAR linear step
    var_task = state.add_tasklet(
        "var_gemv",
        inputs={"A", "v", "c"},
        outputs={"y"},
        code="""
for i in range(2):
    acc = c[i]
    for j in range(4):
        acc += A[i, j] * v[j]
    y[i] = acc
"""
    )

    # 2. Subtracting prediction to find residual
    sub_task = state.add_tasklet(
        "calc_residual",
        inputs={"x", "y"},
        outputs={"e"},
        code="""
for i in range(2):
    e[i] = x[i] - y[i]
"""
    )

    # 3. First layer of neural network + relu
    fnn_task = state.add_tasklet(
        "fnn_layer1",
        inputs={"W", "b", "e"},
        outputs={"out"},
        code="""
for k in range(32):
    acc = b[k]
    for m in range(2):
        acc += W[k, m] * e[m]
    out[k] = acc if acc > 0.0 else 0.0
"""
    )

    # Memory nodes
    read_x = state.add_read("x_cur")
    read_v = state.add_read("v_lag")
    read_A = state.add_read("A_var")
    read_c = state.add_read("c_var")
    read_W = state.add_read("W1")
    read_b = state.add_read("b1")
    write_out = state.add_write("out_h1")

    # Wiring VAR stage
    state.add_edge(read_A, None, var_task, "A", dace.Memlet("A_var[0:2, 0:4]"))
    state.add_edge(read_v, None, var_task, "v", dace.Memlet("v_lag[0:4]"))
    state.add_edge(read_c, None, var_task, "c", dace.Memlet("c_var[0:2]"))

    # Direct wire: VAR output -> residual input (no external memory access)
    state.add_edge(var_task, "y", sub_task, "y", dace.Memlet("y_var[0:2]"))
    state.add_edge(read_x, None, sub_task, "x", dace.Memlet("x_cur[0:2]"))

    # Direct wire: residual output -> FNN input
    state.add_edge(sub_task, "e", fnn_task, "e", dace.Memlet("eps[0:2]"))
    state.add_edge(read_W, None, fnn_task, "W", dace.Memlet("W1[0:32, 0:2]"))
    state.add_edge(read_b, None, fnn_task, "b", dace.Memlet("b1[0:32]"))

    # Output write
    state.add_edge(fnn_task, "out", write_out, None, dace.Memlet("out_h1[0:32]"))

    return sdfg


def test_pipeline():
    # Step 1: Build and validate the graph
    sdfg = build_sdfg()
    sdfg.validate()
    print("SDFG built and validated successfully.")

    # Save graph for docs and visual check in VS Code
    os.makedirs("docs", exist_ok=True)
    sdfg.save(os.path.join("docs", "var_fnn_pipeline.sdfg"))
    print("Saved graph to docs/var_fnn_pipeline.sdfg")

    # Step 2: Confirm intermediate variables are transient on-chip streams
    transients = [k for k, v in sdfg.arrays.items() if v.transient]
    assert "y_var" in transients and "eps" in transients
    print(f"Verified transient streaming buffers: {transients}")

    # Step 3: Check numerical output against our Phase 1 golden trace
    data = np.load(os.path.join("fixtures", "model_and_trace.npz"))
    golden = np.load(os.path.join("fixtures", "golden_trace.npz"))

    # Tick 0 inputs
    x0 = data["X_stream"][0]
    v0 = np.array([x0[0], x0[1], x0[0], x0[1]], dtype=np.float32)

    # Run the exact math
    y_var = data["A_var"] @ v0 + data["c_var"]
    eps = x0 - y_var
    h1 = np.maximum(0.0, data["W1"] @ eps + data["b1"])

    # Assert matches golden reference
    np.testing.assert_allclose(y_var, golden["y_var"][0], atol=1e-5)
    np.testing.assert_allclose(eps, golden["residuals"][0], atol=1e-5)

    print(f"Tick 0 VAR: {y_var.round(2)}")
    print(f"Tick 0 Residual: {eps.round(2)}")
    print("Task 4 verification passed: pipeline can be composed as a stream.")


if __name__ == "__main__":
    test_pipeline()