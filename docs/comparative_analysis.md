# Task 6: Comparative Analysis with Existing FPGA-HFT Architectures


## 1. Baseline 1: Traditional Rule-Based FPGA-HFT Systems
Most production FPGA systems deployed in trading colocation centers (such as the architecture described by He et al.) focus on network packet parsing and book reconstruction
* What they do: They ingest raw exchange feeds (e.g., NASDAQ ITCH, binary FIX), parse packets at line rate, and maintain the Limit Order Book (LOB). They use specialized hardware structures like fixed-tick arrays, clipped reduction trees, and bitonic sorting networks to extract the top-5 price levels.
* Decision logic: Their trading decisions rely on simple, hardcoded arithmetic rules—such as tracking bid-ask spreads, moving average crosses, or calculating basic micro-price thresholds.
* Latency: Extremely low and deterministic, completing updates in 130 to 300 nanoseconds
* Limitation: These architectures do not run predictive statistical models or capture non-linear market shocks. They react to current book states rather than forecasting short-term order flow drift.



## 2. Baseline 2: Monolithic Deep Learning on Hardware (e.g., DeepLOB)
Recent machine learning literature applies deep neural networks directly to high-frequency market data:
* What they do: Models like DeepLOB use deep convolutional networks, spatial feature extractors, and LSTM units across multi-level order books to forecast price directions.
* Hardware limitations:
  1. Parameter volume: Deep models typically require 60,000 to over 500,000 parameters. This footprint exceeds the fast on-chip register/LUT capacity of an FPGA, forcing parameters into external DDR/HBM memory and incurring long off-chip fetch penalties.
  2. Batching penalty: Deep learning accelerators (especially GPUs) achieve high throughput only by batching multiple inputs together. Live trading requires processing single events sequentially (batch size = 1) at the moment a tick arrives.
  3. Latency profile: Inference times range from tens of microseconds to milliseconds, which is too slow for tick-level order book competition.



## 3. How Our Hybrid Econometric-Plus-Neural Pipeline Differs
The pipeline studied here (Rahman & Upadhye, 2024) bridges the gap between rule-based speed and neural network expressiveness through three architectural differences:

### A. Statistical Decomposition into Linear and Residual Stages
* Stage 1 (VAR): A Vector Auto Regression model captures the dominant autoregressive drift of incoming Buy and Sell orders. Because it only has 10 parameters (A matrix of 2x4 and c vector of 2), it requires minimal math.
* Stage 2 (Residual FNN): The neural network is not forced to learn the whole market process from scratch. It receives only the residual prediction error (eps = actual_tick - y_VAR), modeling unexpected market shocks. Because its scope is restricted to residuals, the network remains compact (32-16-2 topology, 658 parameters).

### B. Microscopic Parameter Footprint (Zero Off-Chip RAM)
* The entire pipeline requires only 668 float32 parameters (~2.67 KB).
* Even on compact FPGAs, internal Block RAM (BRAM) provides several megabytes of storage. The entire model uses less than 0.1% of on-chip memory.
* Parameters are loaded once at startup into local BRAM/registers. Live execution never touches off-chip DDR or HBM, removing bus contention and memory jitter.

### C. Fused Streaming Pipeline via Data-Centric SDFG Composition
* In standard CPU/GPU software, running a hybrid model forces intermediate writes to system memory: calculating y_VAR, saving it, reading it back to compute the residual, and writing the residual to memory before calling the neural network.
* In our spatial design (formalized via DaCe SDFG), intermediate buffers (y_var and eps) are declared as transient streaming channels.
* The VAR multiplier connects directly to the residual subtractor, which connects directly to the first FNN layer through on-chip FIFO registers. This eliminates inter-stage data movement, allowing the hybrid model to execute with deterministic latency.