1. Linear VAR(2) Stage:
   - Matrix A has shape (2 rows × 4 columns) = 8 weights
   - Intercept vector c has 2 values          = 2 biases[cite: 2]
   Subtotal = 8 + 2 = 10 parameters[cite: 2]

2. Neural Network Stage (FNN 32-16-2):
   - Layer 1 (Dense 2 → 32):
     Weights W1: (32 × 2) = 64 weights[cite: 2]
     Biases   b1: (32)    = 32 biases[cite: 2]
     Layer 1 total        = 96 parameters[cite: 2]

   - Layer 2 (Dense 32 → 16):
     Weights W2: (16 × 32) = 512 weights[cite: 2]
     Biases   b2: (16)     =  16 biases[cite: 2]
     Layer 2 total         = 528 parameters[cite: 2]

   - Layer 3 (Dense 16 → 2):
     Weights W3: (2 × 16) = 32 weights[cite: 2]
     Biases   b3: (2)     =  2 biases[cite: 2]
     Layer 3 total        = 34 parameters[cite: 2]

Grand Total = 10 (VAR) + 96 (L1) + 528 (L2) + 34 (L3) = 668 parameters

Memory Size= 668 * 4 bytes= 2,672  bytes 

1. Standard FPGA boards have several megabytes (5 MB to 30+ MB) of fast on-chip Block RAM (BRAM) and UltraRAM.
2. The entire parameter set is only ~2.67 KB, which occupies less than 0.1% of available on-chip SRAM.
3. Weights are loaded once into on-chip BRAM at initialization. During live tick-by-tick streaming, each incoming tick is processed entirely through internal registers, DSPs, and FIFOs with no DDR round-trips needed.