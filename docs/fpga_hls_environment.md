I ran an automated discovery script on my development machine to check available compilers and FPGA vendor tools:


1. System Overview:
   OS Platform       : Windows 10 (64bit)
   Python Executable : C:\Users\achyu\quant_fpga_pipeline\.venv\Scripts\python.exe
   Python Version    : 3.11.9

2. Core C/C++ Compilers & Build Tools:
   - cmake      : FOUND (C:\Users\achyu\quant_fpga_pipeline\.venv\Scripts\cmake.EXE)
   - gcc        : FOUND (C:\msys64\ucrt64\bin\gcc.EXE)
   - g++        : FOUND (C:\msys64\ucrt64\bin\g++.EXE)
   - clang      : NOT FOUND
   - clang++    : NOT FOUND
   - cl         : NOT FOUND
   - ninja      : NOT FOUND
   - make       : NOT FOUND

3. FPGA & HLS Vendor Toolchains:
   [-] vitis_hls    : NOT DETECTED in PATH
   [-] vivado       : NOT DETECTED in PATH
   [-] vitis        : NOT DETECTED in PATH
   [-] quartus      : NOT DETECTED in PATH
   [-] aoc          : NOT DETECTED in PATH
   [-] dpcpp        : NOT DETECTED in PATH
   [-] yosys        : NOT DETECTED in PATH
   [-] verilator    : NOT DETECTED in PATH
   [-] iverilog     : NOT DETECTED in PATH

   [NOTE] No commercial FPGA synthesis suite found in active PATH.
          Standard university workflow: local prototyping via C-simulation,
          or remote execution on a lab/QuantHive Linux workstation.

4. DaCe Compiler Configuration:
   - DaCe Version    : 1.0.2
   - Error querying DaCe: 'default_compiler'



To run the full HLS and get the exact cycle count it has to run on LINUX machine with Vitis HLS. 


Model has only 668 parameters which takes about 2.67 MB. Even with fully unrolled matrix multiplications, our pipeline needs fewer than 150 DSP.

Very well with-in the FPGA hardware limits.