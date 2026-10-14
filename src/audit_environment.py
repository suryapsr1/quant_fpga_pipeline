"""
src/audit_environment.py
Task 5: Automated environment discovery to identify available C++/HLS compilers,
FPGA toolchains (AMD Vivado/Vitis, Intel Quartus), and simulation engines.
"""

import os
import sys
import shutil
import platform
import subprocess


# 1. System and Platform Summary

def inspect_system():
    print("=" * 60)
    print("TASK 5: FPGA & HLS ENVIRONMENT AUDIT")
    print("=" * 60)
    print("\n1. System Overview:")
    print(f"   OS Platform       : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"   Python Executable : {sys.executable}")
    print(f"   Python Version    : {platform.python_version()}")


# 2. General C/C++ Compilers and Build Tools

def check_build_tools():
    print("\n2. Core C/C++ Compilers & Build Tools:")
    tools = ["cmake", "gcc", "g++", "clang", "clang++", "cl", "ninja", "make"]
    
    for tool in tools:
        path = shutil.which(tool)
        status = f"FOUND ({path})" if path else "NOT FOUND"
        print(f"   - {tool:<10} : {status}")


# 3. Vendor FPGA & HLS Toolchains
def check_fpga_tools():
    print("\n3. FPGA & HLS Vendor Toolchains:")
    # AMD/Xilinx, Intel, and Open-Source FPGA toolchain binaries
    fpga_bins = [
        "vitis_hls",
        "vivado",
        "vitis",
        "quartus",
        "aoc",
        "dpcpp",
        "yosys",
        "verilator",
        "iverilog"
    ]
    
    found_any = False
    for b in fpga_bins:
        path = shutil.which(b)
        if path:
            found_any = True
            print(f"   [+] {b:<12} : {path}")
        else:
            print(f"   [-] {b:<12} : NOT DETECTED in PATH")
            
    if not found_any:
        print("\n   [NOTE] No commercial FPGA synthesis suite found in active PATH.")
        print("          Standard university workflow: local prototyping via C-simulation,")
        print("          or remote execution on a lab/QuantHive Linux workstation.")

# 4. DaCe Hardware Backends
def check_dace_backends():
    print("\n4. DaCe Compiler Configuration:")
    try:
        import dace
        print(f"   - DaCe Version    : {dace.__version__}")
        default_compiler = dace.Config.get("compiler", "default_compiler")
        print(f"   - Configured Compiler: {default_compiler}")
    except Exception as e:
        print(f"   - Error querying DaCe: {e}")

def main():
    inspect_system()
    check_build_tools()
    check_fpga_tools()
    check_dace_backends()
    print("\n" + "=" * 60)
    print("Audit Complete. Ready to write docs/fpga_hls_environment.md")
    print("=" * 60)

if __name__ == "__main__":
    main()