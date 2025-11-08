"""
Setup script for building Cython C extensions

Build with:
    python setup.py build_ext --inplace

This will compile:
- fire_spread_cython.pyx -> fire_spread_cython.so
- grid_cython.pyx -> grid_cython.so

Requirements:
- Cython: pip install cython
- C compiler (gcc on Linux, MSVC on Windows, clang on macOS)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Compiler optimization flags
extra_compile_args = []
extra_link_args = []

if sys.platform == 'linux':
    # Linux/GCC optimization flags
    extra_compile_args = [
        '-O3',              # Maximum optimization
        '-march=native',    # Optimize for current CPU
        '-ffast-math',      # Fast floating point math
        '-fopenmp',         # OpenMP support (optional)
    ]
    extra_link_args = ['-fopenmp']

elif sys.platform == 'darwin':
    # macOS/Clang
    extra_compile_args = [
        '-O3',
        '-march=native',
        '-ffast-math',
    ]

elif sys.platform == 'win32':
    # Windows/MSVC
    extra_compile_args = [
        '/O2',              # Maximum optimization
        '/fp:fast',         # Fast floating point
    ]

# Define extensions
extensions = [
    Extension(
        "fire_spread_cython",
        ["fire_spread_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "grid_cython",
        ["grid_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
]

setup(
    name="fire_simulator_cython_extensions",
    version="1.0.0",
    description="High-performance C extensions for Fire Evacuation Simulator",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,       # Disable bounds checking for speed
            'wraparound': False,         # Disable negative indexing
            'cdivision': True,           # C-style division
            'initializedcheck': False,   # Disable initialization checks
            'nonecheck': False,          # Disable None checks
            'embedsignature': True,      # Add function signatures to docstrings
        },
        annotate=True,  # Generate HTML annotation files to see C code
    ),
    zip_safe=False,
)

print("\n" + "=" * 70)
print("BUILD INSTRUCTIONS")
print("=" * 70)
print("\nTo build the C extensions, run:")
print("  python setup.py build_ext --inplace")
print("\nThis will create:")
print("  - fire_spread_cython.so (or .pyd on Windows)")
print("  - grid_cython.so (or .pyd on Windows)")
print("\nAfter building, you can import:")
print("  from fire_spread_cython import FireSpreadEngine, simulate_fire_step_fast")
print("  from grid_cython import FastGridCostCalculator")
print("\nExpected speedup:")
print("  - Fire spread: 5-10x faster than pure Python")
print("  - Grid costs: 3-5x faster than pure Python")
print("  - Overall: 2-3x on top of existing NumPy optimizations")
print("=" * 70 + "\n")
