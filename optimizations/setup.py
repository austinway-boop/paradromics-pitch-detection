"""
Setup script for building Cython pitch detection module.

Build commands:
    # Standard build (requires Visual Studio Build Tools on Windows)
    python setup.py build_ext --inplace
    
    # Build with optimization flags
    python setup.py build_ext --inplace --define CYTHON_TRACE=1
    
    # Clean and rebuild
    python setup.py clean --all
    python setup.py build_ext --inplace

Requirements:
    - Cython
    - NumPy
    - C compiler (Visual Studio Build Tools on Windows, GCC on Linux)
    
Install build dependencies:
    pip install cython numpy

Usage after build:
    from pitch_cython import zcr_pitch, autocorr_pitch, yin_pitch, StreamingPitchDetector
"""

import os
import sys
from setuptools import setup, Extension

# Try to import Cython
try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("WARNING: Cython not found. Install with: pip install cython")
    sys.exit(1)

import numpy as np

# Cython compiler directives for maximum performance
cython_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'nonecheck': False,
    'initializedcheck': False,
    'overflowcheck': False,
    'infer_types': True,
    'embedsignature': True,  # Include function signatures in docstrings
}

# Platform-specific compiler flags
if sys.platform == 'win32':
    # Windows (MSVC)
    extra_compile_args = [
        '/O2',           # Maximum optimization
        '/Ob3',          # Aggressive inlining
        '/fp:fast',      # Fast floating-point
        '/GS-',          # Disable buffer security check
        '/GL',           # Whole program optimization
        '/arch:AVX2',    # Use AVX2 instructions if available
    ]
    extra_link_args = ['/LTCG']  # Link-time code generation
else:
    # Linux/macOS (GCC/Clang)
    extra_compile_args = [
        '-O3',                    # Maximum optimization
        '-ffast-math',            # Fast floating-point
        '-march=native',          # Optimize for current CPU
        '-funroll-loops',         # Unroll loops
        '-fno-strict-aliasing',   # Relax aliasing rules
        '-fomit-frame-pointer',   # Free up a register
    ]
    extra_link_args = []

# Define the extension
extensions = [
    Extension(
        name='pitch_cython',
        sources=['pitch_cython.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
]

# Build with Cython
extensions = cythonize(
    extensions,
    compiler_directives=cython_directives,
    annotate=True,  # Generate HTML annotation showing C code
)

setup(
    name='paradromics_pitch_cython',
    version='1.0.0',
    description='High-performance Cython pitch detection for Paradromics',
    author='Paradromics Optimization',
    ext_modules=extensions,
    install_requires=[
        'numpy>=1.20.0',
        'cython>=0.29.0',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)
