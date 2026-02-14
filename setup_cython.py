"""
Setup script for building Cython pitch detector.

Build commands:
    python setup_cython.py build_ext --inplace

For Raspberry Pi (ARM):
    # Optimize for Pi 4 (Cortex-A72):
    CFLAGS="-O3 -march=armv8-a -mtune=cortex-a72" python setup_cython.py build_ext --inplace
    
    # Enable NEON SIMD:
    CFLAGS="-O3 -march=armv8-a+simd -mtune=cortex-a72 -ffast-math" python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import platform

# Detect platform
is_arm = platform.machine().startswith('arm') or platform.machine().startswith('aarch')
is_windows = sys.platform == 'win32'

# Compiler flags
extra_compile_args = []
extra_link_args = []

if is_windows:
    # MSVC flags
    extra_compile_args = ['/O2', '/fp:fast']
else:
    # GCC/Clang flags
    extra_compile_args = ['-O3', '-ffast-math', '-funroll-loops']
    
    if is_arm:
        # Raspberry Pi 4 specific optimizations
        extra_compile_args.extend([
            '-march=armv8-a',
            '-mtune=cortex-a72',
            '-mfpu=neon-fp-armv8',
            '-mfloat-abi=hard',
        ])
    else:
        # x86 optimizations
        extra_compile_args.extend([
            '-march=native',
        ])

# Define extension
extensions = [
    Extension(
        "pitch_detector_cython",
        ["pitch_detector_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

# Cython compiler directives
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'nonecheck': False,
    'initializedcheck': False,
    'embedsignature': True,
}

setup(
    name='pitch_detector_cython',
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generates HTML showing Python vs C code
    ),
    zip_safe=False,
)
