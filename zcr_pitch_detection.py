"""
Zero-Crossing Rate (ZCR) Pitch Detection
=========================================
Optimized O(n) pitch estimation for Paradromics neural signals.

Target: Beat 4.7x real-time with 2048 chunk size, <500MB RAM

Theory:
- ZCR counts sign changes in a signal
- For a pure tone: frequency = ZCR * sample_rate / 2
- Much faster than FFT (O(n) vs O(n log n))
- Works best for monophonic, clean signals

Optimizations:
1. Vectorized numpy operations (no Python loops)
2. Numba JIT compilation for critical paths
3. Pre-allocated buffers to avoid GC
4. Low-pass filtering to reduce noise-induced false crossings
5. Interpolated zero-crossing for sub-sample accuracy
"""

import numpy as np
import time
from typing import Tuple, Optional
import warnings

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# METHOD 1: Pure NumPy ZCR (Baseline)
# =============================================================================

def zcr_numpy_basic(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Basic ZCR using numpy sign function.
    Counts all zero crossings.
    
    Frequency = (crossings / 2) / (samples / sample_rate)
             = crossings * sample_rate / (2 * samples)
    """
    signs = np.sign(signal)
    # Remove zeros (treat as positive)
    signs[signs == 0] = 1
    # Count sign changes
    crossings = np.sum(np.abs(np.diff(signs)) == 2)
    # Convert to frequency
    duration = len(signal) / sample_rate
    frequency = crossings / (2 * duration)
    return frequency


def zcr_numpy_optimized(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Optimized NumPy ZCR - fewer operations.
    Uses multiplication trick: sign change = product < 0
    """
    # Multiply adjacent samples - negative means sign change
    products = signal[:-1] * signal[1:]
    crossings = np.sum(products < 0)
    duration = len(signal) / sample_rate
    return crossings / (2 * duration)


# =============================================================================
# METHOD 2: Interpolated ZCR (Sub-sample accuracy)
# =============================================================================

def zcr_interpolated(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Linear interpolation for precise zero-crossing locations.
    Better accuracy for low frequencies.
    """
    # Find sign change indices
    signs = signal[:-1] * signal[1:]
    crossing_indices = np.where(signs < 0)[0]
    
    if len(crossing_indices) < 2:
        return 0.0
    
    # Interpolate exact crossing positions
    # Zero crossing between samples i and i+1:
    # x_zero = i + |signal[i]| / (|signal[i]| + |signal[i+1]|)
    s0 = np.abs(signal[crossing_indices])
    s1 = np.abs(signal[crossing_indices + 1])
    precise_crossings = crossing_indices + s0 / (s0 + s1)
    
    # Calculate average period from crossing intervals
    intervals = np.diff(precise_crossings)
    if len(intervals) == 0:
        return 0.0
    
    # Two crossings per period
    avg_half_period = np.median(intervals)  # Median is robust to outliers
    frequency = sample_rate / (2 * avg_half_period)
    
    return frequency


# =============================================================================
# METHOD 3: Numba-accelerated ZCR
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _zcr_numba_core(signal: np.ndarray) -> int:
    """Numba-optimized zero crossing counter."""
    crossings = 0
    n = len(signal)
    for i in range(n - 1):
        if signal[i] * signal[i + 1] < 0:
            crossings += 1
    return crossings


def zcr_numba(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """ZCR with Numba JIT compilation."""
    crossings = _zcr_numba_core(signal)
    duration = len(signal) / sample_rate
    return crossings / (2 * duration)


@jit(nopython=True, cache=True, fastmath=True)
def _zcr_numba_interpolated_core(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Numba-optimized interpolated zero crossing finder."""
    n = len(signal)
    # Pre-allocate max possible crossings
    crossings = np.empty(n, dtype=np.float64)
    count = 0
    
    for i in range(n - 1):
        if signal[i] * signal[i + 1] < 0:
            # Interpolate precise position
            s0 = abs(signal[i])
            s1 = abs(signal[i + 1])
            crossings[count] = i + s0 / (s0 + s1)
            count += 1
    
    return crossings[:count], count


def zcr_numba_interpolated(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """Numba-optimized interpolated ZCR."""
    crossings, count = _zcr_numba_interpolated_core(signal)
    
    if count < 2:
        return 0.0
    
    # Calculate frequency from crossing intervals
    intervals = np.diff(crossings)
    avg_half_period = np.median(intervals)
    return sample_rate / (2 * avg_half_period)


# =============================================================================
# METHOD 3b: Ultra-Fast Numba with Simple Lowpass (Best Balance)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _zcr_fast_filtered_core(signal: np.ndarray, alpha: float) -> int:
    """
    Combined lowpass filter + ZCR in single pass.
    This is THE optimal method for noisy signals.
    """
    n = len(signal)
    crossings = 0
    
    # First filtered sample
    prev_filtered = signal[0]
    
    for i in range(1, n):
        # IIR lowpass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        curr_filtered = alpha * signal[i] + (1.0 - alpha) * prev_filtered
        
        # Check zero crossing
        if prev_filtered * curr_filtered < 0:
            crossings += 1
        
        prev_filtered = curr_filtered
    
    return crossings


def zcr_fast_filtered(signal: np.ndarray, sample_rate: int = 44100, 
                      alpha: float = 0.15) -> float:
    """
    Ultra-fast ZCR with inline lowpass filter.
    Single pass, O(n), minimal memory.
    
    alpha: filter coefficient (lower = more smoothing, 0.1-0.3 typical)
    """
    crossings = _zcr_fast_filtered_core(signal, alpha)
    duration = len(signal) / sample_rate
    return crossings / (2 * duration)


@jit(nopython=True, cache=True, fastmath=True)
def _zcr_fast_filtered_interpolated_core(signal: np.ndarray, alpha: float) -> Tuple[np.ndarray, int]:
    """
    Combined lowpass + interpolated ZCR in single pass.
    Best accuracy + speed combination.
    """
    n = len(signal)
    crossings = np.empty(n, dtype=np.float64)
    count = 0
    
    prev_filtered = signal[0]
    
    for i in range(1, n):
        curr_filtered = alpha * signal[i] + (1.0 - alpha) * prev_filtered
        
        if prev_filtered * curr_filtered < 0:
            # Interpolate crossing position
            s0 = abs(prev_filtered)
            s1 = abs(curr_filtered)
            crossings[count] = (i - 1) + s0 / (s0 + s1)
            count += 1
        
        prev_filtered = curr_filtered
    
    return crossings[:count], count


def zcr_fast_filtered_interpolated(signal: np.ndarray, sample_rate: int = 44100,
                                    alpha: float = 0.15) -> float:
    """
    Ultra-fast interpolated ZCR with inline lowpass.
    Best balance of speed and accuracy for noisy signals.
    """
    crossings, count = _zcr_fast_filtered_interpolated_core(signal, alpha)
    
    if count < 2:
        return 0.0
    
    intervals = np.diff(crossings)
    avg_half_period = np.median(intervals)
    return sample_rate / (2 * avg_half_period)


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _zcr_batch_filtered_numba(signals: np.ndarray, alpha: float) -> np.ndarray:
    """
    Batch processing with inline filtering, parallel execution.
    """
    n_chunks, chunk_size = signals.shape
    results = np.empty(n_chunks, dtype=np.int32)
    
    for chunk_idx in prange(n_chunks):
        crossings = 0
        prev_filtered = signals[chunk_idx, 0]
        
        for i in range(1, chunk_size):
            curr_filtered = alpha * signals[chunk_idx, i] + (1.0 - alpha) * prev_filtered
            if prev_filtered * curr_filtered < 0:
                crossings += 1
            prev_filtered = curr_filtered
        
        results[chunk_idx] = crossings
    
    return results


def zcr_batch_filtered(signals: np.ndarray, sample_rate: int = 44100,
                       alpha: float = 0.15) -> np.ndarray:
    """
    Batch processing with filtering - parallel execution.
    """
    crossings = _zcr_batch_filtered_numba(signals, alpha)
    chunk_size = signals.shape[1]
    duration = chunk_size / sample_rate
    frequencies = crossings / (2 * duration)
    return frequencies


# =============================================================================
# METHOD 4: Batch Processing (Multiple Chunks)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _zcr_batch_numba(signals: np.ndarray) -> np.ndarray:
    """
    Process multiple chunks in parallel.
    signals: 2D array of shape (n_chunks, chunk_size)
    Returns: array of crossing counts
    """
    n_chunks, chunk_size = signals.shape
    results = np.empty(n_chunks, dtype=np.int32)
    
    for chunk_idx in prange(n_chunks):
        crossings = 0
        for i in range(chunk_size - 1):
            if signals[chunk_idx, i] * signals[chunk_idx, i + 1] < 0:
                crossings += 1
        results[chunk_idx] = crossings
    
    return results


def zcr_batch(signals: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """
    Process batch of signal chunks.
    signals: 2D array (n_chunks, chunk_size)
    Returns: array of frequencies
    """
    crossings = _zcr_batch_numba(signals)
    chunk_size = signals.shape[1]
    duration = chunk_size / sample_rate
    frequencies = crossings / (2 * duration)
    return frequencies


# =============================================================================
# METHOD 5: Pre-filtered ZCR (For Noisy Signals)
# =============================================================================

def simple_lowpass(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Simple IIR lowpass filter: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    Fast but effective for removing high-frequency noise.
    """
    output = np.empty_like(signal)
    output[0] = signal[0]
    for i in range(1, len(signal)):
        output[i] = alpha * signal[i] + (1 - alpha) * output[i - 1]
    return output


@jit(nopython=True, cache=True, fastmath=True)
def _simple_lowpass_numba(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Numba-accelerated lowpass filter."""
    n = len(signal)
    output = np.empty(n, dtype=np.float64)
    output[0] = signal[0]
    for i in range(1, n):
        output[i] = alpha * signal[i] + (1 - alpha) * output[i - 1]
    return output


def zcr_filtered(signal: np.ndarray, sample_rate: int = 44100, 
                 cutoff_ratio: float = 0.1) -> float:
    """
    ZCR with pre-filtering to reduce noise-induced false crossings.
    cutoff_ratio: approximate cutoff as fraction of sample rate
    """
    # Calculate alpha for desired cutoff
    alpha = cutoff_ratio * 2  # Rough approximation
    alpha = min(0.5, max(0.05, alpha))
    
    if HAS_NUMBA:
        filtered = _simple_lowpass_numba(signal.astype(np.float64), alpha)
    else:
        filtered = simple_lowpass(signal.astype(np.float64), alpha)
    
    return zcr_numba_interpolated(filtered, sample_rate) if HAS_NUMBA else zcr_interpolated(filtered, sample_rate)


# =============================================================================
# METHOD 6: Hybrid ZCR + Autocorrelation (Most Accurate)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _find_peak_in_range(autocorr: np.ndarray, min_lag: int, max_lag: int) -> int:
    """Find the lag with maximum autocorrelation in range."""
    best_lag = min_lag
    best_val = autocorr[min_lag]
    for i in range(min_lag + 1, min(max_lag, len(autocorr))):
        if autocorr[i] > best_val:
            best_val = autocorr[i]
            best_lag = i
    return best_lag


def zcr_autocorr_hybrid(signal: np.ndarray, sample_rate: int = 44100,
                        min_freq: float = 50, max_freq: float = 2000) -> float:
    """
    Use ZCR to get rough estimate, then refine with local autocorrelation.
    Best accuracy while staying fast.
    """
    # Get ZCR estimate
    zcr_freq = zcr_numba_interpolated(signal, sample_rate) if HAS_NUMBA else zcr_interpolated(signal, sample_rate)
    
    if zcr_freq < min_freq or zcr_freq > max_freq:
        return zcr_freq
    
    # Calculate expected period range around ZCR estimate
    expected_period = sample_rate / zcr_freq
    search_range = expected_period * 0.3  # ±30% range
    
    min_lag = max(1, int(expected_period - search_range))
    max_lag = min(len(signal) // 2, int(expected_period + search_range))
    
    # Compute autocorrelation only in relevant range (fast!)
    autocorr = np.correlate(signal, signal[min_lag:max_lag], mode='valid')
    
    # Find peak
    if len(autocorr) < 3:
        return zcr_freq
    
    peak_idx = np.argmax(autocorr)
    refined_period = min_lag + peak_idx
    
    if refined_period > 0:
        return sample_rate / refined_period
    return zcr_freq


# =============================================================================
# STREAMING PROCESSOR (For Real-time Use)
# =============================================================================

class ZCRPitchDetector:
    """
    Streaming pitch detector using ZCR.
    Optimized for real-time processing with minimal allocations.
    """
    
    def __init__(self, chunk_size: int = 2048, sample_rate: int = 44100,
                 method: str = 'numba_interpolated', 
                 smoothing: float = 0.7):
        """
        Initialize detector.
        
        Args:
            chunk_size: Samples per chunk (default 2048)
            sample_rate: Audio sample rate (default 44100)
            method: Detection method ('basic', 'optimized', 'interpolated', 
                    'numba', 'numba_interpolated', 'filtered', 'hybrid')
            smoothing: Exponential smoothing factor (0=no smoothing, 1=max)
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.smoothing = smoothing
        self.last_freq = 0.0
        
        # Select method
        methods = {
            'basic': zcr_numpy_basic,
            'optimized': zcr_numpy_optimized,
            'interpolated': zcr_interpolated,
            'numba': zcr_numba if HAS_NUMBA else zcr_numpy_optimized,
            'numba_interpolated': zcr_numba_interpolated if HAS_NUMBA else zcr_interpolated,
            'filtered': zcr_filtered,
            'hybrid': zcr_autocorr_hybrid,
        }
        self.method = methods.get(method, zcr_numba_interpolated if HAS_NUMBA else zcr_interpolated)
        
        # Pre-allocate buffer
        self._buffer = np.zeros(chunk_size, dtype=np.float64)
        
    def process(self, chunk: np.ndarray) -> float:
        """Process a single chunk and return pitch frequency."""
        # Copy to pre-allocated buffer (avoids allocation)
        np.copyto(self._buffer[:len(chunk)], chunk)
        
        # Detect pitch
        freq = self.method(self._buffer[:len(chunk)], self.sample_rate)
        
        # Apply smoothing
        if self.last_freq > 0 and freq > 0:
            freq = self.smoothing * self.last_freq + (1 - self.smoothing) * freq
        
        self.last_freq = freq
        return freq
    
    def process_batch(self, chunks: np.ndarray) -> np.ndarray:
        """Process multiple chunks at once."""
        if HAS_NUMBA and self.method in [zcr_numba, zcr_numba_interpolated]:
            return zcr_batch(chunks, self.sample_rate)
        else:
            return np.array([self.process(chunk) for chunk in chunks])


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_all_methods(chunk_size: int = 2048, sample_rate: int = 44100,
                          n_chunks: int = 10000, frequency: float = 440.0):
    """
    Benchmark all ZCR methods.
    Returns performance metrics for comparison.
    """
    print(f"\n{'='*60}")
    print(f"ZCR PITCH DETECTION BENCHMARK")
    print(f"{'='*60}")
    print(f"Chunk size: {chunk_size}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Test chunks: {n_chunks}")
    print(f"Target frequency: {frequency} Hz")
    print(f"Numba available: {HAS_NUMBA}")
    print(f"{'='*60}\n")
    
    # Generate test signal
    np.random.seed(42)
    t = np.arange(chunk_size) / sample_rate
    test_chunk = np.sin(2 * np.pi * frequency * t)
    # Add some noise
    test_chunk += np.random.normal(0, 0.1, chunk_size)
    
    # Generate batch data
    batch_data = np.tile(test_chunk, (n_chunks, 1))
    
    methods = [
        ('NumPy Basic', zcr_numpy_basic),
        ('NumPy Optimized', zcr_numpy_optimized),
        ('Interpolated', zcr_interpolated),
        ('Filtered (2-pass)', lambda s, sr: zcr_filtered(s, sr, 0.1)),
    ]
    
    if HAS_NUMBA:
        # Warm up Numba JIT
        _ = zcr_numba(test_chunk, sample_rate)
        _ = zcr_numba_interpolated(test_chunk, sample_rate)
        _ = zcr_fast_filtered(test_chunk, sample_rate)
        _ = zcr_fast_filtered_interpolated(test_chunk, sample_rate)
        _ = _zcr_batch_numba(batch_data[:10])
        _ = _zcr_batch_filtered_numba(batch_data[:10], 0.15)
        
        methods.extend([
            ('Numba Basic', zcr_numba),
            ('Numba Interpolated', zcr_numba_interpolated),
            ('** Fast Filtered **', zcr_fast_filtered),
            ('** Fast Filt+Interp', zcr_fast_filtered_interpolated),
            ('Hybrid (ZCR+Autocorr)', zcr_autocorr_hybrid),
        ])
    
    results = []
    
    for name, method in methods:
        # Test accuracy
        detected_freq = method(test_chunk, sample_rate)
        error_pct = abs(detected_freq - frequency) / frequency * 100
        
        # Benchmark speed
        start = time.perf_counter()
        for i in range(n_chunks):
            _ = method(test_chunk, sample_rate)
        elapsed = time.perf_counter() - start
        
        # Calculate real-time factor
        audio_duration = (chunk_size * n_chunks) / sample_rate
        rt_factor = audio_duration / elapsed
        
        chunks_per_sec = n_chunks / elapsed
        
        results.append({
            'name': name,
            'detected_freq': detected_freq,
            'error_pct': error_pct,
            'elapsed': elapsed,
            'rt_factor': rt_factor,
            'chunks_per_sec': chunks_per_sec,
        })
        
        print(f"{name:25} | {detected_freq:7.1f} Hz | Error: {error_pct:5.2f}% | "
              f"{rt_factor:6.1f}x RT | {chunks_per_sec:,.0f} chunks/s")
    
    # Batch processing benchmark
    if HAS_NUMBA:
        print(f"\n{'='*60}")
        print("BATCH PROCESSING (Parallel)")
        print(f"{'='*60}")
        
        start = time.perf_counter()
        _ = zcr_batch(batch_data, sample_rate)
        elapsed = time.perf_counter() - start
        
        audio_duration = (chunk_size * n_chunks) / sample_rate
        rt_factor = audio_duration / elapsed
        chunks_per_sec = n_chunks / elapsed
        
        print(f"{'Batch (No Filter)':25} | {rt_factor:6.1f}x RT | {chunks_per_sec:,.0f} chunks/s")
        
        # Batch with filtering
        start = time.perf_counter()
        freqs = zcr_batch_filtered(batch_data, sample_rate, 0.15)
        elapsed = time.perf_counter() - start
        
        rt_factor_filt = audio_duration / elapsed
        chunks_per_sec_filt = n_chunks / elapsed
        avg_freq = np.mean(freqs)
        error_pct = abs(avg_freq - frequency) / frequency * 100
        
        print(f"{'** Batch Filtered **':25} | {rt_factor_filt:6.1f}x RT | {chunks_per_sec_filt:,.0f} chunks/s | Avg: {avg_freq:.1f} Hz ({error_pct:.1f}% err)")
        
        results.append({
            'name': 'Numba Batch (Parallel)',
            'rt_factor': rt_factor,
            'chunks_per_sec': chunks_per_sec,
        })
        results.append({
            'name': '** Batch Filtered (Parallel)',
            'rt_factor': rt_factor_filt,
            'chunks_per_sec': chunks_per_sec_filt,
            'detected_freq': avg_freq,
            'error_pct': error_pct,
        })
    
    # Memory estimation
    print(f"\n{'='*60}")
    print("MEMORY USAGE ESTIMATE")
    print(f"{'='*60}")
    
    # Single chunk buffer
    chunk_mem = chunk_size * 8  # float64
    batch_mem = n_chunks * chunk_size * 8
    
    print(f"Single chunk buffer: {chunk_mem / 1024:.1f} KB")
    print(f"Batch buffer ({n_chunks} chunks): {batch_mem / 1024 / 1024:.1f} MB")
    print(f"Streaming mode (1 chunk): ~{chunk_mem * 3 / 1024:.1f} KB total")
    
    return results


def test_accuracy_sweep():
    """Test accuracy across frequency range."""
    print(f"\n{'='*60}")
    print("ACCURACY SWEEP TEST")
    print(f"{'='*60}")
    
    frequencies = [50, 100, 200, 440, 880, 1000, 2000, 4000]
    chunk_size = 2048
    sample_rate = 44100
    
    method = zcr_numba_interpolated if HAS_NUMBA else zcr_interpolated
    
    print(f"{'Frequency':>10} | {'Detected':>10} | {'Error':>8}")
    print("-" * 35)
    
    for freq in frequencies:
        t = np.arange(chunk_size) / sample_rate
        signal = np.sin(2 * np.pi * freq * t)
        detected = method(signal, sample_rate)
        error = abs(detected - freq) / freq * 100
        print(f"{freq:>10.0f} | {detected:>10.1f} | {error:>7.2f}%")


if __name__ == '__main__':
    # Run benchmarks
    benchmark_all_methods(
        chunk_size=2048,
        sample_rate=44100,
        n_chunks=10000,
        frequency=440.0
    )
    
    # Test accuracy
    test_accuracy_sweep()
    
    # Example usage
    print(f"\n{'='*60}")
    print("EXAMPLE USAGE")
    print(f"{'='*60}")
    
    # Create streaming detector
    detector = ZCRPitchDetector(
        chunk_size=2048,
        sample_rate=44100,
        method='numba_interpolated',
        smoothing=0.5
    )
    
    # Simulate streaming
    t = np.arange(2048) / 44100
    test_signal = np.sin(2 * np.pi * 440 * t)
    
    for i in range(5):
        freq = detector.process(test_signal)
        print(f"Chunk {i+1}: {freq:.1f} Hz")
