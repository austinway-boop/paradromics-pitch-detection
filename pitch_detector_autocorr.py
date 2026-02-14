#!/usr/bin/env python3
"""
Paradromics Pi Simulation - Agent 7: Autocorrelation Pitch Detector
Time-domain autocorrelation without FFT - potentially faster on Pi
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import sys

@dataclass
class AutocorrConfig:
    """Configuration for autocorrelation pitch detector"""
    sample_rate: int = 16000
    chunk_size: int = 512
    min_freq: float = 80.0    # Hz - lowest pitch to detect
    max_freq: float = 400.0   # Hz - highest pitch (human voice range)
    threshold: float = 0.3    # Correlation threshold for valid pitch
    
    @property
    def min_lag(self) -> int:
        """Minimum lag (samples) for max frequency"""
        return int(self.sample_rate / self.max_freq)
    
    @property
    def max_lag(self) -> int:
        """Maximum lag (samples) for min frequency"""
        return int(self.sample_rate / self.min_freq)


class AutocorrelationPitchDetector:
    """
    Time-domain autocorrelation pitch detector.
    No FFT - uses direct correlation computation.
    """
    
    def __init__(self, config: Optional[AutocorrConfig] = None):
        self.config = config or AutocorrConfig()
        self.processing_times = []
        
    def autocorrelate_naive(self, signal: np.ndarray) -> np.ndarray:
        """
        Naive O(n*m) autocorrelation - baseline for comparison.
        Computes correlation only for relevant lag range.
        """
        n = len(signal)
        min_lag = self.config.min_lag
        max_lag = min(self.config.max_lag, n - 1)
        
        correlations = np.zeros(max_lag - min_lag + 1)
        
        for i, lag in enumerate(range(min_lag, max_lag + 1)):
            # Direct correlation at this lag
            correlation = 0.0
            for j in range(n - lag):
                correlation += signal[j] * signal[j + lag]
            correlations[i] = correlation
            
        return correlations, min_lag
    
    def autocorrelate_vectorized(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Vectorized autocorrelation using numpy - faster than naive.
        Still O(n*m) but with SIMD optimization.
        """
        n = len(signal)
        min_lag = self.config.min_lag
        max_lag = min(self.config.max_lag, n - 1)
        
        correlations = np.zeros(max_lag - min_lag + 1)
        
        for i, lag in enumerate(range(min_lag, max_lag + 1)):
            # Vectorized dot product for this lag
            correlations[i] = np.dot(signal[:n-lag], signal[lag:])
            
        return correlations, min_lag
    
    def autocorrelate_normalized(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Normalized autocorrelation - better pitch detection accuracy.
        Returns correlation coefficients in [-1, 1] range.
        """
        n = len(signal)
        min_lag = self.config.min_lag
        max_lag = min(self.config.max_lag, n - 1)
        
        # Precompute cumulative energy for normalization
        signal_sq = signal ** 2
        cumsum = np.cumsum(signal_sq)
        
        correlations = np.zeros(max_lag - min_lag + 1)
        
        for i, lag in enumerate(range(min_lag, max_lag + 1)):
            # Correlation
            corr = np.dot(signal[:n-lag], signal[lag:])
            
            # Energy of both segments
            energy1 = cumsum[n-lag-1]
            energy2 = cumsum[n-1] - cumsum[lag-1]
            
            # Normalized correlation
            norm = np.sqrt(energy1 * energy2)
            if norm > 1e-10:
                correlations[i] = corr / norm
            else:
                correlations[i] = 0.0
                
        return correlations, min_lag
    
    def autocorrelate_downsampled(self, signal: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, int]:
        """
        Downsampled autocorrelation for speed - coarse pitch estimation.
        Good for initial estimate, can refine afterward.
        """
        # Downsample signal
        downsampled = signal[::factor]
        n = len(downsampled)
        
        # Adjust lag range for downsampled rate
        min_lag = max(1, self.config.min_lag // factor)
        max_lag = min(self.config.max_lag // factor, n - 1)
        
        correlations = np.zeros(max_lag - min_lag + 1)
        
        for i, lag in enumerate(range(min_lag, max_lag + 1)):
            correlations[i] = np.dot(downsampled[:n-lag], downsampled[lag:])
            
        return correlations, min_lag * factor  # Return original-scale lag
    
    def find_pitch_from_correlation(self, correlations: np.ndarray, min_lag: int) -> Tuple[float, float]:
        """
        Find fundamental frequency from correlation array.
        Returns (frequency_hz, confidence).
        """
        if len(correlations) == 0:
            return 0.0, 0.0
            
        # Find peak in correlations
        peak_idx = np.argmax(correlations)
        peak_value = correlations[peak_idx]
        
        # Normalize confidence - use max value as reference
        max_val = np.max(np.abs(correlations))
        confidence = peak_value / max_val if max_val > 0 else 0.0
        
        # For autocorrelation, we always get a valid pitch (it's about how strong the periodicity is)
        # Convert lag to frequency
        lag = min_lag + peak_idx
        if lag > 0:
            freq = self.config.sample_rate / lag
        else:
            freq = 0.0
            
        return freq, abs(confidence)
    
    def detect_pitch_naive(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """Detect pitch using naive autocorrelation"""
        start = time.perf_counter()
        correlations, min_lag = self.autocorrelate_naive(signal)
        freq, confidence = self.find_pitch_from_correlation(correlations, min_lag)
        elapsed = time.perf_counter() - start
        return freq, confidence, elapsed
    
    def detect_pitch_vectorized(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """Detect pitch using vectorized autocorrelation"""
        start = time.perf_counter()
        correlations, min_lag = self.autocorrelate_vectorized(signal)
        freq, confidence = self.find_pitch_from_correlation(correlations, min_lag)
        elapsed = time.perf_counter() - start
        return freq, confidence, elapsed
    
    def detect_pitch_normalized(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """Detect pitch using normalized autocorrelation"""
        start = time.perf_counter()
        correlations, min_lag = self.autocorrelate_normalized(signal)
        freq, confidence = self.find_pitch_from_correlation(correlations, min_lag)
        elapsed = time.perf_counter() - start
        return freq, confidence, elapsed
    
    def detect_pitch_downsampled(self, signal: np.ndarray, factor: int = 2) -> Tuple[float, float, float]:
        """Detect pitch using downsampled autocorrelation"""
        start = time.perf_counter()
        correlations, min_lag = self.autocorrelate_downsampled(signal, factor)
        freq, confidence = self.find_pitch_from_correlation(correlations, min_lag)
        elapsed = time.perf_counter() - start
        return freq, confidence, elapsed


def generate_test_signal(freq: float, duration: float, sample_rate: int, noise_level: float = 0.1) -> np.ndarray:
    """Generate test signal with known frequency"""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    # Fundamental + harmonics (more realistic voice)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * freq * t)  # 3rd harmonic
    # Add noise
    signal += noise_level * np.random.randn(len(signal))
    return signal.astype(np.float32)


def benchmark_all_methods(iterations: int = 100):
    """Benchmark all autocorrelation methods"""
    config = AutocorrConfig(
        sample_rate=16000,
        chunk_size=512,
        min_freq=80.0,
        max_freq=400.0
    )
    detector = AutocorrelationPitchDetector(config)
    
    # Test frequencies
    test_freqs = [100.0, 150.0, 200.0, 250.0, 300.0]
    
    print("=" * 70)
    print("AUTOCORRELATION PITCH DETECTOR BENCHMARK")
    print("=" * 70)
    print(f"Sample Rate: {config.sample_rate} Hz")
    print(f"Chunk Size: {config.chunk_size} samples ({config.chunk_size/config.sample_rate*1000:.1f} ms)")
    print(f"Frequency Range: {config.min_freq}-{config.max_freq} Hz")
    print(f"Lag Range: {config.min_lag}-{config.max_lag} samples")
    print(f"Iterations: {iterations}")
    print("=" * 70)
    
    methods = [
        ("Naive (pure loops)", detector.detect_pitch_naive),
        ("Vectorized (numpy)", detector.detect_pitch_vectorized),
        ("Normalized", detector.detect_pitch_normalized),
        ("Downsampled 2x", lambda s: detector.detect_pitch_downsampled(s, 2)),
        ("Downsampled 4x", lambda s: detector.detect_pitch_downsampled(s, 4)),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n--- {method_name} ---")
        
        times = []
        errors = []
        
        for test_freq in test_freqs:
            signal = generate_test_signal(test_freq, config.chunk_size / config.sample_rate, config.sample_rate)
            
            method_times = []
            method_errors = []
            
            for _ in range(iterations):
                freq, confidence, elapsed = method_func(signal)
                method_times.append(elapsed)
                if freq > 0:
                    method_errors.append(abs(freq - test_freq))
            
            times.extend(method_times)
            errors.extend(method_errors)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        avg_error = np.mean(errors) if errors else float('inf')
        
        results[method_name] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'avg_error_hz': avg_error,
            'throughput': 1000 / avg_time if avg_time > 0 else 0
        }
        
        print(f"  Avg Time: {avg_time:.3f} ms (±{std_time:.3f})")
        print(f"  Avg Error: {avg_error:.2f} Hz")
        print(f"  Throughput: {results[method_name]['throughput']:.1f} chunks/sec")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Time (ms)':<12} {'Error (Hz)':<12} {'Chunks/s':<12}")
    print("-" * 70)
    
    for method_name, data in results.items():
        print(f"{method_name:<25} {data['avg_time_ms']:<12.3f} {data['avg_error_hz']:<12.2f} {data['throughput']:<12.1f}")
    
    # Pi estimation
    print("\n" + "=" * 70)
    print("RASPBERRY PI ESTIMATION (÷10 performance factor)")
    print("=" * 70)
    
    pi_factor = 10  # Conservative estimate: Pi is ~10x slower
    
    for method_name, data in results.items():
        pi_time = data['avg_time_ms'] * pi_factor
        pi_throughput = data['throughput'] / pi_factor
        realtime = "[OK] REAL-TIME" if pi_time < 32 else "[X] TOO SLOW"  # 32ms = chunk duration
        print(f"{method_name:<25} {pi_time:<12.1f} ms  {pi_throughput:<8.1f} ch/s  {realtime}")
    
    # Comparison with FFT (theoretical)
    print("\n" + "=" * 70)
    print("COMPARISON NOTES")
    print("=" * 70)
    print("""
    Autocorrelation vs FFT:
    
    AUTOCORRELATION PROS:
    - Simpler implementation (no complex math)
    - Can target specific frequency range (fewer computations)
    - No windowing artifacts
    - Better for single dominant frequency
    
    AUTOCORRELATION CONS:
    - O(n*m) complexity vs O(n log n) for FFT
    - Less frequency resolution
    - Can't easily extract multiple frequencies
    
    FOR PARADROMICS PITCH DETECTION:
    - Vectorized or Downsampled 2x are best balance
    - Real-time feasible on Pi with downsampling
    - Could use downsampled for coarse + refine for accuracy
    """)
    
    return results


def main():
    """Main entry point"""
    print("\nParadromics Pi Simulation - Autocorrelation Pitch Detector")
    print("Agent 7: Testing time-domain alternatives to FFT\n")
    
    results = benchmark_all_methods(iterations=100)
    
    # Quick accuracy test
    print("\n" + "=" * 70)
    print("ACCURACY TEST - Known Frequencies")
    print("=" * 70)
    
    config = AutocorrConfig()
    detector = AutocorrelationPitchDetector(config)
    
    test_cases = [
        (100.0, "Low male voice"),
        (150.0, "Typical male voice"),
        (200.0, "High male / low female"),
        (250.0, "Typical female voice"),
        (300.0, "High female voice"),
    ]
    
    print(f"{'True Freq':<12} {'Detected':<12} {'Error':<10} {'Confidence':<12} {'Description'}")
    print("-" * 70)
    
    for true_freq, desc in test_cases:
        signal = generate_test_signal(true_freq, 0.032, config.sample_rate, noise_level=0.1)
        freq, confidence, _ = detector.detect_pitch_vectorized(signal)
        error = abs(freq - true_freq) if freq > 0 else float('inf')
        print(f"{true_freq:<12.1f} {freq:<12.1f} {error:<10.2f} {confidence:<12.3f} {desc}")
    
    print("\n[OK] Autocorrelation benchmark complete!")


if __name__ == "__main__":
    main()
