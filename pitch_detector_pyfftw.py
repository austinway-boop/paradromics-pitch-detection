#!/usr/bin/env python3
"""
Paradromics Pitch Detection - PyFFTW Optimized Version

PyFFTW is a Python wrapper around FFTW ("Fastest Fourier Transform in the West").
FFTW uses SIMD instructions, cache-optimized algorithms, and precomputed "wisdom"
to achieve 2-3x speedups over numpy.fft in many cases.

Key optimizations:
1. FFTW wisdom - precomputed optimal FFT plans cached for reuse
2. SIMD vectorization - uses AVX/SSE instructions
3. Memory alignment - 16-byte aligned arrays for optimal SIMD
4. Plan reuse - same plan used for all chunks (huge savings)
5. In-place transforms where possible

Requirements:
    pip install pyfftw

Benchmarks show PyFFTW can be 2-4x faster than numpy.fft for repeated
transforms of the same size (our exact use case).
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math
import numpy as np

# Try to import pyfftw
try:
    import pyfftw
    HAS_PYFFTW = True
    # Enable FFTW cache for wisdom reuse
    pyfftw.interfaces.cache.enable()
    # Set planning effort (FFTW_MEASURE is good balance of plan time vs speed)
    pyfftw.interfaces.cache.set_keepalive_time(60)  # Keep plans alive for 60s
except ImportError:
    HAS_PYFFTW = False
    print("Warning: PyFFTW not available. Install with: pip install pyfftw")
    print("Falling back to numpy.fft")


class PyFFTWPitchDetector:
    """
    Ultra-optimized pitch detector using PyFFTW for FFT operations.
    
    PyFFTW advantages over numpy.fft:
    - FFTW library uses SIMD (AVX, SSE) 
    - Precomputed "wisdom" for optimal algorithm selection
    - Memory-aligned arrays for cache efficiency
    - Plan reuse (computed once, used millions of times)
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000,
                 fft_size=4096, planning_rigor='FFTW_MEASURE'):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        self.fft_size = fft_size
        
        if HAS_PYFFTW:
            # Create byte-aligned arrays for SIMD efficiency
            # pyfftw.empty_aligned ensures 16-byte alignment (optimal for SSE)
            self._input_buffer = pyfftw.empty_aligned(fft_size, dtype='float32')
            self._fft_output = pyfftw.empty_aligned(fft_size // 2 + 1, dtype='complex64')
            self._ifft_output = pyfftw.empty_aligned(fft_size, dtype='float32')
            
            # Pre-plan FFT operations (this is where FFTW shines!)
            # Planning rigor options:
            #   FFTW_ESTIMATE - fastest planning, decent speed
            #   FFTW_MEASURE - medium planning time, good speed (recommended)
            #   FFTW_PATIENT - slow planning, excellent speed
            #   FFTW_EXHAUSTIVE - very slow planning, optimal speed
            
            # Forward FFT plan (real -> complex)
            self._fft_plan = pyfftw.FFTW(
                self._input_buffer, 
                self._fft_output,
                direction='FFTW_FORWARD',
                flags=[planning_rigor],
                threads=1  # Single thread for streaming (less overhead)
            )
            
            # Create power spectrum buffer
            self._power_buffer = pyfftw.empty_aligned(fft_size // 2 + 1, dtype='float32')
            
            # Inverse FFT plan (complex -> real)  
            # For autocorrelation, we use c2r (complex to real)
            self._ifft_plan = pyfftw.FFTW(
                self._fft_output,
                self._ifft_output,
                direction='FFTW_BACKWARD',
                flags=[planning_rigor],
                threads=1
            )
            
            self._use_pyfftw = True
            print(f"[PyFFTW] Initialized with {planning_rigor}, SIMD: {pyfftw.simd_alignment} bytes")
            
        else:
            # Fallback to numpy
            self._input_buffer = np.zeros(fft_size, dtype=np.float32)
            self._use_pyfftw = False
        
        # Note tracking state
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
    
    def _detect_pitch_pyfftw(self, samples):
        """
        FFT-based autocorrelation using PyFFTW.
        
        Autocorrelation via FFT:
        1. FFT(signal) -> spectrum
        2. power = |spectrum|^2
        3. IFFT(power) -> autocorrelation
        4. Find peak in lag range -> frequency
        """
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Quick energy check
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # Copy samples into aligned input buffer
        use_n = min(n, 2048)
        self._input_buffer[:use_n] = samples[:use_n]
        self._input_buffer[use_n:] = 0  # Zero-pad
        
        # Forward FFT (uses precomputed plan - very fast!)
        self._fft_plan()
        
        # Compute power spectrum in-place
        # power = real^2 + imag^2
        np.multiply(self._fft_output.real, self._fft_output.real, out=self._power_buffer)
        self._power_buffer += self._fft_output.imag ** 2
        
        # Copy power to complex buffer for IFFT
        self._fft_output.real[:] = self._power_buffer
        self._fft_output.imag[:] = 0
        
        # Inverse FFT to get autocorrelation
        self._ifft_plan()
        
        # Find peak in lag range
        corr = self._ifft_output
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
        
        best_idx = np.argmax(search)
        
        if search[best_idx] > 0.2 * corr[0]:
            freq = self.sample_rate / (best_idx + self.min_lag)
            return self._freq_to_midi(freq)
        
        return None
    
    def _detect_pitch_numpy(self, samples):
        """Fallback to numpy FFT."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        use_n = min(n, 2048)
        self._input_buffer[:use_n] = samples[:use_n]
        self._input_buffer[use_n:] = 0
        
        fft = np.fft.rfft(self._input_buffer)
        power = fft.real * fft.real + fft.imag * fft.imag
        corr = np.fft.irfft(power)
        
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
            
        best_idx = search.argmax()
        
        if search[best_idx] > 0.2 * corr[0]:
            freq = self.sample_rate / (best_idx + self.min_lag)
            return self._freq_to_midi(freq)
        
        return None
    
    def _freq_to_midi(self, freq):
        """Convert frequency to MIDI note number."""
        if freq <= 0:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def process_chunk(self, samples):
        """Process a chunk of audio samples."""
        chunk_duration = len(samples) / self.sample_rate
        
        samples = np.asarray(samples, dtype=np.float32)
        
        if self._use_pyfftw:
            midi_note = self._detect_pitch_pyfftw(samples)
        else:
            midi_note = self._detect_pitch_numpy(samples)
        
        self._update_note_state(midi_note, chunk_duration)
        self.time_position += chunk_duration
        
        return midi_note
    
    def _update_note_state(self, new_note, duration):
        """Track note on/off events."""
        if new_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:
                    self.notes.append((
                        self.current_note,
                        self.note_start_time,
                        note_duration
                    ))
            
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    def finalize(self):
        """Finalize and return all notes."""
        if self.current_note is not None:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((
                    self.current_note,
                    self.note_start_time,
                    note_duration
                ))
        return self.notes


class PyFFTWPitchDetectorV2:
    """
    Version 2: Even more optimized PyFFTW usage.
    
    Additional optimizations:
    1. Use pyfftw.interfaces.numpy_fft (drop-in replacement)
    2. Enable cache globally
    3. Avoid unnecessary copies
    4. Use float32 throughout (faster than float64)
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        self.fft_size = 4096
        
        if HAS_PYFFTW:
            # Use the numpy_fft interface which auto-caches plans
            self._rfft = pyfftw.interfaces.numpy_fft.rfft
            self._irfft = pyfftw.interfaces.numpy_fft.irfft
            self._use_pyfftw = True
        else:
            self._rfft = np.fft.rfft
            self._irfft = np.fft.irfft
            self._use_pyfftw = False
        
        # Pre-allocate buffer
        self._buffer = pyfftw.empty_aligned(self.fft_size, dtype='float32') if HAS_PYFFTW else np.zeros(self.fft_size, dtype=np.float32)
        
        # Note tracking
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
    
    def _detect_pitch(self, samples):
        """Pitch detection using cached PyFFTW plans."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        use_n = min(n, 2048)
        self._buffer[:use_n] = samples[:use_n]
        self._buffer[use_n:] = 0
        
        # FFT-based autocorrelation
        fft = self._rfft(self._buffer)
        power = fft.real ** 2 + fft.imag ** 2
        corr = self._irfft(power)
        
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
            
        best_idx = np.argmax(search)
        
        if search[best_idx] > 0.2 * corr[0]:
            freq = self.sample_rate / (best_idx + self.min_lag)
            return self._freq_to_midi(freq)
        
        return None
    
    def _freq_to_midi(self, freq):
        if freq <= 0:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        return midi_rounded if 21 <= midi_rounded <= 108 else None
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        samples = np.asarray(samples, dtype=np.float32)
        midi_note = self._detect_pitch(samples)
        
        if midi_note != self.current_note:
            if self.current_note is not None:
                note_dur = self.time_position - self.note_start_time
                if note_dur > 0.03:
                    self.notes.append((self.current_note, self.note_start_time, note_dur))
            self.current_note = midi_note
            self.note_start_time = self.time_position
        
        self.time_position += chunk_duration
        return midi_note
    
    def finalize(self):
        if self.current_note is not None:
            note_dur = self.time_position - self.note_start_time
            if note_dur > 0.03:
                self.notes.append((self.current_note, self.note_start_time, note_dur))
        return self.notes


class PyFFTWPitchDetectorV3:
    """
    Version 3: Maximum performance with multi-threading.
    
    FFTW supports multi-threaded execution for larger transforms.
    For small transforms like ours, single-threaded is usually faster
    due to threading overhead, but we test both.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000,
                 threads=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        self.fft_size = 4096
        self.threads = threads
        
        if HAS_PYFFTW:
            # Aligned arrays
            self._in_buf = pyfftw.empty_aligned(self.fft_size, dtype='float32')
            self._fft_buf = pyfftw.empty_aligned(self.fft_size // 2 + 1, dtype='complex64')
            self._out_buf = pyfftw.empty_aligned(self.fft_size, dtype='float32')
            
            # Create plans with specified threads
            self._fft = pyfftw.FFTW(
                self._in_buf, self._fft_buf,
                direction='FFTW_FORWARD',
                flags=['FFTW_MEASURE'],
                threads=threads
            )
            self._ifft = pyfftw.FFTW(
                self._fft_buf, self._out_buf,
                direction='FFTW_BACKWARD', 
                flags=['FFTW_MEASURE'],
                threads=threads
            )
            self._use_pyfftw = True
        else:
            self._in_buf = np.zeros(self.fft_size, dtype=np.float32)
            self._use_pyfftw = False
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
    
    def _detect_pitch(self, samples):
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        use_n = min(n, 2048)
        self._in_buf[:use_n] = samples[:use_n]
        self._in_buf[use_n:] = 0
        
        if self._use_pyfftw:
            # Execute pre-planned FFT
            self._fft()
            
            # Power spectrum
            self._fft_buf.real[:] = self._fft_buf.real ** 2 + self._fft_buf.imag ** 2
            self._fft_buf.imag[:] = 0
            
            # IFFT
            self._ifft()
            corr = self._out_buf
        else:
            fft = np.fft.rfft(self._in_buf)
            power = fft.real ** 2 + fft.imag ** 2
            corr = np.fft.irfft(power)
        
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
            
        best_idx = np.argmax(search)
        
        if search[best_idx] > 0.2 * corr[0]:
            freq = self.sample_rate / (best_idx + self.min_lag)
            midi = 69 + 12 * math.log2(freq / 440.0)
            midi_rounded = int(round(midi))
            return midi_rounded if 21 <= midi_rounded <= 108 else None
        
        return None
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        samples = np.asarray(samples, dtype=np.float32)
        midi_note = self._detect_pitch(samples)
        
        if midi_note != self.current_note:
            if self.current_note is not None:
                note_dur = self.time_position - self.note_start_time
                if note_dur > 0.03:
                    self.notes.append((self.current_note, self.note_start_time, note_dur))
            self.current_note = midi_note
            self.note_start_time = self.time_position
        
        self.time_position += chunk_duration
        return midi_note
    
    def finalize(self):
        if self.current_note is not None:
            note_dur = self.time_position - self.note_start_time
            if note_dur > 0.03:
                self.notes.append((self.current_note, self.note_start_time, note_dur))
        return self.notes


# Import original for comparison
try:
    from pitch_detector import StreamingPitchDetector as OriginalDetector
except ImportError:
    OriginalDetector = None


def benchmark(wav_path):
    """Benchmark PyFFTW vs NumPy FFT."""
    
    # Load audio
    with wave.open(wav_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        raw_data = wav.readframes(nf)
    
    # Decode
    ns = len(raw_data) // (sw * ch)
    samples_raw = struct.unpack(f'<{ns * ch}h', raw_data) if sw == 2 else [b-128 for b in raw_data]
    if ch == 2:
        samples_raw = [(samples_raw[i] + samples_raw[i+1]) / 2 for i in range(0, len(samples_raw), 2)]
    
    max_val = 2 ** (sw * 8 - 1)
    all_samples = np.array([s / max_val for s in samples_raw], dtype=np.float32)
    
    duration = len(all_samples) / sr
    chunk_size = 2048
    
    print(f"\n{'='*60}")
    print(f"PYFFTW BENCHMARK: {os.path.basename(wav_path)} ({duration:.1f}s)")
    print(f"{'='*60}")
    print(f"PyFFTW available: {HAS_PYFFTW}")
    if HAS_PYFFTW:
        print(f"SIMD alignment: {pyfftw.simd_alignment} bytes")
        print(f"Threads available: {pyfftw.config.NUM_THREADS}")
    print()
    
    detectors = []
    
    if OriginalDetector:
        detectors.append(("NumPy FFT (Original)", OriginalDetector, {}))
    
    if HAS_PYFFTW:
        detectors.extend([
            ("PyFFTW V1 (FFTW_MEASURE)", PyFFTWPitchDetector, {'planning_rigor': 'FFTW_MEASURE'}),
            ("PyFFTW V1 (FFTW_PATIENT)", PyFFTWPitchDetector, {'planning_rigor': 'FFTW_PATIENT'}),
            ("PyFFTW V2 (numpy_fft interface)", PyFFTWPitchDetectorV2, {}),
            ("PyFFTW V3 (1 thread)", PyFFTWPitchDetectorV3, {'threads': 1}),
            ("PyFFTW V3 (2 threads)", PyFFTWPitchDetectorV3, {'threads': 2}),
        ])
    else:
        detectors.append(("NumPy FFT (Fallback)", PyFFTWPitchDetector, {}))
    
    results = []
    
    for name, DetectorClass, kwargs in detectors:
        # Warm-up run (important for PyFFTW planning)
        detector = DetectorClass(sample_rate=sr, chunk_size=chunk_size, **kwargs)
        for i in range(0, min(len(all_samples), chunk_size * 10), chunk_size):
            chunk = all_samples[i:i+chunk_size]
            if len(chunk) > 0:
                detector.process_chunk(chunk)
        
        # Actual benchmark
        tracemalloc.start()
        t0 = time.perf_counter()
        
        detector = DetectorClass(sample_rate=sr, chunk_size=chunk_size, **kwargs)
        
        for i in range(0, len(all_samples), chunk_size):
            chunk = all_samples[i:i+chunk_size]
            if len(chunk) > 0:
                detector.process_chunk(chunk)
        
        notes = detector.finalize()
        
        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = t1 - t0
        rtf = duration / elapsed if elapsed > 0 else 0
        
        results.append({
            'name': name,
            'elapsed': elapsed,
            'rtf': rtf,
            'memory_mb': peak_mem / (1024 * 1024),
            'notes': len(notes),
        })
        
        print(f"  {name}")
        print(f"    Speed: {rtf:.1f}x RT ({elapsed:.3f}s for {duration:.1f}s audio)")
        print(f"    Memory: {peak_mem/1024/1024:.2f}MB | Notes: {len(notes)}")
        print()
    
    # Summary
    if len(results) > 1:
        baseline = results[0]
        print(f"\n{'='*60}")
        print("SPEEDUP vs BASELINE (NumPy FFT)")
        print(f"{'='*60}")
        
        for r in results[1:]:
            speedup = r['rtf'] / baseline['rtf'] if baseline['rtf'] > 0 else 0
            print(f"  {r['name']}: {speedup:.2f}x faster ({r['rtf']:.1f}x RT vs {baseline['rtf']:.1f}x RT)")
    
    return results


def main():
    print("="*60)
    print("PARADROMICS - PyFFTW OPTIMIZATION ANALYSIS")
    print("="*60)
    
    if not HAS_PYFFTW:
        print("\n⚠️  PyFFTW not installed!")
        print("    Install with: pip install pyfftw")
        print("    On Windows: pip install pyfftw-binary")
        print("\n    Falling back to numpy.fft benchmarks...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_files = ['test_simple.wav', 'test_clean.wav', 'test_complex.wav']
    
    all_results = []
    for test_file in test_files:
        wav_path = os.path.join(base_dir, test_file)
        if os.path.exists(wav_path):
            results = benchmark(wav_path)
            all_results.append((test_file, results))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if all_results and HAS_PYFFTW:
        # Average across all files
        method_rtfs = {}
        for test_file, results in all_results:
            for r in results:
                if r['name'] not in method_rtfs:
                    method_rtfs[r['name']] = []
                method_rtfs[r['name']].append(r['rtf'])
        
        baseline_name = list(method_rtfs.keys())[0]
        baseline_avg = sum(method_rtfs[baseline_name]) / len(method_rtfs[baseline_name])
        
        print(f"\nBaseline ({baseline_name}): {baseline_avg:.1f}x RT")
        print("\nPyFFTW Methods:")
        
        best_method = None
        best_speedup = 0
        
        for name, rtfs in method_rtfs.items():
            if name == baseline_name:
                continue
            avg_rtf = sum(rtfs) / len(rtfs)
            speedup = avg_rtf / baseline_avg if baseline_avg > 0 else 0
            print(f"  {name}: {avg_rtf:.1f}x RT ({speedup:.2f}x speedup)")
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_method = name
        
        print(f"\n✓ BEST: {best_method}")
        print(f"  Speedup: {best_speedup:.2f}x over baseline")
        
        if best_speedup < 1.1:
            print("\n⚠️  PyFFTW shows minimal improvement for this workload.")
            print("   Reasons:")
            print("   - Small FFT sizes (4096) don't benefit much from FFTW")
            print("   - NumPy uses optimized BLAS/LAPACK under the hood")
            print("   - Python overhead dominates at this scale")
            print("   - Consider Numba/Cython for more gains")
    
    elif not HAS_PYFFTW:
        print("\n📦 To test PyFFTW:")
        print("    pip install pyfftw")
        print("    # or on Windows:")
        print("    pip install pyfftw-binary")
        print("\n    Then re-run this benchmark.")


if __name__ == '__main__':
    main()
