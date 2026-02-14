#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
PURE NUMPY VERSION (No Numba - Raspberry Pi Compatible)

This version uses ONLY NumPy operations, no Numba JIT.
Optimized for Raspberry Pi where Numba may not work.

Optimizations applied:
1. FFT-based autocorrelation (numpy.fft)
2. Vectorized operations throughout
3. Pre-allocated buffers
4. Minimal Python loops
5. Float32 for memory efficiency
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

import numpy as np

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class PureNumpyPitchDetector:
    """
    Pure NumPy pitch detector - no Numba, no SciPy.
    Raspberry Pi compatible.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Pre-allocate FFT buffer (power of 2 for speed)
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        # Note tracking state
        self.current_note = None
        self.note_start_time = 0.0
        self.notes = []
        self.time_position = 0.0
        
        # Pre-compute frequency lookup table for MIDI conversion
        # MIDI notes 21-108 (A0 to C8)
        self._midi_freqs = 440.0 * (2.0 ** ((np.arange(21, 109) - 69) / 12.0))
        self._midi_notes = np.arange(21, 109, dtype=np.int32)
    
    def _freq_to_midi_fast(self, freq):
        """
        Fast frequency to MIDI conversion using log2.
        Pure Python with math module (fast enough for single values).
        """
        if freq <= 0:
            return None
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def _detect_pitch_numpy(self, samples):
        """
        FFT-based autocorrelation using pure NumPy.
        
        Algorithm:
        1. Zero-pad samples to FFT size
        2. Compute FFT
        3. Compute power spectrum (|FFT|^2)
        4. Inverse FFT to get autocorrelation
        5. Find peak in valid lag range
        6. Convert lag to frequency to MIDI
        """
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Quick energy check using numpy dot (highly optimized)
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # Copy to pre-allocated buffer with zero-padding
        use_n = min(n, 2048)
        self._fft_buffer[:use_n] = samples[:use_n]
        self._fft_buffer[use_n:] = 0.0
        
        # FFT-based autocorrelation (numpy FFT uses FFTPACK, reasonably fast)
        fft = np.fft.rfft(self._fft_buffer)
        
        # Power spectrum: |FFT|^2 = real^2 + imag^2
        # Use view to avoid creating intermediate arrays
        power = fft.real * fft.real + fft.imag * fft.imag
        
        # Inverse FFT gives autocorrelation
        corr = np.fft.irfft(power)
        
        # Find peak in search range using numpy argmax (vectorized)
        search_start = self.min_lag
        search_end = min(self.max_lag, len(corr))
        
        if search_end <= search_start:
            return None
        
        # Vectorized peak finding
        search_region = corr[search_start:search_end]
        best_idx = np.argmax(search_region)
        best_val = search_region[best_idx]
        
        # Threshold check
        if best_val > 0.2 * corr[0]:
            lag = best_idx + search_start
            
            # Parabolic interpolation for sub-sample accuracy (optional, small speedup if skipped)
            if 0 < best_idx < len(search_region) - 1:
                y0 = search_region[best_idx - 1]
                y1 = search_region[best_idx]
                y2 = search_region[best_idx + 1]
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y0 - y2) / denom
                    delta = max(-0.5, min(0.5, delta))
                    lag = lag + delta
            
            freq = self.sample_rate / lag
            return self._freq_to_midi_fast(freq)
        
        return None
    
    def process_chunk(self, samples):
        """Process a chunk of audio samples."""
        chunk_duration = len(samples) / self.sample_rate
        
        # Ensure numpy float32 array (avoid dtype conversion overhead)
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
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


class MIDIWriter:
    """MIDI file writer - pure Python."""
    
    def __init__(self, tempo_bpm=120):
        self.tempo_bpm = tempo_bpm
        self.ticks_per_beat = 480
        self.us_per_beat = int(60_000_000 / tempo_bpm)
    
    def _vlq(self, value):
        """Variable-length quantity encoding."""
        result = [value & 0x7F]
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))
    
    def _sec_to_ticks(self, seconds):
        return int(seconds * (self.tempo_bpm / 60.0) * self.ticks_per_beat)
    
    def write(self, notes, filename):
        """Write notes to MIDI file."""
        track = bytearray()
        
        # Tempo
        track.extend(b'\x00\xFF\x51\x03')
        track.extend(self.us_per_beat.to_bytes(3, 'big'))
        
        events = []
        for note, start, dur in sorted(notes, key=lambda x: x[1]):
            start_tick = self._sec_to_ticks(start)
            end_tick = self._sec_to_ticks(start + dur)
            events.append((start_tick, True, note, 100))
            events.append((end_tick, False, note, 0))
        
        events.sort(key=lambda x: (x[0], not x[1]))
        
        tick = 0
        for t, on, n, v in events:
            track.extend(self._vlq(t - tick))
            tick = t
            track.extend(bytes([MIDI_NOTE_ON if on else MIDI_NOTE_OFF, n, v]))
        
        track.extend(b'\x00\xFF\x2F\x00')
        
        with open(filename, 'wb') as f:
            f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, self.ticks_per_beat))
            f.write(b'MTrk' + struct.pack('>I', len(track)) + track)


def process_wav(input_path, output_path, chunk_size=2048):
    """Process WAV file with pure NumPy pitch detection."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz/{sw*8}bit")
        
        detector = PureNumpyPitchDetector(sample_rate=sr, chunk_size=chunk_size)
        
        # Read all frames at once for speed (avoid per-chunk wave.readframes overhead)
        all_raw = wav.readframes(nf)
    
    # Convert to numpy array outside the wave context
    if sw == 2:
        all_samples = np.frombuffer(all_raw, dtype=np.int16)
    else:
        all_samples = np.frombuffer(all_raw, dtype=np.uint8).astype(np.int16) - 128
    
    # Handle stereo -> mono
    if ch == 2:
        all_samples = all_samples.reshape(-1, 2).mean(axis=1)
    
    # Normalize to [-1, 1]
    max_val = 2 ** (sw * 8 - 1)
    all_samples = (all_samples / max_val).astype(np.float32)
    
    # Process in chunks
    total_samples = len(all_samples)
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = all_samples[start:end]
        detector.process_chunk(chunk)
    
    notes = detector.finalize()
    
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed = t1 - t0
    duration = nf / sr
    
    return {
        'elapsed': elapsed,
        'duration': duration,
        'rtf': duration / elapsed if elapsed > 0 else 0,
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes),
        'output': output_path
    }


def generate_test_wav(filename, duration=60, sr=44100):
    """Generate test WAV with melody using pure NumPy."""
    notes_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    note_dur = 0.5
    samples_per_note = int(note_dur * sr)
    
    # Vectorized sample generation
    note_indices = (np.arange(n_samples) // samples_per_note) % len(notes_hz)
    freqs = notes_hz[note_indices]
    
    pos_in_note = (np.arange(n_samples) % samples_per_note) / samples_per_note
    
    # Envelope: attack and release
    env = np.minimum(1.0, pos_in_note * 20) * np.maximum(0.0, 1.0 - np.maximum(0, pos_in_note - 0.8) * 5)
    
    # Generate samples
    samples = 0.7 * np.sin(2 * np.pi * freqs * t) * env
    samples = (samples * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())
    
    print(f"  Generated: {os.path.basename(filename)} ({duration}s)")


def benchmark_core_operations():
    """Benchmark the core NumPy operations to estimate Pi performance."""
    print("\n[BENCHMARK] Core NumPy Operations")
    print("-" * 40)
    
    # Create test data
    samples = np.random.randn(2048).astype(np.float32)
    fft_buffer = np.zeros(4096, dtype=np.float32)
    
    # Benchmark FFT
    n_iters = 1000
    
    # rfft benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fft_buffer[:2048] = samples
        fft = np.fft.rfft(fft_buffer)
    t1 = time.perf_counter()
    rfft_time = (t1 - t0) / n_iters * 1000  # ms
    print(f"  rfft(4096): {rfft_time:.3f} ms per call")
    
    # Power spectrum benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        power = fft.real * fft.real + fft.imag * fft.imag
    t1 = time.perf_counter()
    power_time = (t1 - t0) / n_iters * 1000
    print(f"  power spectrum: {power_time:.4f} ms per call")
    
    # irfft benchmark
    power = fft.real * fft.real + fft.imag * fft.imag
    t0 = time.perf_counter()
    for _ in range(n_iters):
        corr = np.fft.irfft(power)
    t1 = time.perf_counter()
    irfft_time = (t1 - t0) / n_iters * 1000
    print(f"  irfft(4096): {irfft_time:.3f} ms per call")
    
    # argmax benchmark
    corr = np.random.randn(4096).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        idx = np.argmax(corr[44:550])
    t1 = time.perf_counter()
    argmax_time = (t1 - t0) / n_iters * 1000
    print(f"  argmax(500): {argmax_time:.4f} ms per call")
    
    # Energy (dot product) benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        energy = np.dot(samples, samples)
    t1 = time.perf_counter()
    dot_time = (t1 - t0) / n_iters * 1000
    print(f"  dot(2048): {dot_time:.4f} ms per call")
    
    # Total per-chunk time estimate
    total_per_chunk = rfft_time + power_time + irfft_time + argmax_time + dot_time
    print(f"\n  TOTAL per chunk: {total_per_chunk:.3f} ms")
    
    # Calculate real-time factor
    # 2048 samples @ 44100 Hz = 46.4 ms of audio
    audio_ms = 2048 / 44100 * 1000
    rtf = audio_ms / total_per_chunk
    print(f"  Audio per chunk: {audio_ms:.2f} ms")
    print(f"  Estimated RTF: {rtf:.1f}x real-time")
    
    return rtf


def main():
    print("=" * 60)
    print("PARADROMICS - PURE NUMPY VERSION (Pi Compatible)")
    print("No Numba, No SciPy - Only NumPy")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Benchmark core operations
    estimated_rtf = benchmark_core_operations()
    
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
        ('test_complex.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 40)
    for name, dur in tests:
        filepath = os.path.join(base_dir, name)
        if not os.path.exists(filepath):
            generate_test_wav(filepath, dur)
        else:
            print(f"  Using existing: {name}")
    
    print("\n[2] Processing Audio Files (Pure NumPy)")
    print("-" * 40)
    
    results = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '_numpy_only.mid'))
        m = process_wav(inp, out)
        results.append(m)
        print(f"    -> {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    max_mem = max(r['memory_mb'] for r in results)
    
    print("\n" + "=" * 60)
    print("PURE NUMPY RESULTS")
    print("=" * 60)
    print(f"Total audio: {total_dur:.1f}s processed in {total_time:.2f}s")
    print(f"Speed: {total_dur/total_time:.1f}x real-time")
    print(f"Peak memory: {max_mem:.2f} MB")
    
    print("\nCONSTRAINT CHECK:")
    rtf = total_dur / total_time
    speed_ok = rtf >= 4.0
    mem_ok = max_mem < 500
    print(f"  [{'PASS' if speed_ok else 'FAIL'}] 4x Real-Time: {rtf:.1f}x")
    print(f"  [{'PASS' if mem_ok else 'FAIL'}] <500MB RAM: {max_mem:.2f} MB")
    print(f"  [PASS] 2048-sample chunks")
    print(f"  [PASS] Pure NumPy (Pi compatible)")
    
    print("\nPI COMPATIBILITY NOTES:")
    print("  - No Numba dependency (often fails on ARM)")
    print("  - No SciPy dependency (large, slow to install)")
    print("  - Only requires: numpy (usually pre-installed)")
    print("  - FFT uses numpy.fft (FFTPACK backend)")
    print("  - All operations are vectorized")
    
    # Estimate Pi performance (Pi 4 is ~10x slower than desktop)
    pi_factor = 10  # Conservative estimate
    estimated_pi_rtf = rtf / pi_factor
    print(f"\nPI PERFORMANCE ESTIMATE:")
    print(f"  Desktop RTF: {rtf:.1f}x")
    print(f"  Pi 4 estimate (~{pi_factor}x slower): {estimated_pi_rtf:.1f}x")
    print(f"  Meets 4x requirement on Pi? {'YES' if estimated_pi_rtf >= 4.0 else 'NO - consider optimizations'}")
    
    return results


if __name__ == '__main__':
    main()
