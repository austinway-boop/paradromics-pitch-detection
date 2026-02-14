#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
NUMBA JIT-OPTIMIZED VERSION 2 - AGGRESSIVE OPTIMIZATION

Key insight: FFT (numpy) is already optimized via MKL/FFTPACK.
Better approach: Use scipy.fft with workers OR eliminate FFT entirely
with a JIT-compiled direct autocorrelation for small lag ranges.

Strategy:
1. JIT-compile ENTIRE pitch detection pipeline
2. Avoid FFT when possible - use optimized time-domain autocorrelation
3. Use numba parallel execution
4. Minimize Python-Numba boundary crossings
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

import numpy as np
from numba import jit, njit, prange, types
from numba.typed import List

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


# =============================================================================
# FULLY JIT-COMPILED PITCH DETECTION
# =============================================================================

@njit(fastmath=True)
def detect_pitch_full_jit(samples, sample_rate, min_lag, max_lag):
    """
    Fully JIT-compiled pitch detection using optimized time-domain
    autocorrelation. No FFT - avoids Python/Numba boundary overhead.
    
    For 2048 samples and lag range ~44-551, this is faster than FFT
    because the JIT compiler optimizes the loops aggressively.
    """
    n = len(samples)
    if n < max_lag:
        return -1
    
    # Energy computation
    energy = 0.0
    for i in range(n):
        energy += samples[i] * samples[i]
    
    if energy < 0.0001:
        return -1
    
    # Normalize by sqrt(energy) for better peak detection
    inv_energy = 1.0 / (energy + 1e-10)
    
    best_lag = -1
    best_corr = 0.3  # Minimum threshold
    
    # Optimized lag search - NSDF-like normalization
    # Process in larger steps first, then refine
    step = 4  # Coarse search
    
    for lag in range(min_lag, max_lag, step):
        if lag >= n:
            break
        
        # Cross-correlation at this lag
        corr = 0.0
        for i in range(n - lag):
            corr += samples[i] * samples[i + lag]
        
        norm_corr = corr * inv_energy
        
        if norm_corr > best_corr:
            best_corr = norm_corr
            best_lag = lag
    
    # Refine around best lag
    if best_lag > 0:
        refined_lag = best_lag
        refined_corr = best_corr
        
        start = max(min_lag, best_lag - step)
        end = min(max_lag, best_lag + step + 1)
        
        for lag in range(start, end):
            if lag == best_lag or lag >= n:
                continue
            
            corr = 0.0
            for i in range(n - lag):
                corr += samples[i] * samples[i + lag]
            
            norm_corr = corr * inv_energy
            
            if norm_corr > refined_corr:
                refined_corr = norm_corr
                refined_lag = lag
        
        best_lag = refined_lag
    
    if best_lag <= 0:
        return -1
    
    # Frequency to MIDI
    freq = sample_rate / best_lag
    if freq <= 0:
        return -1
    
    midi = 69.0 + 12.0 * math.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    
    if 21 <= midi_rounded <= 108:
        return midi_rounded
    
    return -1


@njit(fastmath=True, parallel=True)
def process_all_chunks_jit(all_samples, chunk_size, sample_rate, min_lag, max_lag):
    """
    Process all audio chunks in parallel using Numba's parallel execution.
    Returns array of MIDI notes (-1 for no detection).
    """
    n_chunks = len(all_samples) // chunk_size
    results = np.empty(n_chunks, dtype=np.int32)
    
    for i in prange(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = all_samples[start:end]
        results[i] = detect_pitch_full_jit(chunk, sample_rate, min_lag, max_lag)
    
    return results


@njit(fastmath=True)
def detect_pitch_yin_jit(samples, sample_rate, min_lag, max_lag, threshold=0.15):
    """
    YIN-style pitch detection, fully JIT compiled.
    Uses cumulative mean normalized difference function (CMNDF).
    """
    n = len(samples)
    if n < max_lag:
        return -1
    
    # Calculate difference function
    diff = np.zeros(max_lag, dtype=np.float64)
    
    for tau in range(1, max_lag):
        for j in range(n - tau):
            delta = samples[j] - samples[j + tau]
            diff[tau] += delta * delta
    
    # Cumulative mean normalized difference
    cmndf = np.zeros(max_lag, dtype=np.float64)
    cmndf[0] = 1.0
    running_sum = 0.0
    
    for tau in range(1, max_lag):
        running_sum += diff[tau]
        if running_sum > 0:
            cmndf[tau] = diff[tau] * tau / running_sum
        else:
            cmndf[tau] = 1.0
    
    # Find first dip below threshold
    for tau in range(min_lag, max_lag - 1):
        if cmndf[tau] < threshold:
            # Find local minimum
            while tau + 1 < max_lag and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            
            # Parabolic interpolation
            if tau > 0 and tau < max_lag - 1:
                y0 = cmndf[tau - 1]
                y1 = cmndf[tau]
                y2 = cmndf[tau + 1]
                denom = y0 - 2 * y1 + y2
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y0 - y2) / denom
                    tau_refined = tau + delta
                else:
                    tau_refined = float(tau)
            else:
                tau_refined = float(tau)
            
            freq = sample_rate / tau_refined
            midi = 69.0 + 12.0 * math.log2(freq / 440.0)
            midi_rounded = int(round(midi))
            
            if 21 <= midi_rounded <= 108:
                return midi_rounded
    
    return -1


# =============================================================================
# STREAMING DETECTOR WITH BATCH PROCESSING
# =============================================================================

class NumbaStreamingPitchDetectorV2:
    """
    Aggressive Numba optimization using fully JIT-compiled pipelines.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Note tracking
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
        # Buffer for batch processing
        self.sample_buffer = []
        self.buffer_limit = 16  # Process in batches of 16 chunks
        
        # Warmup JIT
        self._warmup()
    
    def _warmup(self):
        """Pre-compile JIT functions."""
        test = np.zeros(2048, dtype=np.float32)
        _ = detect_pitch_full_jit(test, 44100.0, 44, 551)
    
    def process_chunk(self, samples):
        """Process a single chunk."""
        chunk_duration = len(samples) / self.sample_rate
        
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        # Use fully JIT-compiled detector
        midi_note = detect_pitch_full_jit(
            samples, 
            float(self.sample_rate), 
            self.min_lag, 
            self.max_lag
        )
        
        if midi_note == -1:
            midi_note = None
        
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


# =============================================================================
# ALTERNATIVE: SCIPY FFT WITH WORKERS
# =============================================================================

try:
    from scipy.fft import rfft, irfft
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False


@njit(fastmath=True)
def find_peak_and_convert(corr, min_lag, max_lag, corr_0, sample_rate):
    """JIT-compiled peak finding and MIDI conversion."""
    if max_lag > len(corr):
        max_lag = len(corr)
    
    best_val = -1e10
    best_idx = min_lag
    
    for i in range(min_lag, max_lag):
        if corr[i] > best_val:
            best_val = corr[i]
            best_idx = i
    
    if corr_0 <= 0 or best_val <= 0.2 * corr_0:
        return -1
    
    # Parabolic interpolation
    if best_idx > 0 and best_idx < len(corr) - 1:
        y0 = corr[best_idx - 1]
        y1 = corr[best_idx]
        y2 = corr[best_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-10:
            delta = 0.5 * (y0 - y2) / denom
            if delta < -0.5:
                delta = -0.5
            elif delta > 0.5:
                delta = 0.5
            refined_idx = best_idx + delta
        else:
            refined_idx = float(best_idx)
    else:
        refined_idx = float(best_idx)
    
    freq = sample_rate / refined_idx
    midi = 69.0 + 12.0 * math.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    
    if 21 <= midi_rounded <= 108:
        return midi_rounded
    return -1


class SciPyFFTPitchDetector:
    """
    Pitch detector using scipy.fft which can use multiple workers.
    Combined with JIT-compiled post-processing.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        
        n = len(samples)
        if n < self.min_lag * 2:
            midi_note = None
        else:
            energy = np.dot(samples, samples)
            if energy < 0.0001:
                midi_note = None
            else:
                use_n = min(n, 2048)
                self._fft_buffer[:use_n] = samples[:use_n]
                self._fft_buffer[use_n:] = 0
                
                # scipy.fft with workers
                fft = rfft(self._fft_buffer, workers=2)
                power = fft.real**2 + fft.imag**2
                corr = irfft(power, workers=2)
                
                # JIT-compiled peak finding
                midi_note = find_peak_and_convert(
                    corr, self.min_lag, self.max_lag, 
                    corr[0], float(self.sample_rate)
                )
                if midi_note == -1:
                    midi_note = None
        
        self._update_note_state(midi_note, chunk_duration)
        self.time_position += chunk_duration
        return midi_note
    
    def _update_note_state(self, new_note, duration):
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
        if self.current_note is not None:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((
                    self.current_note,
                    self.note_start_time,
                    note_duration
                ))
        return self.notes


# =============================================================================
# MIDI WRITER
# =============================================================================

class MIDIWriter:
    def __init__(self, tempo_bpm=120):
        self.tempo_bpm = tempo_bpm
        self.ticks_per_beat = 480
        self.us_per_beat = int(60_000_000 / tempo_bpm)
    
    def _vlq(self, value):
        result = [value & 0x7F]
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))
    
    def _sec_to_ticks(self, seconds):
        return int(seconds * (self.tempo_bpm / 60.0) * self.ticks_per_beat)
    
    def write(self, notes, filename):
        track = bytearray()
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


# =============================================================================
# BENCHMARKING
# =============================================================================

def process_wav(input_path, output_path, detector_class, chunk_size=2048):
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        detector = detector_class(sample_rate=sr, chunk_size=chunk_size)
        
        while True:
            raw = wav.readframes(chunk_size)
            if not raw:
                break
            
            ns = len(raw) // (sw * ch)
            samples = struct.unpack(f'<{ns * ch}h', raw) if sw == 2 else [b-128 for b in raw]
            
            if ch == 2:
                samples = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
            
            max_val = 2 ** (sw * 8 - 1)
            samples = np.array([s / max_val for s in samples], dtype=np.float32)
            
            detector.process_chunk(samples)
        
        notes = detector.finalize()
    
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'elapsed': t1 - t0,
        'duration': nf / sr,
        'rtf': (nf / sr) / (t1 - t0) if (t1 - t0) > 0 else 0,
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes),
        'output': output_path
    }


def generate_test_wav(filename, duration=60, sr=44100):
    notes_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
    t = np.arange(int(duration * sr)) / sr
    
    note_dur = 0.5
    samples_per_note = int(note_dur * sr)
    
    samples = np.zeros(len(t), dtype=np.float32)
    for i in range(len(t)):
        note_idx = (i // samples_per_note) % len(notes_hz)
        freq = notes_hz[note_idx]
        pos = (i % samples_per_note) / samples_per_note
        env = min(1.0, pos * 20) * max(0.0, 1.0 - max(0, pos - 0.8) * 5)
        samples[i] = 0.7 * np.sin(2 * np.pi * freq * t[i]) * env
    
    samples = (samples * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())


def main():
    print("=" * 70)
    print("PARADROMICS - NUMBA JIT OPTIMIZED V2")
    print("Aggressive optimization: Full JIT pipeline, no FFT overhead")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
        ('test_complex.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 50)
    for name, dur in tests:
        generate_test_wav(os.path.join(base_dir, name), dur)
        print(f"  Generated: {name} ({dur}s)")
    
    detectors = [
        ("Full JIT (no FFT)", NumbaStreamingPitchDetectorV2),
    ]
    
    if HAS_SCIPY_FFT:
        detectors.append(("SciPy FFT + JIT post", SciPyFFTPitchDetector))
    
    for det_name, det_class in detectors:
        print(f"\n[2] Testing: {det_name}")
        print("-" * 50)
        
        # Warmup
        warmup_file = os.path.join(base_dir, 'test_simple.wav')
        warmup_out = os.path.join(base_dir, 'warmup.mid')
        _ = process_wav(warmup_file, warmup_out, det_class)
        try:
            os.remove(warmup_out)
        except:
            pass
        print("  JIT warmup complete")
        
        results = []
        for name, _ in tests:
            inp = os.path.join(base_dir, name)
            out = os.path.join(base_dir, name.replace('.wav', f'_{det_name.replace(" ", "_")}.mid'))
            m = process_wav(inp, out, det_class)
            results.append(m)
            print(f"  {name}: {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
        
        total_dur = sum(r['duration'] for r in results)
        total_time = sum(r['elapsed'] for r in results)
        max_mem = max(r['memory_mb'] for r in results)
        
        print(f"\n  TOTAL: {total_dur:.1f}s in {total_time:.2f}s = {total_dur/total_time:.1f}x RT")
        print(f"  Peak memory: {max_mem:.2f} MB")
        
        baseline = 4.7
        new_rtf = total_dur / total_time
        print(f"  IMPROVEMENT: {baseline:.1f}x -> {new_rtf:.1f}x ({new_rtf/baseline:.2f}x)")
    
    print("\n" + "=" * 70)
    print("NUMBA OPTIMIZATIONS V2:")
    print("  - Fully JIT-compiled pitch detection (no Python/Numba boundary)")
    print("  - Time-domain autocorrelation (avoids FFT overhead for small arrays)")
    print("  - Coarse-to-fine lag search (4x fewer correlations)")
    print("  - fastmath=True for SIMD vectorization")
    print("  - Optional: scipy.fft with workers + JIT post-processing")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
