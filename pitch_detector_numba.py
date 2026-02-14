#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
NUMBA JIT-OPTIMIZED VERSION

Optimizations applied:
1. Numba @jit decorators on hot loops
2. JIT-compiled peak finding
3. JIT-compiled frequency-to-MIDI conversion
4. Numpy FFT with JIT post-processing
5. nopython mode where possible for maximum speed
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

import numpy as np
from numba import jit, njit, prange
from numba import float32, float64, int32, int64

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


# =============================================================================
# NUMBA JIT-COMPILED HOT FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_energy_jit(samples):
    """JIT-compiled energy computation using dot product."""
    energy = 0.0
    for i in range(len(samples)):
        energy += samples[i] * samples[i]
    return energy


@njit(cache=True, fastmath=True)
def find_best_peak_jit(corr, min_lag, max_lag, threshold_ratio=0.2):
    """
    JIT-compiled peak finding in autocorrelation array.
    Returns (best_lag, peak_value) or (-1, 0.0) if no valid peak.
    """
    if max_lag > len(corr):
        max_lag = len(corr)
    if min_lag >= max_lag:
        return -1, 0.0
    
    best_val = corr[min_lag]
    best_idx = min_lag
    
    # Find maximum in search range
    for i in range(min_lag + 1, max_lag):
        if corr[i] > best_val:
            best_val = corr[i]
            best_idx = i
    
    # Check if peak is significant enough
    if corr[0] > 0 and best_val > threshold_ratio * corr[0]:
        return best_idx, best_val
    
    return -1, 0.0


@njit(cache=True, fastmath=True)
def freq_to_midi_jit(freq):
    """JIT-compiled frequency to MIDI note conversion."""
    if freq <= 0:
        return -1
    midi = 69.0 + 12.0 * math.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    if 21 <= midi_rounded <= 108:
        return midi_rounded
    return -1


@njit(cache=True, fastmath=True)
def prepare_fft_buffer_jit(samples, fft_buffer, use_n):
    """JIT-compiled FFT buffer preparation with zero-padding."""
    n = len(samples)
    actual_n = min(n, use_n)
    
    # Copy samples to buffer
    for i in range(actual_n):
        fft_buffer[i] = samples[i]
    
    # Zero-pad the rest
    for i in range(actual_n, len(fft_buffer)):
        fft_buffer[i] = 0.0


@njit(cache=True, fastmath=True)
def compute_power_spectrum_jit(fft_real, fft_imag, power_out):
    """JIT-compiled power spectrum computation."""
    n = len(fft_real)
    for i in range(n):
        power_out[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]


@njit(cache=True, fastmath=True, parallel=True)
def process_samples_batch_jit(samples_batch, sample_rate, min_lag, max_lag):
    """
    JIT-compiled batch processing of multiple sample chunks.
    Returns array of detected frequencies (or -1 for no detection).
    """
    n_chunks = len(samples_batch)
    results = np.empty(n_chunks, dtype=np.int32)
    
    for chunk_idx in prange(n_chunks):
        samples = samples_batch[chunk_idx]
        n = len(samples)
        
        # Energy check
        energy = 0.0
        for i in range(n):
            energy += samples[i] * samples[i]
        
        if energy < 0.0001:
            results[chunk_idx] = -1
            continue
        
        # Simple time-domain autocorrelation for small arrays
        # (FFT is done outside JIT due to np.fft limitations)
        best_lag = -1
        best_corr = -1.0
        
        for lag in range(min_lag, min(max_lag, n)):
            corr = 0.0
            for i in range(n - lag):
                corr += samples[i] * samples[i + lag]
            
            norm_corr = corr / (energy + 1e-10)
            if norm_corr > best_corr:
                best_corr = norm_corr
                best_lag = lag
        
        if best_lag > 0 and best_corr > 0.3:
            freq = sample_rate / best_lag
            results[chunk_idx] = freq_to_midi_jit(freq)
        else:
            results[chunk_idx] = -1
    
    return results


@njit(cache=True, fastmath=True)
def parabolic_interpolation_jit(corr, peak_idx):
    """
    JIT-compiled parabolic interpolation for sub-sample peak refinement.
    Returns refined peak position.
    """
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return float(peak_idx)
    
    y0 = corr[peak_idx - 1]
    y1 = corr[peak_idx]
    y2 = corr[peak_idx + 1]
    
    # Parabolic interpolation formula
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-10:
        return float(peak_idx)
    
    delta = 0.5 * (y0 - y2) / denom
    
    # Clamp delta to [-0.5, 0.5]
    if delta < -0.5:
        delta = -0.5
    elif delta > 0.5:
        delta = 0.5
    
    return float(peak_idx) + delta


# =============================================================================
# OPTIMIZED PITCH DETECTOR CLASS
# =============================================================================

class NumbaStreamingPitchDetector:
    """
    Numba-optimized pitch detector using FFT autocorrelation
    with JIT-compiled hot loops.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Pre-allocate arrays (float32 for speed)
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self._power_buffer = np.zeros(self.fft_size // 2 + 1, dtype=np.float32)
        
        # Note tracking state
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
        # Warm up JIT functions (first call compiles them)
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Pre-compile JIT functions on first use."""
        test_samples = np.zeros(512, dtype=np.float32)
        _ = compute_energy_jit(test_samples)
        _ = freq_to_midi_jit(440.0)
        test_corr = np.zeros(100, dtype=np.float64)
        _ = find_best_peak_jit(test_corr, 10, 90)
        _ = parabolic_interpolation_jit(test_corr, 50)
    
    def _detect_pitch_numba(self, samples):
        """
        FFT-based autocorrelation with Numba-optimized post-processing.
        """
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # JIT-compiled energy check
        energy = compute_energy_jit(samples)
        if energy < 0.0001:
            return None
        
        # Prepare FFT buffer (JIT-compiled)
        use_n = min(n, 2048)
        prepare_fft_buffer_jit(samples, self._fft_buffer, use_n)
        
        # FFT-based autocorrelation (numpy FFT is already optimized via FFTPACK/MKL)
        fft = np.fft.rfft(self._fft_buffer)
        
        # Power spectrum computation
        power = fft.real * fft.real + fft.imag * fft.imag
        
        # Inverse FFT for autocorrelation
        corr = np.fft.irfft(power)
        
        # JIT-compiled peak finding
        best_lag, peak_val = find_best_peak_jit(corr, self.min_lag, self.max_lag, 0.2)
        
        if best_lag > 0:
            # JIT-compiled parabolic interpolation for sub-sample accuracy
            refined_lag = parabolic_interpolation_jit(corr, best_lag)
            freq = self.sample_rate / refined_lag
            return freq_to_midi_jit(freq)
        
        return None
    
    def process_chunk(self, samples):
        """Process a chunk of audio samples."""
        chunk_duration = len(samples) / self.sample_rate
        
        # Ensure numpy float32 array
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        midi_note = self._detect_pitch_numba(samples)
        
        # Handle -1 return from JIT function
        if midi_note is not None and midi_note == -1:
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
# MIDI WRITER (unchanged from original)
# =============================================================================

class MIDIWriter:
    """MIDI file writer."""
    
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


# =============================================================================
# PROCESSING AND BENCHMARKING
# =============================================================================

def process_wav(input_path, output_path, chunk_size=2048):
    """Process WAV file with Numba-optimized streaming pitch detection."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz/{sw*8}bit")
        
        detector = NumbaStreamingPitchDetector(sample_rate=sr, chunk_size=chunk_size)
        
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
    """Generate test WAV with melody."""
    notes_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
    t = np.arange(int(duration * sr)) / sr
    
    note_dur = 0.5
    samples_per_note = int(note_dur * sr)
    
    samples = np.zeros(len(t), dtype=np.float32)
    for i in range(len(t)):
        note_idx = (i // samples_per_note) % len(notes_hz)
        freq = notes_hz[note_idx]
        pos_in_note = (i % samples_per_note) / samples_per_note
        
        env = min(1.0, pos_in_note * 20) * max(0.0, 1.0 - max(0, pos_in_note - 0.8) * 5)
        samples[i] = 0.7 * np.sin(2 * np.pi * freq * t[i]) * env
    
    samples = (samples * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())
    
    print(f"  Generated: {os.path.basename(filename)} ({duration}s)")


def main():
    print("=" * 60)
    print("PARADROMICS - NUMBA JIT OPTIMIZED VERSION")
    print("Method 6: FFT Autocorrelation + Numba @jit Decorators")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
        ('test_complex.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 40)
    for name, dur in tests:
        generate_test_wav(os.path.join(base_dir, name), dur)
    
    # Warmup run (JIT compilation happens on first call)
    print("\n[2] JIT Warmup (compiling Numba functions...)")
    print("-" * 40)
    warmup_file = os.path.join(base_dir, 'test_simple.wav')
    warmup_out = os.path.join(base_dir, 'warmup.mid')
    _ = process_wav(warmup_file, warmup_out)
    os.remove(warmup_out)
    print("  JIT compilation complete!")
    
    print("\n[3] Processing Audio Files (JIT-optimized)")
    print("-" * 40)
    
    results = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '_numba.mid'))
        m = process_wav(inp, out)
        results.append(m)
        print(f"    -> {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    max_mem = max(r['memory_mb'] for r in results)
    
    print("\n" + "=" * 60)
    print("NUMBA JIT RESULTS")
    print("=" * 60)
    print(f"Total audio: {total_dur:.1f}s processed in {total_time:.2f}s")
    print(f"Speed: {total_dur/total_time:.1f}x real-time")
    print(f"Peak memory: {max_mem:.2f} MB")
    
    print("\nNUMBA OPTIMIZATIONS APPLIED:")
    print("  - @njit(cache=True, fastmath=True) on hot loops")
    print("  - JIT-compiled energy computation")
    print("  - JIT-compiled peak finding")
    print("  - JIT-compiled frequency-to-MIDI conversion")
    print("  - JIT-compiled parabolic interpolation")
    print("  - Function caching for subsequent runs")
    
    baseline_rtf = 4.7
    new_rtf = total_dur / total_time
    improvement = new_rtf / baseline_rtf
    print(f"\nIMPROVEMENT: {baseline_rtf:.1f}x -> {new_rtf:.1f}x RT ({improvement:.2f}x speedup)")
    
    return results


if __name__ == '__main__':
    main()
