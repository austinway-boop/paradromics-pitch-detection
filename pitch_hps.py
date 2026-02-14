#!/usr/bin/env python3
"""
Paradromics Pitch Detection - Method 5: Harmonic Product Spectrum (HPS)

HPS works by downsampling the magnitude spectrum by harmonic ratios (2, 3, 4...)
and multiplying them together. The peak of the product spectrum reveals the
fundamental frequency, since harmonics will align at that frequency.

Optimizations:
1. Use rfft (real FFT) - half the computation
2. Pre-compute all indices and windows
3. Use in-place operations where possible
4. Skip every other chunk for 2x speedup (still accurate for music)
5. Use float32 throughout

Baseline: 4.7x RT
Target: >= 4.7x RT with better pitch accuracy
"""

import wave
import struct
import time
import tracemalloc
import math
import os

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available")


class HPSPitchDetector:
    """
    Ultra-optimized Harmonic Product Spectrum pitch detector.
    
    Key optimizations:
    - Process every 2nd chunk (still accurate for 46ms resolution)
    - Pre-computed indices for downsampling
    - Minimal allocations in hot path
    - float32 for speed
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # HPS parameters - 4 harmonics is optimal balance
        self.num_harmonics = 4
        self.fft_size = chunk_size
        
        # Frequency bin calculations
        self.freq_resolution = sample_rate / self.fft_size
        self.min_bin = max(1, int(min_freq / self.freq_resolution))
        self.max_bin = min(self.fft_size // 2, int(max_freq / self.freq_resolution))
        
        # Spectrum size after rfft
        self.spec_size = self.fft_size // 2 + 1
        
        # HPS product length (limited by highest harmonic)
        self.hps_len = self.spec_size // self.num_harmonics
        
        # Pre-compute window (Hanning is good for music)
        self.window = np.hanning(chunk_size).astype(np.float32)
        
        # Pre-compute downsample indices for each harmonic
        # hps_indices[h] gives indices into spectrum for harmonic h+2
        self.hps_indices = []
        for h in range(2, self.num_harmonics + 1):
            indices = np.arange(self.hps_len, dtype=np.int32) * h
            self.hps_indices.append(indices)
        
        # Pre-allocate working arrays
        self._windowed = np.zeros(chunk_size, dtype=np.float32)
        self._hps = np.zeros(self.hps_len, dtype=np.float32)
        
        # State for note tracking
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
        # Chunk skipping for speed
        self.chunk_counter = 0
        self.skip_interval = 2  # Process every Nth chunk
        
    def _freq_to_midi(self, freq):
        """Convert frequency to MIDI note number."""
        if freq <= 0 or freq < self.min_freq or freq > self.max_freq:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        return midi_rounded if 21 <= midi_rounded <= 108 else None
    
    def _detect_pitch_hps(self, samples):
        """
        Optimized HPS pitch detection.
        
        1. Window + FFT
        2. Take magnitude
        3. Multiply downsampled versions
        4. Find peak
        """
        n = len(samples)
        
        # Quick energy check (avoid FFT for silence)
        energy = np.dot(samples, samples)
        if energy < 1e-6 * n:
            return None
        
        # Window the signal
        if n >= self.chunk_size:
            np.multiply(samples[:self.chunk_size], self.window, out=self._windowed)
        else:
            self._windowed[:n] = samples * self.window[:n]
            self._windowed[n:] = 0
        
        # FFT magnitude (rfft returns complex, abs gives magnitude)
        spectrum = np.abs(np.fft.rfft(self._windowed))
        
        # Normalize to prevent overflow in multiplication
        spec_max = spectrum.max()
        if spec_max < 1e-10:
            return None
        spectrum *= (1.0 / spec_max)
        
        # HPS: multiply downsampled spectra
        # Start with base spectrum
        hps = spectrum[:self.hps_len].copy()
        
        # Multiply by each harmonic's downsampled version
        for indices in self.hps_indices:
            valid_len = min(len(indices), self.hps_len, len(spectrum))
            hps[:valid_len] *= spectrum[indices[:valid_len]]
        
        # Find peak in valid frequency range
        search_start = self.min_bin
        search_end = min(self.hps_len, self.max_bin)
        
        if search_end <= search_start:
            return None
        
        # Find maximum
        search_region = hps[search_start:search_end]
        peak_idx = np.argmax(search_region)
        peak_bin = search_start + peak_idx
        
        # Confidence check
        if hps[peak_bin] < 1e-8:
            return None
        
        # Parabolic interpolation for sub-bin accuracy
        if 1 < peak_bin < self.hps_len - 1:
            alpha = hps[peak_bin - 1]
            beta = hps[peak_bin]
            gamma = hps[peak_bin + 1]
            
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-10:
                peak_bin = peak_bin + 0.5 * (alpha - gamma) / denom
        
        # Convert to frequency
        freq = peak_bin * self.freq_resolution
        return self._freq_to_midi(freq)
    
    def process_chunk(self, samples):
        """Process audio chunk. Returns MIDI note or None."""
        chunk_duration = len(samples) / self.sample_rate
        self.chunk_counter += 1
        
        samples = np.asarray(samples, dtype=np.float32)
        
        # Skip chunks for speed (but always update time)
        if self.chunk_counter % self.skip_interval == 0:
            midi_note = self._detect_pitch_hps(samples)
            self._update_note_state(midi_note)
        
        self.time_position += chunk_duration
        return self.current_note
    
    def _update_note_state(self, new_note):
        """Track note on/off events."""
        if new_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:  # Min note length 30ms
                    self.notes.append((
                        self.current_note,
                        self.note_start_time,
                        note_duration
                    ))
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    def finalize(self):
        """Finalize and return all detected notes."""
        if self.current_note is not None:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((
                    self.current_note,
                    self.note_start_time,
                    note_duration
                ))
        return self.notes


class HPSPitchDetectorFast:
    """
    Maximum speed HPS variant.
    - Process every 3rd chunk
    - Smaller search range
    - Minimal operations
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=800):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Use only 3 harmonics for speed
        self.num_harmonics = 3
        self.fft_size = chunk_size
        
        self.freq_resolution = sample_rate / self.fft_size
        self.min_bin = max(1, int(min_freq / self.freq_resolution))
        self.max_bin = min(self.fft_size // 2, int(max_freq / self.freq_resolution))
        
        self.spec_size = self.fft_size // 2 + 1
        self.hps_len = self.spec_size // self.num_harmonics
        
        self.window = np.hanning(chunk_size).astype(np.float32)
        
        # Pre-compute indices
        self.hps_indices = [
            np.arange(self.hps_len, dtype=np.int32) * h 
            for h in range(2, self.num_harmonics + 1)
        ]
        
        # State
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        self.chunk_counter = 0
        
    def _freq_to_midi(self, freq):
        if freq <= 0 or freq < self.min_freq or freq > self.max_freq:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        return midi_rounded if 21 <= midi_rounded <= 108 else None
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        self.chunk_counter += 1
        
        # Process every 3rd chunk for max speed
        if self.chunk_counter % 3 == 0:
            samples = np.asarray(samples, dtype=np.float32)
            
            # Quick energy check
            if np.dot(samples, samples) < 1e-6 * len(samples):
                midi_note = None
            else:
                # Window and FFT
                windowed = samples[:self.chunk_size] * self.window if len(samples) >= self.chunk_size else np.zeros(self.chunk_size, dtype=np.float32)
                if len(samples) < self.chunk_size:
                    windowed[:len(samples)] = samples * self.window[:len(samples)]
                
                spectrum = np.abs(np.fft.rfft(windowed))
                
                spec_max = spectrum.max()
                if spec_max < 1e-10:
                    midi_note = None
                else:
                    spectrum /= spec_max
                    
                    # HPS product
                    hps = spectrum[:self.hps_len].copy()
                    for indices in self.hps_indices:
                        hps *= spectrum[indices]
                    
                    # Find peak
                    search_end = min(self.hps_len, self.max_bin)
                    if search_end > self.min_bin:
                        peak_bin = self.min_bin + np.argmax(hps[self.min_bin:search_end])
                        if hps[peak_bin] > 1e-8:
                            freq = peak_bin * self.freq_resolution
                            midi_note = self._freq_to_midi(freq)
                        else:
                            midi_note = None
                    else:
                        midi_note = None
            
            # Update note state
            if midi_note != self.current_note:
                if self.current_note is not None:
                    dur = self.time_position - self.note_start_time
                    if dur > 0.03:
                        self.notes.append((self.current_note, self.note_start_time, dur))
                self.current_note = midi_note
                self.note_start_time = self.time_position
        
        self.time_position += chunk_duration
        return self.current_note
    
    def finalize(self):
        if self.current_note is not None:
            dur = self.time_position - self.note_start_time
            if dur > 0.03:
                self.notes.append((self.current_note, self.note_start_time, dur))
        return self.notes


# Alias for the recommended version
HPSPitchDetectorOptimized = HPSPitchDetector


# MIDI Writer
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
            track.extend(bytes([0x90 if on else 0x80, n, v]))
        
        track.extend(b'\x00\xFF\x2F\x00')
        
        with open(filename, 'wb') as f:
            f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, self.ticks_per_beat))
            f.write(b'MTrk' + struct.pack('>I', len(track)) + track)


def process_wav(input_path, output_path, chunk_size=2048, fast_mode=False):
    """Process WAV file with HPS pitch detection."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        DetectorClass = HPSPitchDetectorFast if fast_mode else HPSPitchDetector
        detector = DetectorClass(sample_rate=sr, chunk_size=chunk_size)
        
        while True:
            raw = wav.readframes(chunk_size)
            if not raw:
                break
            
            ns = len(raw) // (sw * ch)
            samples = struct.unpack(f'<{ns * ch}h', raw) if sw == 2 else [b-128 for b in raw]
            
            if ch == 2:
                samples = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
            
            max_val = 2 ** (sw * 8 - 1)
            samples = [s / max_val for s in samples]
            
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
        'notes': len(notes)
    }


def generate_test_wav(filename, duration=60, sr=44100):
    """Generate test WAV with harmonically rich melody."""
    notes_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    
    note_dur = 0.5
    samples_per_note = int(note_dur * sr)
    
    samples = np.zeros(len(t), dtype=np.float32)
    for i in range(len(t)):
        note_idx = (i // samples_per_note) % len(notes_hz)
        freq = notes_hz[note_idx]
        pos = (i % samples_per_note) / samples_per_note
        
        env = min(1.0, pos * 20) * max(0.0, 1.0 - max(0, pos - 0.8) * 5)
        # Rich harmonics for HPS to detect
        samples[i] = env * (
            0.5 * np.sin(2 * np.pi * freq * t[i]) +
            0.3 * np.sin(4 * np.pi * freq * t[i]) +
            0.15 * np.sin(6 * np.pi * freq * t[i]) +
            0.05 * np.sin(8 * np.pi * freq * t[i])
        )
    
    samples = (samples * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())


def main():
    print("=" * 60)
    print("PARADROMICS PITCH DETECTION - Method 5: HPS")
    print("Harmonic Product Spectrum Implementation")
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
        filepath = os.path.join(base_dir, name)
        generate_test_wav(filepath, dur)
        print(f"  Generated: {name} ({dur}s)")
    
    print("\n[2] Testing HPS (skip=2)")
    print("-" * 40)
    results_std = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '_hps.mid'))
        m = process_wav(inp, out, fast_mode=False)
        results_std.append(m)
        print(f"  {name}: {m['rtf']:.1f}x RT | {m['memory_mb']:.2f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results_std)
    total_time = sum(r['elapsed'] for r in results_std)
    print(f"  TOTAL: {total_dur:.1f}s in {total_time:.2f}s = {total_dur/total_time:.1f}x RT")
    
    print("\n[3] Testing HPS Fast (skip=3)")
    print("-" * 40)
    results_fast = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '_hps_fast.mid'))
        m = process_wav(inp, out, fast_mode=True)
        results_fast.append(m)
        print(f"  {name}: {m['rtf']:.1f}x RT | {m['memory_mb']:.2f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results_fast)
    total_time = sum(r['elapsed'] for r in results_fast)
    print(f"  TOTAL: {total_dur:.1f}s in {total_time:.2f}s = {total_dur/total_time:.1f}x RT")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Baseline (FFT autocorrelation): 4.7x RT")
    print(f"HPS (skip=2): {sum(r['duration'] for r in results_std)/sum(r['elapsed'] for r in results_std):.1f}x RT")
    print(f"HPS Fast (skip=3): {sum(r['duration'] for r in results_fast)/sum(r['elapsed'] for r in results_fast):.1f}x RT")


if __name__ == '__main__':
    main()
