#!/usr/bin/env python3
"""
Paradromics Optimization - Method 11: Scipy Signal Processing

Uses scipy.signal.correlate with method='fft' instead of manual FFT.
Testing if scipy's optimized routines are faster than manual np.fft.rfft approach.

Current baseline: ~4.7x RT
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

import numpy as np
from scipy import signal
from scipy.signal import correlate, find_peaks

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class ScipyPitchDetector:
    """Pitch detector using scipy.signal.correlate with method='fft'."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch_scipy_correlate(self, samples):
        """Use scipy.signal.correlate with method='fft' for autocorrelation."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Quick energy check
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # scipy.signal.correlate with method='fft'
        # mode='full' gives correlation at all lags
        # We want autocorrelation, so correlate signal with itself
        corr = correlate(samples, samples, mode='full', method='fft')
        
        # Take only positive lags (second half)
        mid = len(corr) // 2
        corr = corr[mid:]  # Now corr[0] is lag 0, corr[1] is lag 1, etc.
        
        # Search in lag range
        if self.max_lag > len(corr):
            return None
            
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
            
        best_idx = np.argmax(search)
        
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
        midi_note = self._detect_pitch_scipy_correlate(samples)
        
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


class ScipyPitchDetectorV2:
    """Alternative: Use scipy.signal.fftconvolve for potentially faster autocorr."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch_fftconvolve(self, samples):
        """Use scipy.signal.fftconvolve for autocorrelation."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Quick energy check
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # fftconvolve with reversed signal = correlation
        # This is mathematically equivalent but may be faster
        corr = signal.fftconvolve(samples, samples[::-1], mode='full')
        
        # Take only positive lags
        mid = len(corr) // 2
        corr = corr[mid:]
        
        if self.max_lag > len(corr):
            return None
            
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
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        samples = np.asarray(samples, dtype=np.float32)
        midi_note = self._detect_pitch_fftconvolve(samples)
        self._update_note_state(midi_note, chunk_duration)
        self.time_position += chunk_duration
        return midi_note
    
    def _update_note_state(self, new_note, duration):
        if new_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:
                    self.notes.append((self.current_note, self.note_start_time, note_duration))
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    def finalize(self):
        if self.current_note is not None:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((self.current_note, self.note_start_time, note_duration))
        return self.notes


class ManualFFTPitchDetector:
    """Original manual FFT approach for comparison."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Pre-allocate FFT arrays
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch_manual_fft(self, samples):
        """Original manual FFT-based autocorrelation."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        use_n = min(n, 2048)
        self._fft_buffer[:use_n] = samples[:use_n]
        self._fft_buffer[use_n:] = 0
        
        fft = np.fft.rfft(self._fft_buffer)
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
        if freq <= 0:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def process_chunk(self, samples):
        chunk_duration = len(samples) / self.sample_rate
        samples = np.asarray(samples, dtype=np.float32)
        midi_note = self._detect_pitch_manual_fft(samples)
        self._update_note_state(midi_note, chunk_duration)
        self.time_position += chunk_duration
        return midi_note
    
    def _update_note_state(self, new_note, duration):
        if new_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:
                    self.notes.append((self.current_note, self.note_start_time, note_duration))
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    def finalize(self):
        if self.current_note is not None:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((self.current_note, self.note_start_time, note_duration))
        return self.notes


class MIDIWriter:
    """MIDI file writer."""
    
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


def benchmark_detector(detector_class, detector_name, input_path, output_path, chunk_size=2048):
    """Benchmark a pitch detector."""
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
            samples = struct.unpack(f'<{ns * ch}h', raw)
            
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
        'name': detector_name,
        'elapsed': elapsed,
        'duration': duration,
        'rtf': duration / elapsed if elapsed > 0 else 0,
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes)
    }


def main():
    print("=" * 70)
    print("PARADROMICS OPTIMIZATION - Method 11: Scipy Signal Processing")
    print("Comparing scipy.signal.correlate vs manual FFT autocorrelation")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test files
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
    ]
    
    # Generate test files if needed
    for name, dur in tests:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            print(f"Generating {name}...")
            generate_test_wav(path, dur)
    
    # Detectors to benchmark
    detectors = [
        (ManualFFTPitchDetector, "Manual FFT (baseline)"),
        (ScipyPitchDetector, "scipy.correlate(method='fft')"),
        (ScipyPitchDetectorV2, "scipy.fftconvolve"),
    ]
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    all_results = {}
    
    for name, dur in tests:
        input_path = os.path.join(base_dir, name)
        print(f"\n{name} ({dur}s):")
        print("-" * 50)
        
        for detector_class, detector_name in detectors:
            output_path = os.path.join(base_dir, f"out_{detector_name.split()[0].lower()}_{name.replace('.wav', '.mid')}")
            
            result = benchmark_detector(detector_class, detector_name, input_path, output_path)
            
            if detector_name not in all_results:
                all_results[detector_name] = []
            all_results[detector_name].append(result)
            
            print(f"  {detector_name:35s} | {result['rtf']:6.2f}x RT | {result['elapsed']:.3f}s | {result['memory_mb']:.1f}MB | {result['notes']} notes")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (Combined)")
    print("=" * 70)
    
    for detector_name, results in all_results.items():
        total_dur = sum(r['duration'] for r in results)
        total_time = sum(r['elapsed'] for r in results)
        max_mem = max(r['memory_mb'] for r in results)
        rtf = total_dur / total_time if total_time > 0 else 0
        
        print(f"  {detector_name:35s} | {rtf:6.2f}x RT | {total_time:.3f}s total | {max_mem:.1f}MB peak")
    
    # Comparison
    baseline = all_results.get("Manual FFT (baseline)", [])
    scipy_corr = all_results.get("scipy.correlate(method='fft')", [])
    scipy_conv = all_results.get("scipy.fftconvolve", [])
    
    if baseline and scipy_corr:
        base_rtf = sum(r['duration'] for r in baseline) / sum(r['elapsed'] for r in baseline)
        scipy_rtf = sum(r['duration'] for r in scipy_corr) / sum(r['elapsed'] for r in scipy_corr)
        
        print(f"\n  scipy.correlate vs Manual FFT: {scipy_rtf/base_rtf:.2f}x relative speed")
        
        if scipy_rtf > base_rtf:
            print(f"  ✓ scipy.correlate is {(scipy_rtf/base_rtf - 1)*100:.1f}% FASTER")
        else:
            print(f"  ✗ scipy.correlate is {(1 - scipy_rtf/base_rtf)*100:.1f}% SLOWER")
    
    if baseline and scipy_conv:
        base_rtf = sum(r['duration'] for r in baseline) / sum(r['elapsed'] for r in baseline)
        conv_rtf = sum(r['duration'] for r in scipy_conv) / sum(r['elapsed'] for r in scipy_conv)
        
        print(f"  scipy.fftconvolve vs Manual FFT: {conv_rtf/base_rtf:.2f}x relative speed")
        
        if conv_rtf > base_rtf:
            print(f"  ✓ scipy.fftconvolve is {(conv_rtf/base_rtf - 1)*100:.1f}% FASTER")
        else:
            print(f"  ✗ scipy.fftconvolve is {(1 - conv_rtf/base_rtf)*100:.1f}% SLOWER")
    
    print("\n" + "=" * 70)
    
    return all_results


if __name__ == '__main__':
    main()
