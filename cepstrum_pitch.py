#!/usr/bin/env python3
"""
Cepstrum Pitch Detection - Optimized Implementation

Cepstrum = IFFT(log(|FFT(signal)|^2))

The cepstrum transforms periodic components in the frequency domain 
into peaks in the "quefrency" domain. For voiced speech/music, the 
fundamental frequency appears as a clear peak at quefrency = 1/f0.

Optimizations:
1. Real FFT (rfft) - half the computation
2. Pre-allocated buffers - avoid allocation overhead
3. Windowing - reduce spectral leakage
4. Vectorized peak finding
5. Log floor - avoid log(0)
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math
import numpy as np


class CepstrumPitchDetector:
    """Cepstrum-based pitch detector optimized for streaming."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Quefrency bounds (in samples) corresponding to frequency range
        # quefrency = sample_rate / frequency
        self.min_quefrency = max(2, int(sample_rate / max_freq))  # High freq -> low quefrency
        self.max_quefrency = min(chunk_size // 2, int(sample_rate / min_freq))  # Low freq -> high quefrency
        
        # Pre-compute Hann window (reduces spectral leakage)
        self.window = np.hanning(chunk_size).astype(np.float32)
        
        # Pre-allocate buffers
        self.fft_size = chunk_size
        self._windowed = np.zeros(chunk_size, dtype=np.float32)
        
        # Note tracking state
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch_cepstrum(self, samples):
        """
        Cepstrum pitch detection:
        1. Apply window
        2. FFT -> magnitude squared (power spectrum)
        3. Log of power spectrum
        4. IFFT -> cepstrum
        5. Find peak in quefrency range
        """
        n = len(samples)
        if n < self.min_quefrency * 2:
            return None
        
        # Quick energy check
        energy = np.dot(samples, samples)
        if energy < 1e-6:
            return None
        
        # Handle variable chunk sizes (especially at end of file)
        use_n = min(n, self.chunk_size)
        self._windowed[:use_n] = samples[:use_n] * self.window[:use_n]
        self._windowed[use_n:] = 0
        
        # Real FFT (faster than full FFT)
        spectrum = np.fft.rfft(self._windowed)
        
        # Power spectrum with floor to avoid log(0)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        np.maximum(power, 1e-10, out=power)
        
        # Log power spectrum (this is the "cepstrum" part)
        log_power = np.log(power)
        
        # Inverse FFT to get cepstrum (real part only)
        cepstrum = np.fft.irfft(log_power)
        
        # Search for peak in valid quefrency range
        search_region = cepstrum[self.min_quefrency:self.max_quefrency]
        if len(search_region) == 0:
            return None
        
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]
        
        # Threshold: peak must be significant relative to mean
        # Lower threshold for better detection
        mean_abs = np.mean(np.abs(search_region))
        if peak_value < mean_abs * 1.5:
            return None
        
        # Convert quefrency to frequency
        quefrency = peak_idx + self.min_quefrency
        if quefrency <= 0:
            return None
            
        frequency = self.sample_rate / quefrency
        
        return self._freq_to_midi(frequency)
    
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
        midi_note = self._detect_pitch_cepstrum(samples)
        
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


class CepstrumPitchDetectorV2:
    """
    Optimized Cepstrum V2 - With:
    1. Zero-padded FFT for better frequency resolution
    2. Pre-computed quefrency-to-midi lookup
    3. Parabolic interpolation for sub-sample peak finding
    4. Better adaptive thresholding
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Use zero-padding to 4096 for better frequency resolution
        self.fft_size = 4096
        
        # Quefrency bounds
        self.min_quefrency = max(2, int(sample_rate / max_freq))
        self.max_quefrency = min(self.fft_size // 2, int(sample_rate / min_freq))
        
        # Pre-compute window
        self.window = np.hanning(chunk_size).astype(np.float32)
        
        # Pre-allocate zero-padded buffer
        self._padded = np.zeros(self.fft_size, dtype=np.float32)
        
        # Pre-compute quefrency to MIDI lookup table
        self._quefrency_to_midi = np.zeros(self.max_quefrency + 1, dtype=np.int8)
        for q in range(1, self.max_quefrency + 1):
            freq = sample_rate / q
            if freq > 0:
                midi = 69 + 12 * math.log2(freq / 440.0)
                midi_rounded = int(round(midi))
                if 21 <= midi_rounded <= 108:
                    self._quefrency_to_midi[q] = midi_rounded
        
        # Note tracking state
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch_cepstrum(self, samples):
        """Optimized cepstrum with zero-padding and interpolation."""
        n = len(samples)
        if n < 64:
            return None
        
        # Quick energy check
        energy = np.dot(samples, samples)
        if energy < 1e-6:
            return None
        
        # Zero-padded windowed signal - handle variable sizes
        use_n = min(n, self.chunk_size)
        self._padded[:use_n] = samples[:use_n] * self.window[:use_n]
        self._padded[use_n:] = 0
        
        # Real FFT
        spectrum = np.fft.rfft(self._padded)
        
        # Power spectrum with floor
        power = spectrum.real ** 2 + spectrum.imag ** 2
        np.maximum(power, 1e-10, out=power)
        
        # Log power spectrum
        log_power = np.log(power)
        
        # Inverse FFT -> cepstrum
        cepstrum = np.fft.irfft(log_power, n=self.fft_size)
        
        # Search for peak in valid quefrency range
        search_start = self.min_quefrency
        search_end = min(self.max_quefrency, len(cepstrum) - 1)
        
        if search_end <= search_start:
            return None
            
        search_region = cepstrum[search_start:search_end]
        peak_local_idx = np.argmax(search_region)
        peak_value = search_region[peak_local_idx]
        
        # Better adaptive threshold - use mean + std
        mean_val = np.mean(search_region)
        std_val = np.std(search_region)
        if peak_value < mean_val + 1.5 * std_val:
            return None
        
        # Parabolic interpolation for sub-sample accuracy
        peak_idx = peak_local_idx + search_start
        if 1 <= peak_local_idx < len(search_region) - 1:
            alpha = search_region[peak_local_idx - 1]
            beta = peak_value
            gamma = search_region[peak_local_idx + 1]
            
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-10:
                delta = 0.5 * (alpha - gamma) / denom
                refined_quefrency = peak_idx + delta
            else:
                refined_quefrency = peak_idx
        else:
            refined_quefrency = peak_idx
        
        # Convert to frequency and MIDI
        if refined_quefrency <= 0:
            return None
            
        frequency = self.sample_rate / refined_quefrency
        
        # Use lookup table for integer quefrency
        int_quefrency = int(round(refined_quefrency))
        if 1 <= int_quefrency <= self.max_quefrency:
            return self._quefrency_to_midi[int_quefrency]
        
        return self._freq_to_midi(frequency)
    
    def _freq_to_midi(self, freq):
        """Convert frequency to MIDI note number (fallback)."""
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
        midi_note = self._detect_pitch_cepstrum(samples)
        
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


class CepstrumPitchDetectorV3:
    """
    Cepstrum V3 - Maximum speed optimizations:
    1. Use scipy.fft if available (faster than numpy.fft)
    2. Simplified threshold
    3. Skip low-energy chunks entirely
    4. Minimal allocations
    """
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Try scipy for faster FFT
        try:
            import scipy.fft
            self._rfft = scipy.fft.rfft
            self._irfft = scipy.fft.irfft
            self._fft_backend = "scipy"
        except ImportError:
            self._rfft = np.fft.rfft
            self._irfft = np.fft.irfft
            self._fft_backend = "numpy"
        
        # Quefrency bounds
        self.min_quefrency = max(2, int(sample_rate / max_freq))
        self.max_quefrency = min(chunk_size // 2, int(sample_rate / min_freq))
        
        # Pre-compute window
        self.window = np.hanning(chunk_size).astype(np.float32)
        
        # Pre-allocate buffers
        self._windowed = np.zeros(chunk_size, dtype=np.float32)
        
        # Pre-compute MIDI lookup
        self._q_to_midi = np.zeros(self.max_quefrency + 1, dtype=np.int8)
        for q in range(1, self.max_quefrency + 1):
            freq = sample_rate / q
            if freq > 0:
                midi = 69 + 12 * math.log2(freq / 440.0)
                midi_rounded = int(round(midi))
                if 21 <= midi_rounded <= 108:
                    self._q_to_midi[q] = midi_rounded
        
        # Note tracking
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch(self, samples):
        """Ultra-fast cepstrum pitch detection."""
        n = len(samples)
        if n < 64:
            return None
            
        # Energy check
        energy = np.dot(samples, samples)
        if energy < 1e-6:
            return None
        
        # Window - handle variable sizes
        use_n = min(n, self.chunk_size)
        self._windowed[:use_n] = samples[:use_n] * self.window[:use_n]
        self._windowed[use_n:] = 0
        
        # FFT -> log power -> IFFT
        spectrum = self._rfft(self._windowed[:self.chunk_size])
        power = spectrum.real ** 2 + spectrum.imag ** 2
        np.maximum(power, 1e-10, out=power)
        cepstrum = self._irfft(np.log(power))
        
        # Find peak
        region = cepstrum[self.min_quefrency:self.max_quefrency]
        if len(region) == 0:
            return None
            
        peak_idx = region.argmax()
        peak_value = region[peak_idx]
        
        # Use mean-based threshold
        if peak_value < np.mean(region) + np.std(region):
            return None
        
        # Convert to MIDI
        quefrency = peak_idx + self.min_quefrency
        return self._q_to_midi[quefrency] if quefrency <= self.max_quefrency else None
    
    def process_chunk(self, samples):
        """Process a chunk."""
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
        """Finalize notes."""
        if self.current_note is not None:
            note_dur = self.time_position - self.note_start_time
            if note_dur > 0.03:
                self.notes.append((self.current_note, self.note_start_time, note_dur))
        return self.notes


# Import original detector for comparison
from pitch_detector import StreamingPitchDetector as OriginalDetector, MIDIWriter


def benchmark_detectors(wav_path):
    """Benchmark all detector implementations."""
    
    # Load audio once
    with wave.open(wav_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        raw_data = wav.readframes(nf)
    
    # Decode audio
    ns = len(raw_data) // (sw * ch)
    samples_raw = struct.unpack(f'<{ns * ch}h', raw_data) if sw == 2 else [b-128 for b in raw_data]
    if ch == 2:
        samples_raw = [(samples_raw[i] + samples_raw[i+1]) / 2 for i in range(0, len(samples_raw), 2)]
    
    max_val = 2 ** (sw * 8 - 1)
    all_samples = [s / max_val for s in samples_raw]
    
    duration = len(all_samples) / sr
    chunk_size = 2048
    
    print(f"\nBenchmarking on: {os.path.basename(wav_path)} ({duration:.1f}s @ {sr}Hz)")
    print("-" * 60)
    
    detectors = [
        ("Original (FFT-autocorr)", OriginalDetector),
        ("Cepstrum V1 (Basic)", CepstrumPitchDetector),
        ("Cepstrum V2 (Interpolated)", CepstrumPitchDetectorV2),
        ("Cepstrum V3 (Maximum Speed)", CepstrumPitchDetectorV3),
    ]
    
    results = []
    
    for name, DetectorClass in detectors:
        tracemalloc.start()
        t0 = time.perf_counter()
        
        detector = DetectorClass(sample_rate=sr, chunk_size=chunk_size)
        
        # Process in chunks
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
            'notes_data': notes,
        })
        
        backend = ""
        if hasattr(detector, '_fft_backend'):
            backend = f" ({detector._fft_backend})"
        
        print(f"  {name}{backend}: {rtf:.1f}x RT | {elapsed:.3f}s | {len(notes)} notes | {peak_mem/1024/1024:.2f}MB")
    
    return results


def compare_accuracy(results, ground_truth_notes=None):
    """Compare note detection accuracy between methods."""
    print("\nAccuracy Comparison:")
    print("-" * 60)
    
    # Use original detector as baseline
    baseline = results[0]['notes_data']
    
    for r in results[1:]:
        detected = r['notes_data']
        
        # Simple overlap-based comparison
        matched = 0
        for bn, bstart, bdur in baseline:
            for dn, dstart, ddur in detected:
                # Same note within 100ms timing tolerance
                if bn == dn and abs(bstart - dstart) < 0.1:
                    matched += 1
                    break
        
        precision = matched / len(detected) if detected else 0
        recall = matched / len(baseline) if baseline else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {r['name']}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} ({len(detected)} notes)")


def main():
    print("=" * 60)
    print("CEPSTRUM PITCH DETECTION - OPTIMIZATION STUDY")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test on all available WAV files
    test_files = ['test_simple.wav', 'test_clean.wav', 'test_complex.wav']
    
    all_results = []
    
    for test_file in test_files:
        wav_path = os.path.join(base_dir, test_file)
        if os.path.exists(wav_path):
            results = benchmark_detectors(wav_path)
            all_results.append((test_file, results))
            compare_accuracy(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Average RTF across all tests
    method_rtfs = {}
    method_notes = {}
    for test_file, results in all_results:
        for r in results:
            if r['name'] not in method_rtfs:
                method_rtfs[r['name']] = []
                method_notes[r['name']] = []
            method_rtfs[r['name']].append(r['rtf'])
            method_notes[r['name']].append(r['notes'])
    
    print("\nAverage Real-Time Factor & Notes Detected:")
    for name in method_rtfs:
        avg_rtf = sum(method_rtfs[name]) / len(method_rtfs[name])
        total_notes = sum(method_notes[name])
        print(f"  {name}: {avg_rtf:.1f}x RT | {total_notes} total notes")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Find fastest cepstrum method
    cepstrum_methods = {k: v for k, v in method_rtfs.items() if 'Cepstrum' in k}
    if cepstrum_methods:
        best = max(cepstrum_methods.items(), key=lambda x: sum(x[1])/len(x[1]))
        orig_avg = sum(method_rtfs['Original (FFT-autocorr)']) / len(method_rtfs['Original (FFT-autocorr)'])
        best_avg = sum(best[1]) / len(best[1])
        
        orig_notes = sum(method_notes['Original (FFT-autocorr)'])
        best_notes = sum(method_notes[best[0]])
        
        if best_avg > orig_avg:
            speedup = (best_avg - orig_avg) / orig_avg * 100
            print(f"\n[SPEED] {best[0]} is {speedup:.1f}% FASTER than original")
            print(f"  Original: {orig_avg:.1f}x RT -> Cepstrum: {best_avg:.1f}x RT")
        else:
            print(f"\n[SPEED] Cepstrum is SLOWER than FFT-autocorrelation")
            print(f"  Original: {orig_avg:.1f}x RT, Best Cepstrum: {best_avg:.1f}x RT")
        
        print(f"\n[ACCURACY] Note detection comparison:")
        print(f"  Original detected: {orig_notes} notes")
        print(f"  Best Cepstrum detected: {best_notes} notes")
        
        if best_notes < orig_notes * 0.5:
            print("\n  WARNING: Cepstrum detects significantly fewer notes.")
            print("  This indicates threshold tuning is needed for your audio type.")
            print("  Cepstrum works best on monophonic audio with clear harmonics.")


if __name__ == '__main__':
    main()
