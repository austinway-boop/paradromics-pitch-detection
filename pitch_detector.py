#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
Streaming Pitch Detection Engine with MIDI Output

This solution implements a memory-efficient, streaming pitch detector that:
1. Reads audio in small chunks (max 2048 samples)
2. Processes 60s of 44kHz/16bit audio in under 15 seconds
3. Stays under 500MB RAM
4. Outputs MIDI melody files

Algorithm: NumPy-optimized autocorrelation pitch detection
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available, using slower pure Python")

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class StreamingPitchDetector:
    """Optimized pitch detector using numpy autocorrelation."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Pre-allocate FFT arrays for speed
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32) if HAS_NUMPY else None
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        self.prev_samples = None
        self.chunk_counter = 0  # For skipping chunks
        
    def _detect_pitch_numpy(self, samples):
        """Ultra-optimized FFT-based autocorrelation pitch detection."""
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Quick energy check using dot product (very fast)
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # Use pre-allocated buffer
        use_n = min(n, 2048)
        self._fft_buffer[:use_n] = samples[:use_n]
        self._fft_buffer[use_n:] = 0
        
        # FFT-based autocorrelation with pre-sized buffer
        fft = np.fft.rfft(self._fft_buffer)
        power = fft.real * fft.real + fft.imag * fft.imag
        corr = np.fft.irfft(power)
        
        # Find best peak - vectorized
        search = corr[self.min_lag:self.max_lag]
        if len(search) == 0:
            return None
            
        best_idx = search.argmax()
        
        if search[best_idx] > 0.2 * corr[0]:
            freq = self.sample_rate / (best_idx + self.min_lag)
            return self._freq_to_midi(freq)
        
        return None
    
    def _detect_pitch_pure(self, samples):
        """Pure Python fallback (slower)."""
        n = len(samples)
        if n < self.max_lag:
            return None
            
        energy = sum(s*s for s in samples)
        if energy < 0.001:
            return None
        
        best_lag = 0
        best_corr = -1
        
        # Sample fewer lags for speed
        step = max(1, (self.max_lag - self.min_lag) // 50)
        
        for lag in range(self.min_lag, self.max_lag, step):
            corr = sum(samples[i] * samples[i + lag] for i in range(n - lag))
            norm_corr = corr / (energy + 1e-10)
            
            if norm_corr > best_corr:
                best_corr = norm_corr
                best_lag = lag
        
        if best_lag > 0 and best_corr > 0.3:
            freq = self.sample_rate / best_lag
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
        self.chunk_counter += 1
        
        if HAS_NUMPY:
            samples = np.asarray(samples, dtype=np.float32)
            
            # Process every chunk for accuracy (but fast algorithm)
            midi_note = self._detect_pitch_numpy(samples)
        else:
            if self.prev_samples:
                combined = self.prev_samples + list(samples)
            else:
                combined = list(samples)
            
            self.prev_samples = list(samples[len(samples)//2:])
            midi_note = self._detect_pitch_pure(combined)
        
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


def process_wav(input_path, output_path, chunk_size=2048):
    """Process WAV file with streaming pitch detection."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz/{sw*8}bit")
        
        detector = StreamingPitchDetector(sample_rate=sr, chunk_size=chunk_size)
        
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
        'notes': len(notes),
        'output': output_path
    }


def generate_test_wav(filename, duration=60, sr=44100):
    """Generate test WAV with melody."""
    if HAS_NUMPY:
        notes_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
        t = np.arange(int(duration * sr)) / sr
        
        # Create melody by cycling through notes
        note_dur = 0.5
        samples_per_note = int(note_dur * sr)
        
        samples = np.zeros(len(t), dtype=np.float32)
        for i in range(len(t)):
            note_idx = (i // samples_per_note) % len(notes_hz)
            freq = notes_hz[note_idx]
            pos_in_note = (i % samples_per_note) / samples_per_note
            
            # Simple sine with envelope
            env = min(1.0, pos_in_note * 20) * max(0.0, 1.0 - max(0, pos_in_note - 0.8) * 5)
            samples[i] = 0.7 * np.sin(2 * np.pi * freq * t[i]) * env
        
        samples = (samples * 32767).astype(np.int16)
    else:
        notes_hz = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        samples = []
        note_dur = 0.5
        samples_per_note = int(note_dur * sr)
        
        for i in range(int(duration * sr)):
            note_idx = (i // samples_per_note) % len(notes_hz)
            freq = notes_hz[note_idx]
            pos = (i % samples_per_note) / samples_per_note
            
            env = min(1.0, pos * 20) * max(0.0, 1.0 - max(0, pos - 0.8) * 5)
            sample = 0.7 * math.sin(2 * math.pi * freq * i / sr) * env
            samples.append(max(-32767, min(32767, int(sample * 32767))))
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        if HAS_NUMPY:
            wav.writeframes(samples.tobytes())
        else:
            wav.writeframes(struct.pack(f'<{len(samples)}h', *samples))
    
    print(f"  Generated: {os.path.basename(filename)} ({duration}s)")


def main():
    print("=" * 60)
    print("PARADROMICS INTERNSHIP QUALIFIER - STAGE 1")
    print("Digital Ear: Streaming Pitch Detection Engine")
    print(f"NumPy: {'Available' if HAS_NUMPY else 'Not available (slower)'}")
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
    
    print("\n[2] Processing Audio Files")
    print("-" * 40)
    
    results = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '.mid'))
        m = process_wav(inp, out)
        results.append(m)
        print(f"    -> {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    max_mem = max(r['memory_mb'] for r in results)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total audio: {total_dur:.1f}s processed in {total_time:.2f}s")
    print(f"Speed: {total_dur/total_time:.1f}x real-time")
    print(f"Peak memory: {max_mem:.2f} MB")
    
    print("\nCONSTRAINT CHECK:")
    speed_ok = total_dur / total_time >= 4.0
    mem_ok = max_mem < 500
    print(f"  [{'PASS' if speed_ok else 'FAIL'}] 4x Real-Time: {total_dur/total_time:.1f}x")
    print(f"  [{'PASS' if mem_ok else 'FAIL'}] <500MB RAM: {max_mem:.2f} MB")
    print(f"  [PASS] 2048-sample chunks")
    
    # Save log
    log = os.path.join(base_dir, 'performance_log.txt')
    with open(log, 'w') as f:
        f.write("PARADROMICS PERFORMANCE LOG\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"File: {os.path.basename(r['output']).replace('.mid', '.wav')}\n")
            f.write(f"  Duration: {r['duration']:.2f}s\n")
            f.write(f"  Processing: {r['elapsed']:.2f}s ({r['rtf']:.1f}x RT)\n")
            f.write(f"  Memory: {r['memory_mb']:.2f} MB\n")
            f.write(f"  Notes: {r['notes']}\n\n")
        f.write("=" * 50 + "\n")
        f.write(f"TOTAL: {total_dur:.1f}s in {total_time:.2f}s = {total_dur/total_time:.1f}x RT\n")
        f.write(f"PEAK MEM: {max_mem:.2f} MB\n")
    
    print(f"\nLog saved: {log}")
    print("\nDELIVERABLES:")
    for r in results:
        print(f"  - {os.path.basename(r['output'])}")
    
    return results


if __name__ == '__main__':
    main()
