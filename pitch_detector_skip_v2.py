#!/usr/bin/env python3
"""
PARADROMICS OPTIMIZATION - Method 7 v2: Skip Chunk Analysis

Key insight from v1: Skipping chunks at factor=2 gave 4.0x RT (WORSE than 4.7x baseline!)
This means the bottleneck is NOT in FFT processing - it's in:
1. File I/O (wave.readframes)
2. Sample unpacking (struct.unpack)
3. Channel mixing / normalization
4. Python list operations

SOLUTION: Read larger chunks but only process portions
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

MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class UltraFastPitchDetector:
    """Pitch detector optimized for maximum throughput."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        
        # Pre-allocate everything
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
    def _detect_pitch(self, samples):
        """Minimal FFT pitch detection."""
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
            if freq <= 0:
                return None
            midi = 69 + 12 * math.log2(freq / 440.0)
            midi_rounded = int(round(midi))
            if 21 <= midi_rounded <= 108:
                return midi_rounded
        
        return None
    
    def process_chunk(self, samples, duration):
        """Process chunk."""
        midi_note = self._detect_pitch(samples)
        
        # Note state update
        if midi_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:
                    self.notes.append((
                        self.current_note,
                        self.note_start_time,
                        note_duration
                    ))
            
            self.current_note = midi_note
            self.note_start_time = self.time_position
        
        self.time_position += duration
        return midi_note
    
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


def process_wav_large_chunks(input_path, output_path, read_chunk_size=8192, process_every_n=4):
    """
    OPTIMIZED: Read in large chunks but process sub-chunks
    
    Strategy:
    - Read 8192 samples at once (reduces I/O overhead)
    - Only process first 2048 samples of each large chunk
    - Apply detected pitch to entire chunk duration
    - Effectively: 4x speedup from reduced I/O + 4x from skipping = ~16x potential
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        detector = UltraFastPitchDetector(sample_rate=sr, chunk_size=2048)
        
        max_val = 2 ** (sw * 8 - 1)
        
        while True:
            raw = wav.readframes(read_chunk_size)
            if not raw:
                break
            
            ns = len(raw) // (sw * ch)
            chunk_duration = ns / sr
            
            # Numpy-optimized unpacking and mixing
            if sw == 2:
                samples = np.frombuffer(raw, dtype=np.int16)
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
            
            if ch == 2:
                samples = (samples[0::2] + samples[1::2]) / 2
            
            samples = samples.astype(np.float32) / max_val
            
            # Only process first 2048 samples but apply to whole chunk
            process_samples = samples[:2048] if len(samples) >= 2048 else samples
            detector.process_chunk(process_samples, chunk_duration)
        
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
        'read_chunk_size': read_chunk_size
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
    print("METHOD 7 v2: LARGE CHUNK READ + SKIP PROCESSING")
    print("Read 8192+ samples, only process 2048 for pitch")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate test files
    tests = [
        ('test_v2_30s.wav', 30),
        ('test_v2_60s.wav', 60),
        ('test_v2_45s.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 40)
    for name, dur in tests:
        generate_test_wav(os.path.join(base_dir, name), dur)
    
    # Test different read chunk sizes
    chunk_sizes = [2048, 4096, 8192, 16384, 32768]
    
    for chunk_size in chunk_sizes:
        print(f"\n[2] Processing with read_chunk_size={chunk_size}")
        print("-" * 40)
        
        results = []
        for name, _ in tests:
            inp = os.path.join(base_dir, name)
            out = os.path.join(base_dir, name.replace('.wav', f'_cs{chunk_size}.mid'))
            m = process_wav_large_chunks(inp, out, read_chunk_size=chunk_size)
            results.append(m)
            print(f"    {os.path.basename(inp)}: {m['rtf']:.1f}x RT | {m['memory_mb']:.2f}MB | {m['notes']} notes")
        
        total_dur = sum(r['duration'] for r in results)
        total_time = sum(r['elapsed'] for r in results)
        
        print(f"\n  CHUNK SIZE {chunk_size} RESULTS:")
        print(f"  Speed: {total_dur/total_time:.1f}x real-time")
        
        if total_dur/total_time >= 9.0:
            print(f"  [PASS] TARGET HIT: {total_dur/total_time:.1f}x >= 9x RT")
        else:
            print(f"  [MISS] Below target: {total_dur/total_time:.1f}x < 9x RT")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("If larger chunks don't help, bottleneck is FFT itself.")
    print("Solution: Use cheaper pitch detection (YIN, peak detection)")
    print("=" * 60)


if __name__ == '__main__':
    main()
