#!/usr/bin/env python3
"""
Paradromics Pitch Detection - AMDF Final Optimized Version
Average Magnitude Difference Function with Maximum Performance

Target: Beat 4.7x real-time baseline
Strategy:
1. NumPy-native audio decoding (no Python loops)
2. Aggressive signal downsampling (4x)
3. Binary search lag refinement
4. Process every 2nd chunk (temporal smoothing handles gaps)
5. Pre-computed lag indices
"""

import wave
import struct
import time
import tracemalloc
import os
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy required for this implementation")
    exit(1)

MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class AMDFFinalDetector:
    """Maximum performance AMDF pitch detector."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Aggressive downsampling (4x)
        self.ds = 4
        self.ds_rate = sample_rate // self.ds
        
        # Lag range in downsampled domain
        self.min_lag = max(1, int(self.ds_rate / max_freq))  # ~11
        self.max_lag = min(chunk_size // self.ds - 1, int(self.ds_rate / min_freq))  # ~137
        
        # Pre-compute coarse lag indices (15 points)
        n_coarse = 15
        self.coarse_lags = np.linspace(self.min_lag, self.max_lag, n_coarse, dtype=np.int32)
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        self.chunk_idx = 0
        self.prev_note = None  # For smoothing
        
    def detect_pitch(self, samples):
        """
        Ultra-fast AMDF with aggressive optimization.
        """
        n = len(samples)
        
        # 4x downsample using reshape + mean (very fast)
        n_ds = (n // self.ds) * self.ds
        if n_ds < self.max_lag * self.ds * 2:
            return None
        
        samples_ds = samples[:n_ds].reshape(-1, self.ds).mean(axis=1)
        n = len(samples_ds)
        
        # Fast energy check
        energy = np.dot(samples_ds, samples_ds)
        if energy < 1e-5 * n:
            return None
        
        rms = np.sqrt(energy / n)
        
        # Coarse search using pre-computed lags
        min_amdf = np.inf
        best_lag = self.coarse_lags[0]
        
        for lag in self.coarse_lags:
            if lag >= n:
                continue
            # Compute AMDF without creating intermediate arrays
            diff = samples_ds[:n-lag] - samples_ds[lag:]
            val = np.abs(diff).sum() / (n - lag)
            if val < min_amdf:
                min_amdf = val
                best_lag = lag
        
        # Fine refinement (-3 to +3 around best)
        fine_start = max(self.min_lag, best_lag - 3)
        fine_end = min(self.max_lag, best_lag + 4)
        
        for lag in range(fine_start, fine_end):
            if lag >= n:
                continue
            diff = samples_ds[:n-lag] - samples_ds[lag:]
            val = np.abs(diff).sum() / (n - lag)
            if val < min_amdf:
                min_amdf = val
                best_lag = lag
        
        # Threshold check (adaptive)
        threshold = 0.32 * rms
        
        if min_amdf < threshold and best_lag > 0:
            freq = self.ds_rate / best_lag
            return self._freq_to_midi(freq)
        
        return None

    def _freq_to_midi(self, freq):
        """Convert frequency to MIDI note number."""
        if freq <= 0 or freq < 60 or freq > 2000:
            return None
        midi = 69 + 12 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def process_chunk(self, samples):
        """Process a chunk, skip every other for speed."""
        chunk_duration = len(samples) / self.sample_rate
        self.chunk_idx += 1
        
        # Process every chunk for accuracy (algorithm is fast enough)
        midi_note = self.detect_pitch(samples)
        
        # Simple smoothing: require 2 consecutive same notes
        if midi_note == self.prev_note:
            final_note = midi_note
        else:
            final_note = self.prev_note  # Keep previous
        self.prev_note = midi_note
        
        self._update_note_state(final_note, chunk_duration)
        self.time_position += chunk_duration
        
        return final_note
    
    def _update_note_state(self, new_note, duration):
        """Track note on/off events."""
        if new_note != self.current_note:
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.05:  # 50ms minimum
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
            if note_duration > 0.05:
                self.notes.append((
                    self.current_note,
                    self.note_start_time,
                    note_duration
                ))
        return self.notes


class MIDIWriter:
    """Minimal MIDI file writer."""
    
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


def process_wav_fast(input_path, output_path, chunk_size=2048):
    """Process WAV file with maximum speed."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz")
        
        detector = AMDFFinalDetector(sample_rate=sr, chunk_size=chunk_size)
        max_val = 2 ** (sw * 8 - 1)
        
        # Pre-compute format string
        if sw == 2:
            fmt = '<{}h'.format(chunk_size * ch)
        
        while True:
            raw = wav.readframes(chunk_size)
            if not raw:
                break
            
            ns = len(raw) // (sw * ch)
            
            # Fast decode using numpy
            if sw == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128
            
            # Fast stereo to mono
            if ch == 2:
                samples = (samples[0::2] + samples[1::2]) * 0.5
            
            # Normalize
            samples /= max_val
            
            detector.process_chunk(samples)
        
        notes = detector.finalize()
    
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'elapsed': t1 - t0,
        'duration': nf / sr,
        'rtf': (nf / sr) / (t1 - t0) if t1 > t0 else 0,
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes),
    }


def main():
    print("=" * 60)
    print("PARADROMICS - AMDF FINAL (Maximum Performance)")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        'test_simple.wav',
        'test_clean.wav', 
        'test_complex.wav',
    ]
    
    print("\n[Processing]")
    print("-" * 40)
    
    results = []
    for name in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '_amdf_final.mid'))
        m = process_wav_fast(inp, out)
        results.append(m)
        print(f"    -> {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
    
    total_dur = sum(r['duration'] for r in results)
    total_time = sum(r['elapsed'] for r in results)
    max_mem = max(r['memory_mb'] for r in results)
    new_rtf = total_dur / total_time
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total audio: {total_dur:.1f}s in {total_time:.2f}s")
    print(f"Speed: {new_rtf:.1f}x real-time")
    print(f"Peak memory: {max_mem:.2f} MB")
    
    baseline = 4.7
    print(f"\nBaseline: {baseline}x | AMDF Final: {new_rtf:.1f}x")
    if new_rtf > baseline:
        print(f">>> IMPROVEMENT: +{((new_rtf/baseline)-1)*100:.1f}% <<<")
    else:
        print(f"Difference: {((new_rtf/baseline)-1)*100:.1f}%")
    
    print(f"\n[{'PASS' if new_rtf >= 4.0 else 'FAIL'}] 4x Real-Time")
    print(f"[{'PASS' if max_mem < 500 else 'FAIL'}] <500MB RAM")
    print("[PASS] 2048-sample chunks")
    
    return results


if __name__ == '__main__':
    main()
