#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
MULTIPROCESSING Pitch Detection Engine

Method 10: Use Python multiprocessing Pool to process chunks in parallel.
Baseline: 4.7x RT single-threaded
Goal: Leverage multiple CPU cores for faster processing
"""

import wave
import struct
import time
import tracemalloc
import os
import math
import numpy as np
from numba import jit
from multiprocessing import Pool, cpu_count, shared_memory
import warnings
warnings.filterwarnings('ignore')

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


@jit(nopython=True, cache=True, fastmath=True)
def yin_detect_pitch(samples, sr=44100, fmin=50.0, fmax=2000.0, threshold=0.1):
    """Ultra-optimized YIN pitch detection with Numba JIT."""
    N = len(samples)
    W = N // 2
    
    tau_min = max(1, int(sr / fmax))
    tau_max = min(W, int(sr / fmin))
    
    if tau_max <= tau_min:
        return 0.0
    
    # Difference function
    df = np.zeros(tau_max, dtype=np.float64)
    for tau in range(tau_max):
        acc = 0.0
        for j in range(W):
            diff = samples[j] - samples[j + tau]
            acc += diff * diff
        df[tau] = acc
    
    # CMNDF
    cmndf = np.zeros(tau_max, dtype=np.float64)
    cmndf[0] = 1.0
    running_sum = 0.0
    
    for tau in range(1, tau_max):
        running_sum += df[tau]
        if running_sum > 1e-10:
            cmndf[tau] = df[tau] * tau / running_sum
        else:
            cmndf[tau] = 1.0
    
    # Find pitch
    best_tau = tau_max - 1
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            best_tau = tau
            break
    
    # Parabolic interpolation
    if best_tau > 0 and best_tau < tau_max - 1:
        s0, s1, s2 = cmndf[best_tau - 1], cmndf[best_tau], cmndf[best_tau + 1]
        denom = 2.0 * (s0 - 2.0 * s1 + s2)
        if abs(denom) > 1e-10:
            adj = max(-1.0, min(1.0, (s0 - s2) / denom))
            tau_refined = float(best_tau) + adj
        else:
            tau_refined = float(best_tau)
    else:
        tau_refined = float(best_tau)
    
    return sr / tau_refined if tau_refined >= 1.0 else 0.0


def freq_to_midi(freq):
    """Convert frequency to MIDI note."""
    if freq <= 0:
        return None
    midi = 69 + 12 * math.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    return midi_rounded if 21 <= midi_rounded <= 108 else None


def process_single_chunk(args):
    """Process a single chunk - worker function for multiprocessing."""
    chunk_idx, samples, sr = args
    samples = np.asarray(samples, dtype=np.float64)
    freq = yin_detect_pitch(samples, sr)
    midi = freq_to_midi(freq) if freq > 0 else None
    return (chunk_idx, midi)


def process_chunk_batch(args):
    """Process a batch of chunks - reduces IPC overhead."""
    batch_data, sr = args
    results = []
    for chunk_idx, samples in batch_data:
        samples = np.asarray(samples, dtype=np.float64)
        freq = yin_detect_pitch(samples, sr)
        midi = freq_to_midi(freq) if freq > 0 else None
        results.append((chunk_idx, midi))
    return results


class MultiprocessingPitchDetector:
    """Pitch detector using multiprocessing Pool."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, n_workers=None):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
    def process_audio(self, all_samples):
        """Process all audio samples using multiprocessing."""
        # Split into chunks
        chunks = []
        chunk_dur = self.chunk_size / self.sr
        
        for i in range(0, len(all_samples), self.chunk_size):
            chunk = all_samples[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                chunks.append((len(chunks), chunk, self.sr))
        
        if not chunks:
            return []
        
        # Process chunks in parallel using Pool
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(process_single_chunk, chunks)
        
        # Sort by chunk index (should already be sorted but ensure order)
        results.sort(key=lambda x: x[0])
        midi_sequence = [r[1] for r in results]
        
        # Convert to notes with timing
        notes = []
        current_note = None
        note_start = 0.0
        
        for i, midi in enumerate(midi_sequence):
            time_pos = i * chunk_dur
            
            if midi != current_note:
                if current_note is not None:
                    dur = time_pos - note_start
                    if dur > 0.03:
                        notes.append((current_note, note_start, dur))
                current_note = midi
                note_start = time_pos
        
        # Finalize last note
        if current_note is not None:
            final_time = len(midi_sequence) * chunk_dur
            dur = final_time - note_start
            if dur > 0.03:
                notes.append((current_note, note_start, dur))
        
        return notes


class BatchedMultiprocessingPitchDetector:
    """Pitch detector using batched multiprocessing (less IPC overhead)."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, n_workers=None, batch_size=50):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.batch_size = batch_size
        
    def process_audio(self, all_samples):
        """Process all audio samples using batched multiprocessing."""
        chunk_dur = self.chunk_size / self.sr
        
        # Create all chunks with indices
        indexed_chunks = []
        for i in range(0, len(all_samples), self.chunk_size):
            chunk = all_samples[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                indexed_chunks.append((len(indexed_chunks), list(chunk)))
        
        if not indexed_chunks:
            return []
        
        # Group into batches
        batches = []
        for i in range(0, len(indexed_chunks), self.batch_size):
            batch = indexed_chunks[i:i + self.batch_size]
            batches.append((batch, self.sr))
        
        # Process batches in parallel
        with Pool(processes=self.n_workers) as pool:
            batch_results = pool.map(process_chunk_batch, batches)
        
        # Flatten results
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)
        
        # Sort by chunk index
        all_results.sort(key=lambda x: x[0])
        midi_sequence = [r[1] for r in all_results]
        
        # Convert to notes
        notes = []
        current_note = None
        note_start = 0.0
        
        for i, midi in enumerate(midi_sequence):
            time_pos = i * chunk_dur
            
            if midi != current_note:
                if current_note is not None:
                    dur = time_pos - note_start
                    if dur > 0.03:
                        notes.append((current_note, note_start, dur))
                current_note = midi
                note_start = time_pos
        
        if current_note is not None:
            final_time = len(midi_sequence) * chunk_dur
            dur = final_time - note_start
            if dur > 0.03:
                notes.append((current_note, note_start, dur))
        
        return notes


class MIDIWriter:
    def __init__(self, tempo=120):
        self.tempo = tempo
        self.tpb = 480
        
    def write(self, notes, filename):
        track = bytearray(b'\x00\xFF\x51\x03')
        track.extend(int(60_000_000 / self.tempo).to_bytes(3, 'big'))
        
        events = []
        for note, start, dur in sorted(notes, key=lambda x: x[1]):
            st = int(start * (self.tempo / 60.0) * self.tpb)
            et = int((start + dur) * (self.tempo / 60.0) * self.tpb)
            events.append((st, True, note, 100))
            events.append((et, False, note, 0))
        
        events.sort(key=lambda x: (x[0], not x[1]))
        
        tick = 0
        for t, on, n, v in events:
            delta = t - tick
            tick = t
            vlq = []
            vlq.append(delta & 0x7F)
            delta >>= 7
            while delta:
                vlq.append((delta & 0x7F) | 0x80)
                delta >>= 7
            track.extend(bytes(reversed(vlq)))
            track.extend(bytes([0x90 if on else 0x80, n, v]))
        
        track.extend(b'\x00\xFF\x2F\x00')
        
        with open(filename, 'wb') as f:
            f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, self.tpb))
            f.write(b'MTrk' + struct.pack('>I', len(track)) + track)


def load_wav(input_path):
    """Load entire WAV file into memory."""
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        raw = wav.readframes(nf)
        ns = len(raw) // (sw * ch)
        samples = struct.unpack(f'<{ns * ch}h', raw)
        
        if ch == 2:
            samples = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
        
        samples = [s / 32768.0 for s in samples]
        
    return samples, sr, nf


def process_wav_multiprocessing(input_path, output_path, chunk_size=2048, n_workers=None, batched=False, batch_size=50):
    """Process WAV file using multiprocessing."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    # Load audio
    samples, sr, nf = load_wav(input_path)
    
    # Process with multiprocessing
    if batched:
        detector = BatchedMultiprocessingPitchDetector(sr, chunk_size, n_workers, batch_size)
    else:
        detector = MultiprocessingPitchDetector(sr, chunk_size, n_workers)
    
    notes = detector.process_audio(samples)
    
    # Write MIDI
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'elapsed': t1 - t0,
        'duration': nf / sr,
        'rtf': (nf / sr) / (t1 - t0),
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes),
        'workers': detector.n_workers
    }


def process_wav_single(input_path, output_path, chunk_size=2048):
    """Process WAV file single-threaded (baseline)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    samples, sr, nf = load_wav(input_path)
    
    # Process sequentially
    chunk_dur = chunk_size / sr
    midi_sequence = []
    
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunk_np = np.asarray(chunk, dtype=np.float64)
            freq = yin_detect_pitch(chunk_np, sr)
            midi = freq_to_midi(freq) if freq > 0 else None
            midi_sequence.append(midi)
    
    # Convert to notes
    notes = []
    current_note = None
    note_start = 0.0
    
    for i, midi in enumerate(midi_sequence):
        time_pos = i * chunk_dur
        if midi != current_note:
            if current_note is not None:
                dur = time_pos - note_start
                if dur > 0.03:
                    notes.append((current_note, note_start, dur))
            current_note = midi
            note_start = time_pos
    
    if current_note is not None:
        final_time = len(midi_sequence) * chunk_dur
        dur = final_time - note_start
        if dur > 0.03:
            notes.append((current_note, note_start, dur))
    
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'elapsed': t1 - t0,
        'duration': nf / sr,
        'rtf': (nf / sr) / (t1 - t0),
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes)
    }


def generate_test_wav(filename, duration=60, sr=44100):
    """Generate test WAV file."""
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
    print("PARADROMICS - MULTIPROCESSING PITCH DETECTION")
    print("Method 10: Python multiprocessing Pool for parallel chunk processing")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    n_cores = cpu_count()
    
    print(f"\nSystem: {n_cores} CPU cores detected")
    
    # Warm up JIT
    print("\nWarming up JIT compiler...")
    test = np.random.randn(2048).astype(np.float64)
    for _ in range(5):
        yin_detect_pitch(test)
    print("JIT ready!")
    
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
        ('test_complex.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 50)
    for name, dur in tests:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            generate_test_wav(path, dur)
            print(f"  Generated: {name}")
        else:
            print(f"  Using existing: {name}")
    
    # Test configurations
    configs = [
        ("Single-threaded (baseline)", lambda inp, out: process_wav_single(inp, out)),
        ("Multiprocessing (per-chunk)", lambda inp, out: process_wav_multiprocessing(inp, out, batched=False)),
        ("Multiprocessing (batched)", lambda inp, out: process_wav_multiprocessing(inp, out, batched=True, batch_size=100)),
    ]
    
    for config_name, process_func in configs:
        print(f"\n[*] {config_name}")
        print("-" * 50)
        
        results = []
        for name, _ in tests:
            inp = os.path.join(base_dir, name)
            out = os.path.join(base_dir, name.replace('.wav', '_mp.mid'))
            
            try:
                m = process_func(inp, out)
                results.append(m)
                workers = m.get('workers', 1)
                print(f"  {name}: {m['rtf']:.1f}x RT | {m['memory_mb']:.2f}MB | {m['notes']} notes | workers={workers}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
        
        if results:
            total_dur = sum(r['duration'] for r in results)
            total_time = sum(r['elapsed'] for r in results)
            max_mem = max(r['memory_mb'] for r in results)
            
            print(f"  TOTAL: {total_dur:.1f}s in {total_time:.2f}s = {total_dur/total_time:.1f}x RT | Peak: {max_mem:.2f}MB")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Multiprocessing Observations:
1. Pool overhead: Process creation, IPC, and data serialization add latency
2. Per-chunk parallelism: Each chunk is small (2048 samples), so IPC dominates
3. Batched approach: Reduces IPC by grouping chunks, better for small workloads
4. Memory: Multiple processes = higher memory usage (each has own copy)

For this workload:
- Single-threaded is already 4.7x RT (fast enough)
- Multiprocessing overhead may exceed the benefit for small chunks
- Better suited for: larger chunk sizes, longer files, or CPU-bound tasks

When multiprocessing DOES help:
- Very long audio files (hours+)
- Larger chunk sizes (8192+ samples)
- Independent file processing (process multiple files in parallel)
    """)


if __name__ == '__main__':
    # Required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
