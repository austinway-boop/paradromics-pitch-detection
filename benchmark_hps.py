#!/usr/bin/env python3
"""Benchmark HPS pitch detection."""
import sys
import os
import time
import tracemalloc
import wave
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pitch_hps import HPSPitchDetector, HPSPitchDetectorFast, MIDIWriter, generate_test_wav

def benchmark(duration_sec=60):
    """Benchmark HPS on a 60s file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(base_dir, f"bench_{duration_sec}s.wav")
    
    # Generate test audio
    print(f"Generating {duration_sec}s test audio...", flush=True)
    generate_test_wav(test_file, duration=duration_sec, sr=44100)
    
    # Test both versions
    for name, DetectorClass in [("HPS (skip=2)", HPSPitchDetector), ("HPS Fast (skip=3)", HPSPitchDetectorFast)]:
        tracemalloc.start()
        t0 = time.perf_counter()
        
        with wave.open(test_file, 'rb') as wav:
            sr = wav.getframerate()
            nf = wav.getnframes()
            sw = wav.getsampwidth()
            ch = wav.getnchannels()
            
            detector = DetectorClass(sample_rate=sr, chunk_size=2048)
            
            chunk_size = 2048
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
        
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = t1 - t0
        duration = nf / sr
        rtf = duration / elapsed
        
        print(f"{name}: {rtf:.1f}x RT | {peak/(1024*1024):.2f}MB | {len(notes)} notes", flush=True)
    
    # Clean up
    os.remove(test_file)

if __name__ == '__main__':
    print("=" * 50, flush=True)
    print("HPS Pitch Detection Benchmark", flush=True)
    print("=" * 50, flush=True)
    
    for dur in [30, 60, 45]:
        print(f"\n{dur}s audio:", flush=True)
        benchmark(dur)
    
    print("\nDone!", flush=True)
