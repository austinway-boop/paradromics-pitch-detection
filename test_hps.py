#!/usr/bin/env python3
"""Quick test for HPS pitch detection."""
import sys
import time
import tracemalloc
import wave
import struct
import numpy as np
import os

# Import from the HPS module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pitch_hps import HPSPitchDetectorOptimized, MIDIWriter, generate_test_wav

def test_hps():
    print("HPS Pitch Detection Test")
    print("=" * 50)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate a short test file
    test_file = os.path.join(base_dir, "test_hps_quick.wav")
    print("Generating 10s test audio...")
    generate_test_wav(test_file, duration=10, sr=44100)
    
    # Process it
    print("Processing with HPS...")
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(test_file, 'rb') as wav:
        sr = wav.getframerate()
        nf = wav.getnframes()
        sw = wav.getsampwidth()
        ch = wav.getnchannels()
        
        detector = HPSPitchDetectorOptimized(sample_rate=sr, chunk_size=2048)
        
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
    
    print(f"\nResults:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Processing: {elapsed:.3f}s")
    print(f"  Speed: {rtf:.1f}x real-time")
    print(f"  Memory: {peak / (1024*1024):.2f} MB")
    print(f"  Notes detected: {len(notes)}")
    
    # Write MIDI
    out_file = test_file.replace('.wav', '.mid')
    MIDIWriter().write(notes, out_file)
    print(f"  MIDI output: {os.path.basename(out_file)}")
    
    return rtf

if __name__ == '__main__':
    test_hps()
