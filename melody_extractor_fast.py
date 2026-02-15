#!/usr/bin/env python3
"""
Real Music Melody Extractor - Fast Version
Pure numpy, no scipy. Optimized for speed.
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math
import numpy as np

CHUNK_SIZE = 2048
HOP_SIZE = 1024  # Larger hop for speed


class FastMelodyExtractor:
    """Fast melody extractor using pure numpy."""
    
    def __init__(self, min_freq=80, max_freq=1000):
        self.min_freq = min_freq
        self.max_freq = max_freq
        
    def _freq_to_midi(self, freq):
        if freq is None or freq <= 0:
            return None
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 36 <= midi_rounded <= 96:
            return midi_rounded
        return None
    
    def _detect_pitch(self, chunk, sr):
        """Fast pitch detection using autocorrelation."""
        n = len(chunk)
        if n < 512:
            return None
        
        # Energy check
        energy = np.dot(chunk, chunk) / n
        if energy < 1e-6:
            return None
        
        # Normalize
        chunk = chunk - np.mean(chunk)
        
        # FFT-based autocorrelation
        fft = np.fft.rfft(chunk, n=2*n)
        power = fft.real**2 + fft.imag**2
        corr = np.fft.irfft(power)[:n]
        
        if corr[0] <= 0:
            return None
        corr = corr / corr[0]
        
        # Search range
        min_lag = max(2, int(sr / self.max_freq))
        max_lag = min(n // 2, int(sr / self.min_freq))
        
        if max_lag <= min_lag + 5:
            return None
        
        search = corr[min_lag:max_lag]
        
        # Find peak
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]
        
        if peak_val > 0.3:
            lag = peak_idx + min_lag
            return sr / lag
        return None
    
    def extract_melody(self, wav_path, mid_path):
        """Extract melody from audio file."""
        start_time = time.perf_counter()
        tracemalloc.start()
        
        # Load audio
        with wave.open(wav_path, 'rb') as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            raw = wav.readframes(n_frames)
        
        # Convert to float32
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Mono
        if n_channels == 2:
            samples = (samples[::2] + samples[1::2]) / 2
        
        audio_duration = len(samples) / sr
        
        # Detect pitches
        pitches = []
        hop_duration = HOP_SIZE / sr
        
        for i in range(0, len(samples) - CHUNK_SIZE, HOP_SIZE):
            chunk = samples[i:i + CHUNK_SIZE]
            freq = self._detect_pitch(chunk, sr)
            pitches.append(freq if freq else 0)
        
        # Convert to notes (simple approach - no smoothing for speed)
        notes = []
        current_midi = None
        note_start = 0
        min_duration = 0.08
        
        for i, freq in enumerate(pitches):
            time_pos = i * hop_duration
            midi = self._freq_to_midi(freq)
            
            if midi != current_midi:
                if current_midi is not None:
                    duration = time_pos - note_start
                    if duration >= min_duration:
                        notes.append((current_midi, note_start, duration))
                current_midi = midi
                note_start = time_pos
        
        if current_midi is not None:
            duration = len(pitches) * hop_duration - note_start
            if duration >= min_duration:
                notes.append((current_midi, note_start, duration))
        
        # Write MIDI
        self._write_midi(notes, mid_path)
        
        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'notes': len(notes),
            'duration': audio_duration,
            'processing_time': elapsed,
            'rtf': audio_duration / elapsed if elapsed > 0 else 0,
            'memory_mb': peak / 1024 / 1024
        }
    
    def _write_midi(self, notes, output_path):
        """Write notes to MIDI file."""
        tempo_bpm = 120
        ticks_per_beat = 480
        us_per_beat = int(60_000_000 / tempo_bpm)
        
        notes = sorted(notes, key=lambda x: x[1])
        
        track_data = bytearray()
        current_tick = 0
        
        for midi_note, start_time, duration in notes:
            start_tick = int(start_time * tempo_bpm / 60 * ticks_per_beat)
            duration_ticks = max(1, int(duration * tempo_bpm / 60 * ticks_per_beat))
            
            delta = start_tick - current_tick
            track_data.extend(self._var_length(delta))
            track_data.extend([0x90, midi_note, 100])
            current_tick = start_tick
            
            track_data.extend(self._var_length(duration_ticks))
            track_data.extend([0x80, midi_note, 0])
            current_tick += duration_ticks
        
        track_data.extend([0x00, 0xFF, 0x2F, 0x00])
        
        header = b'MThd' + struct.pack('>I', 6) + struct.pack('>HHH', 1, 2, ticks_per_beat)
        
        tempo_data = bytearray([0x00, 0xFF, 0x51, 0x03])
        tempo_data.extend([(us_per_beat >> 16) & 0xFF, (us_per_beat >> 8) & 0xFF, us_per_beat & 0xFF])
        tempo_data.extend([0x00, 0xFF, 0x2F, 0x00])
        tempo_track = b'MTrk' + struct.pack('>I', len(tempo_data)) + bytes(tempo_data)
        
        melody_track = b'MTrk' + struct.pack('>I', len(track_data)) + bytes(track_data)
        
        with open(output_path, 'wb') as f:
            f.write(header + tempo_track + melody_track)
    
    def _var_length(self, value):
        result = []
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))


def process_wav_streaming(wav_path, mid_path):
    """Main entry point."""
    extractor = FastMelodyExtractor()
    return extractor.extract_melody(wav_path, mid_path)


if __name__ == '__main__':
    test_files = [
        'test_music/0232a66e-d766-4dbc-bd9e-ee1c219c2027.wav',
        'test_music/16349331-6143-41bb-9caa-268ced3a9b3b.wav',
        'test_music/48752d06-6c23-4a22-b64b-232394156bae.wav',
        'test_music/996a38c8-48de-4a7c-b87f-6b20bc189054.wav'
    ]
    
    print("=" * 60)
    print("FAST MELODY EXTRACTOR - Real Music Test")
    print("=" * 60)
    
    total_notes = 0
    total_duration = 0
    total_time = 0
    max_mem = 0
    
    for wav_path in test_files:
        if not os.path.exists(wav_path):
            print(f"SKIP: {wav_path} not found")
            continue
        
        mid_path = wav_path.replace('.wav', '_fast.mid')
        result = process_wav_streaming(wav_path, mid_path)
        
        print(f"\n{os.path.basename(wav_path)}:")
        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Notes: {result['notes']}")
        print(f"  Speed: {result['rtf']:.1f}x real-time")
        print(f"  Memory: {result['memory_mb']:.1f} MB")
        
        total_notes += result['notes']
        total_duration += result['duration']
        total_time += result['processing_time']
        max_mem = max(max_mem, result['memory_mb'])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_speed = total_duration / total_time if total_time > 0 else 0
    print(f"Total notes: {total_notes}")
    print(f"Avg speed: {avg_speed:.1f}x real-time (req: >=4x)")
    print(f"Peak memory: {max_mem:.1f} MB (req: <500 MB)")
    
    if avg_speed >= 4 and max_mem < 500:
        print("\n*** PASS - Meets all requirements ***")
    else:
        print("\n*** FAIL - Does not meet requirements ***")
