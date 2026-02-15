#!/usr/bin/env python3
"""
Real Music Melody Extractor - Smooth Version
Extracts actual melody notes, not rapid pitch fluctuations.
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
HOP_SIZE = 512  # Smaller hop for better time resolution


class SmoothMelodyExtractor:
    """
    Melody extractor that produces natural-sounding note sequences.
    
    Key improvements:
    1. Median filtering to smooth pitch contour
    2. Longer minimum note duration (200ms)
    3. Pitch stability requirement (note must hold for multiple frames)
    4. Semitone tolerance to avoid micro-variations
    """
    
    def __init__(self, min_freq=100, max_freq=800, min_note_duration=0.2):
        self.min_freq = min_freq  # Higher min to focus on melody range
        self.max_freq = max_freq  # Lower max to avoid harmonics
        self.min_note_duration = min_note_duration  # 200ms minimum
        
    def _freq_to_midi(self, freq):
        if freq is None or freq <= 0:
            return None
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 48 <= midi_rounded <= 84:  # C3 to C6 - typical melody range
            return midi_rounded
        return None
    
    def _detect_pitch(self, chunk, sr):
        """Pitch detection with higher confidence threshold."""
        n = len(chunk)
        if n < 512:
            return None, 0
        
        # Energy check - higher threshold
        energy = np.dot(chunk, chunk) / n
        if energy < 1e-5:
            return None, 0
        
        # Normalize
        chunk = chunk - np.mean(chunk)
        
        # FFT-based autocorrelation
        fft = np.fft.rfft(chunk, n=2*n)
        power = fft.real**2 + fft.imag**2
        corr = np.fft.irfft(power)[:n]
        
        if corr[0] <= 0:
            return None, 0
        corr = corr / corr[0]
        
        # Search range
        min_lag = max(2, int(sr / self.max_freq))
        max_lag = min(n // 2, int(sr / self.min_freq))
        
        if max_lag <= min_lag + 5:
            return None, 0
        
        search = corr[min_lag:max_lag]
        
        # Find peak
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]
        
        # Higher threshold for cleaner detection
        if peak_val > 0.5:
            lag = peak_idx + min_lag
            return sr / lag, peak_val
        return None, 0
    
    def _median_filter(self, pitches, window=5):
        """Apply median filter to smooth pitch sequence."""
        result = np.zeros_like(pitches)
        half = window // 2
        
        for i in range(len(pitches)):
            start = max(0, i - half)
            end = min(len(pitches), i + half + 1)
            window_vals = pitches[start:end]
            # Only use non-zero values
            nonzero = window_vals[window_vals > 0]
            if len(nonzero) > 0:
                result[i] = np.median(nonzero)
            else:
                result[i] = 0
        
        return result
    
    def _pitches_to_stable_notes(self, midi_notes, hop_duration):
        """
        Convert MIDI note sequence to stable notes.
        Requires notes to be held for minimum duration.
        """
        notes = []
        
        if len(midi_notes) == 0:
            return notes
        
        # Group consecutive same notes
        current_note = midi_notes[0]
        note_start = 0
        note_frames = 1
        
        min_frames = int(self.min_note_duration / hop_duration)
        
        for i in range(1, len(midi_notes)):
            # Allow 1 semitone tolerance for held notes
            if midi_notes[i] != 0 and current_note != 0:
                if abs(midi_notes[i] - current_note) <= 1:
                    note_frames += 1
                    continue
            
            if midi_notes[i] == current_note:
                note_frames += 1
            else:
                # End current note if it's long enough
                if current_note != 0 and note_frames >= min_frames:
                    duration = note_frames * hop_duration
                    notes.append((int(current_note), note_start * hop_duration, duration))
                
                # Start new note
                current_note = midi_notes[i]
                note_start = i
                note_frames = 1
        
        # Handle final note
        if current_note != 0 and note_frames >= min_frames:
            duration = note_frames * hop_duration
            notes.append((int(current_note), note_start * hop_duration, duration))
        
        return notes
    
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
        hop_duration = HOP_SIZE / sr
        
        # Detect pitches
        raw_pitches = []
        confidences = []
        
        for i in range(0, len(samples) - CHUNK_SIZE, HOP_SIZE):
            chunk = samples[i:i + CHUNK_SIZE]
            freq, conf = self._detect_pitch(chunk, sr)
            raw_pitches.append(freq if freq else 0)
            confidences.append(conf)
        
        raw_pitches = np.array(raw_pitches)
        confidences = np.array(confidences)
        
        # Apply median filter to smooth
        smoothed_pitches = self._median_filter(raw_pitches, window=7)
        
        # Convert to MIDI notes
        midi_notes = np.array([self._freq_to_midi(f) or 0 if f > 0 else 0 for f in smoothed_pitches], dtype=np.float64)
        
        # Apply another median filter on MIDI notes
        midi_notes_smooth = np.zeros_like(midi_notes)
        for i in range(len(midi_notes)):
            start = max(0, i - 3)
            end = min(len(midi_notes), i + 4)
            window = midi_notes[start:end]
            nonzero = window[window > 0]
            if len(nonzero) >= 3:  # Need at least 3 valid notes in window
                midi_notes_smooth[i] = int(np.median(nonzero))
        
        # Convert to stable notes
        notes = self._pitches_to_stable_notes(midi_notes_smooth, hop_duration)
        
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
    extractor = SmoothMelodyExtractor()
    return extractor.extract_melody(wav_path, mid_path)


if __name__ == '__main__':
    test_files = [
        'test_music/0232a66e-d766-4dbc-bd9e-ee1c219c2027.wav',
        'test_music/16349331-6143-41bb-9caa-268ced3a9b3b.wav',
        'test_music/48752d06-6c23-4a22-b64b-232394156bae.wav',
        'test_music/996a38c8-48de-4a7c-b87f-6b20bc189054.wav'
    ]
    
    print("=" * 60)
    print("SMOOTH MELODY EXTRACTOR - Real Music Test")
    print("=" * 60)
    
    total_notes = 0
    total_duration = 0
    total_time = 0
    max_mem = 0
    
    for wav_path in test_files:
        if not os.path.exists(wav_path):
            print(f"SKIP: {wav_path} not found")
            continue
        
        mid_path = wav_path.replace('.wav', '_smooth.mid')
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
    
    avg_notes_per_sec = total_notes / total_duration if total_duration > 0 else 0
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_speed = total_duration / total_time if total_time > 0 else 0
    print(f"Total notes: {total_notes}")
    print(f"Notes per second: {avg_notes_per_sec:.2f} (typical melody: 1-4)")
    print(f"Avg speed: {avg_speed:.1f}x real-time (req: >=4x)")
    print(f"Peak memory: {max_mem:.1f} MB (req: <500 MB)")
    
    if avg_speed >= 4 and max_mem < 500:
        print("\n*** PASS - Meets all requirements ***")
    else:
        print("\n*** FAIL - Does not meet requirements ***")
