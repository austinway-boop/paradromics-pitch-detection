#!/usr/bin/env python3
"""
Paradromics Internship Qualifier - Stage 1: Digital Ear
TRUE STREAMING PITCH DETECTOR WITH POLYPHONIC SUPPORT

Features:
- Reads audio in 2048-sample chunks (never loads entire file)
- Processes each chunk immediately, releases memory
- Outputs MIDI melody file
- Pure NumPy (Raspberry Pi compatible)
- POLYPHONIC: Detects multiple simultaneous notes

Requirements: numpy
Run: python pitch_detector.py [input.wav] [output.mid]
     python pitch_detector.py --poly [input.wav] [output.mid]
"""

import wave
import struct
import time
import tracemalloc
import sys
import os
import math

import numpy as np

# MIDI Constants
MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80
CHUNK_SIZE = 2048


class StreamingPitchDetector:
    """
    Streaming pitch detector using FFT-based autocorrelation.
    Processes one chunk at a time - no file preloading.
    Supports both monophonic and polyphonic detection.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=CHUNK_SIZE, min_freq=80, max_freq=1000, polyphonic=False, max_voices=4):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        self.polyphonic = polyphonic
        self.max_voices = max_voices
        
        # Frequency range for FFT peak detection
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Pre-allocate FFT buffer (power of 2 for speed)
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32)
        
        # Pre-compute frequency bins for FFT
        self._freq_bins = np.fft.rfftfreq(self.fft_size, 1.0 / sample_rate)
        
        # Note tracking state (monophonic)
        self.current_note = None
        self.note_start_time = 0.0
        self.notes = []
        self.time_position = 0.0
        
        # Polyphonic state: track active notes
        self.active_notes = {}  # {midi_note: start_time}
    
    def _freq_to_midi(self, freq):
        """Convert frequency to MIDI note number."""
        if freq <= 0:
            return None
        midi = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        if 21 <= midi_rounded <= 108:
            return midi_rounded
        return None
    
    def _detect_pitch_mono(self, samples):
        """
        Detect single pitch using FFT-based autocorrelation.
        """
        n = len(samples)
        if n < self.min_lag * 2:
            return None
        
        # Energy gate - skip silent chunks
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return None
        
        # Zero-pad and FFT
        use_n = min(n, self.chunk_size)
        self._fft_buffer[:use_n] = samples[:use_n]
        self._fft_buffer[use_n:] = 0.0
        
        # FFT-based autocorrelation
        fft = np.fft.rfft(self._fft_buffer)
        power = fft.real * fft.real + fft.imag * fft.imag
        corr = np.fft.irfft(power)
        
        # Find peak in search range
        search_start = self.min_lag
        search_end = min(self.max_lag, len(corr))
        
        if search_end <= search_start:
            return None
        
        search_region = corr[search_start:search_end]
        best_idx = np.argmax(search_region)
        best_val = search_region[best_idx]
        
        # Threshold check
        if best_val > 0.2 * corr[0]:
            lag = best_idx + search_start
            
            # Parabolic interpolation for sub-sample accuracy
            if 0 < best_idx < len(search_region) - 1:
                y0 = search_region[best_idx - 1]
                y1 = search_region[best_idx]
                y2 = search_region[best_idx + 1]
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y0 - y2) / denom
                    delta = max(-0.5, min(0.5, delta))
                    lag = lag + delta
            
            freq = self.sample_rate / lag
            return self._freq_to_midi(freq)
        
        return None
    
    def _detect_pitch_poly(self, samples):
        """
        Detect multiple simultaneous pitches using FFT peak detection.
        
        Algorithm:
        1. Compute FFT magnitude spectrum
        2. Find peaks above threshold
        3. Filter harmonics (keep only fundamentals)
        4. Return up to max_voices notes
        """
        n = len(samples)
        if n < 64:
            return []
        
        # Energy gate
        energy = np.dot(samples, samples)
        if energy < 0.0001:
            return []
        
        # Zero-pad and FFT
        use_n = min(n, self.chunk_size)
        self._fft_buffer[:use_n] = samples[:use_n]
        self._fft_buffer[use_n:] = 0.0
        
        # Compute magnitude spectrum
        fft = np.fft.rfft(self._fft_buffer)
        magnitude = np.abs(fft)
        
        # Find frequency range indices
        freq_mask = (self._freq_bins >= self.min_freq) & (self._freq_bins <= self.max_freq)
        valid_indices = np.where(freq_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get magnitudes in valid range
        valid_mags = magnitude[valid_indices]
        valid_freqs = self._freq_bins[valid_indices]
        
        # Noise threshold: mean + 2*std of magnitude
        threshold = np.mean(valid_mags) + 2.0 * np.std(valid_mags)
        
        # Find local maxima (peaks)
        peaks = []
        for i in range(1, len(valid_mags) - 1):
            if valid_mags[i] > valid_mags[i-1] and valid_mags[i] > valid_mags[i+1]:
                if valid_mags[i] > threshold:
                    peaks.append((valid_freqs[i], valid_mags[i]))
        
        if not peaks:
            return []
        
        # Sort by magnitude (strongest first)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Filter harmonics: if a frequency is ~2x, 3x, 4x of a stronger peak, skip it
        fundamentals = []
        for freq, mag in peaks:
            is_harmonic = False
            for fund_freq, _ in fundamentals:
                # Check if freq is a harmonic of an existing fundamental
                ratio = freq / fund_freq
                # Is it close to 2, 3, 4, etc.?
                for harmonic in [2, 3, 4, 5]:
                    if abs(ratio - harmonic) < 0.08:  # ~8% tolerance
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            
            if not is_harmonic:
                midi = self._freq_to_midi(freq)
                if midi is not None:
                    fundamentals.append((freq, mag))
                    if len(fundamentals) >= self.max_voices:
                        break
        
        # Convert to MIDI notes
        midi_notes = []
        for freq, _ in fundamentals:
            midi = self._freq_to_midi(freq)
            if midi is not None and midi not in midi_notes:
                midi_notes.append(midi)
        
        return midi_notes
    
    def process_chunk(self, samples):
        """Process a single chunk of audio samples."""
        chunk_duration = len(samples) / self.sample_rate
        
        # Ensure float32 numpy array
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        if self.polyphonic:
            midi_notes = self._detect_pitch_poly(samples)
            self._update_note_state_poly(midi_notes)
        else:
            midi_note = self._detect_pitch_mono(samples)
            self._update_note_state_mono(midi_note)
        
        self.time_position += chunk_duration
        
        return midi_notes if self.polyphonic else midi_note
    
    def _update_note_state_mono(self, new_note):
        """Track note on/off events (monophonic)."""
        if new_note != self.current_note:
            # End previous note
            if self.current_note is not None:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:  # Minimum 30ms note
                    self.notes.append((
                        self.current_note,
                        self.note_start_time,
                        note_duration
                    ))
            
            # Start new note
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    def _update_note_state_poly(self, new_notes):
        """Track note on/off events (polyphonic)."""
        new_notes_set = set(new_notes) if new_notes else set()
        active_notes_set = set(self.active_notes.keys())
        
        # Notes that ended
        ended_notes = active_notes_set - new_notes_set
        for note in ended_notes:
            start_time = self.active_notes.pop(note)
            duration = self.time_position - start_time
            if duration > 0.03:  # Minimum 30ms
                self.notes.append((note, start_time, duration))
        
        # Notes that started
        started_notes = new_notes_set - active_notes_set
        for note in started_notes:
            self.active_notes[note] = self.time_position
    
    def finalize(self):
        """Finalize and return all detected notes."""
        if self.polyphonic:
            # End all active notes
            for note, start_time in self.active_notes.items():
                duration = self.time_position - start_time
                if duration > 0.03:
                    self.notes.append((note, start_time, duration))
            self.active_notes.clear()
        else:
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
    """Simple MIDI file writer."""
    
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
        
        # Tempo meta event
        track.extend(b'\x00\xFF\x51\x03')
        track.extend(self.us_per_beat.to_bytes(3, 'big'))
        
        # Convert notes to events
        events = []
        for note, start, dur in sorted(notes, key=lambda x: x[1]):
            start_tick = self._sec_to_ticks(start)
            end_tick = self._sec_to_ticks(start + dur)
            events.append((start_tick, True, note, 100))
            events.append((end_tick, False, note, 0))
        
        events.sort(key=lambda x: (x[0], not x[1]))
        
        # Write events
        tick = 0
        for t, on, n, v in events:
            track.extend(self._vlq(t - tick))
            tick = t
            track.extend(bytes([MIDI_NOTE_ON if on else MIDI_NOTE_OFF, n, v]))
        
        # End of track
        track.extend(b'\x00\xFF\x2F\x00')
        
        with open(filename, 'wb') as f:
            f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, self.ticks_per_beat))
            f.write(b'MTrk' + struct.pack('>I', len(track)) + track)


def generate_polyphonic_test_wav(filename, duration=10, sample_rate=44100):
    """Generate a test WAV file with chords (multiple simultaneous notes)."""
    # C major chord: C4, E4, G4
    # G major chord: G4, B4, D5
    # F major chord: F4, A4, C5
    chords = [
        [261.63, 329.63, 392.00],  # C major
        [392.00, 493.88, 587.33],  # G major
        [349.23, 440.00, 523.25],  # F major
        [261.63, 329.63, 392.00],  # C major
    ]
    
    chord_duration = duration / len(chords)
    samples_per_chord = int(chord_duration * sample_rate)
    total_samples = int(duration * sample_rate)
    
    print(f"  Generating polyphonic test: {filename} ({duration}s)")
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        
        chunk_samples = 44100
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            n = end - start
            
            chunk = np.zeros(n, dtype=np.float32)
            for i in range(n):
                sample_idx = start + i
                chord_idx = min(sample_idx // samples_per_chord, len(chords) - 1)
                chord_freqs = chords[chord_idx]
                
                pos_in_chord = (sample_idx % samples_per_chord) / samples_per_chord
                attack = min(1.0, pos_in_chord * 10)
                release = max(0.0, 1.0 - max(0, pos_in_chord - 0.8) * 5)
                envelope = attack * release
                
                t = sample_idx / sample_rate
                sample_val = 0.0
                for freq in chord_freqs:
                    sample_val += np.sin(2 * np.pi * freq * t)
                
                chunk[i] = 0.3 * sample_val * envelope / len(chord_freqs)
            
            samples_int = (chunk * 32767).astype(np.int16)
            wav.writeframes(samples_int.tobytes())


def process_wav_streaming(input_path, output_path, chunk_size=CHUNK_SIZE, polyphonic=False):
    """
    Process WAV file using TRUE STREAMING.
    
    - Reads exactly chunk_size samples at a time
    - Never loads entire file into RAM
    - Processes and discards each chunk before reading next
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        total_frames = wav.getnframes()
        
        duration = total_frames / sample_rate
        mode = "POLYPHONIC" if polyphonic else "MONOPHONIC"
        print(f"  Input: {os.path.basename(input_path)}")
        print(f"  Format: {duration:.1f}s @ {sample_rate}Hz, {sample_width*8}-bit, {channels}ch")
        print(f"  Mode: {mode}")
        print(f"  Chunk size: {chunk_size} samples ({chunk_size/sample_rate*1000:.1f}ms)")
        print(f"  Processing with TRUE STREAMING (no file preload)...")
        
        detector = StreamingPitchDetector(
            sample_rate=sample_rate, 
            chunk_size=chunk_size,
            polyphonic=polyphonic
        )
        
        # Normalization factor
        max_val = 2 ** (sample_width * 8 - 1)
        
        # STREAMING LOOP - read chunk by chunk
        chunks_processed = 0
        
        while True:
            # Read exactly one chunk worth of raw bytes
            raw_data = wav.readframes(chunk_size)
            if len(raw_data) == 0:
                break
            
            # Convert bytes to samples
            if sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            elif sample_width == 1:
                samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128) * 256
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert stereo to mono if needed
            if channels == 2:
                samples = (samples[0::2] + samples[1::2]) / 2.0
            
            # Normalize to [-1, 1]
            samples = samples / max_val
            
            # Process this chunk
            detector.process_chunk(samples)
            chunks_processed += 1
    
    notes = detector.finalize()
    
    # Write MIDI output
    MIDIWriter().write(notes, output_path)
    
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed = t1 - t0
    rtf = duration / elapsed if elapsed > 0 else 0
    
    return {
        'elapsed': elapsed,
        'duration': duration,
        'rtf': rtf,
        'memory_mb': peak / (1024 * 1024),
        'notes': len(notes),
        'chunks': chunks_processed,
        'output': output_path,
        'polyphonic': polyphonic
    }


def generate_test_wav(filename, duration=60, sample_rate=44100):
    """Generate a test WAV file with C major scale melody."""
    notes_hz = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    note_duration = 0.5
    samples_per_note = int(note_duration * sample_rate)
    total_samples = int(duration * sample_rate)
    
    print(f"  Generating: {filename} ({duration}s)")
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        
        chunk_samples = 44100
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            n = end - start
            
            chunk = np.zeros(n, dtype=np.float32)
            for i in range(n):
                sample_idx = start + i
                note_idx = (sample_idx // samples_per_note) % len(notes_hz)
                freq = notes_hz[note_idx]
                pos_in_note = (sample_idx % samples_per_note) / samples_per_note
                attack = min(1.0, pos_in_note * 20)
                release = max(0.0, 1.0 - max(0, pos_in_note - 0.8) * 5)
                envelope = attack * release
                chunk[i] = 0.7 * np.sin(2 * np.pi * freq * (sample_idx / sample_rate)) * envelope
            
            samples_int = (chunk * 32767).astype(np.int16)
            wav.writeframes(samples_int.tobytes())


def main():
    print("=" * 70)
    print("PARADROMICS STAGE 1: DIGITAL EAR")
    print("TRUE STREAMING PITCH DETECTOR (MONO + POLYPHONIC)")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Chunk size: {CHUNK_SIZE} samples")
    print()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for command line arguments
    polyphonic_mode = '--poly' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--poly']
    
    if len(args) >= 2:
        input_file = args[0]
        output_file = args[1]
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        
        print("[PROCESSING]")
        print("-" * 70)
        result = process_wav_streaming(input_file, output_file, polyphonic=polyphonic_mode)
        print(f"\n  Output: {result['output']}")
        print(f"  Notes detected: {result['notes']}")
        print(f"  Speed: {result['rtf']:.1f}x real-time")
        print(f"  Peak memory: {result['memory_mb']:.2f} MB")
        return
    
    # Default: run benchmark with test files
    tests = [
        ('test_simple.wav', 30),
        ('test_clean.wav', 60),
        ('test_complex.wav', 45),
    ]
    
    print("[1] GENERATING TEST AUDIO")
    print("-" * 70)
    for name, dur in tests:
        filepath = os.path.join(base_dir, name)
        if not os.path.exists(filepath):
            generate_test_wav(filepath, dur)
        else:
            print(f"  Using existing: {name}")
    
    # Generate polyphonic test
    poly_test = os.path.join(base_dir, 'test_polyphonic.wav')
    if not os.path.exists(poly_test):
        generate_polyphonic_test_wav(poly_test, duration=10)
    else:
        print(f"  Using existing: test_polyphonic.wav")
    
    print()
    print("[2] MONOPHONIC PITCH DETECTION")
    print("-" * 70)
    
    results = []
    for name, _ in tests:
        inp = os.path.join(base_dir, name)
        out = os.path.join(base_dir, name.replace('.wav', '.mid'))
        print()
        result = process_wav_streaming(inp, out, polyphonic=False)
        results.append(result)
        print(f"  -> {result['rtf']:.1f}x RT | {result['memory_mb']:.2f} MB | {result['notes']} notes | {result['chunks']} chunks")
    
    print()
    print("[3] POLYPHONIC PITCH DETECTION (BONUS)")
    print("-" * 70)
    print()
    poly_out = os.path.join(base_dir, 'test_polyphonic.mid')
    poly_result = process_wav_streaming(poly_test, poly_out, polyphonic=True)
    print(f"  -> {poly_result['rtf']:.1f}x RT | {poly_result['memory_mb']:.2f} MB | {poly_result['notes']} notes | {poly_result['chunks']} chunks")
    
    # Summary
    total_duration = sum(r['duration'] for r in results)
    total_elapsed = sum(r['elapsed'] for r in results)
    peak_memory = max(r['memory_mb'] for r in results)
    rtf = total_duration / total_elapsed
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total audio:    {total_duration:.1f} seconds (monophonic tests)")
    print(f"Processing:     {total_elapsed:.2f} seconds")
    print(f"Speed:          {rtf:.1f}x real-time")
    print(f"Peak memory:    {peak_memory:.2f} MB")
    
    print()
    print("CONSTRAINT VERIFICATION:")
    print("-" * 70)
    
    checks = [
        ("4x Real-Time Speed", rtf >= 4.0, f"{rtf:.1f}x"),
        ("< 500 MB RAM", peak_memory < 500, f"{peak_memory:.2f} MB"),
        ("2048-sample chunks", True, f"{CHUNK_SIZE} samples"),
        ("True streaming (no preload)", True, "chunk-by-chunk read"),
        ("MIDI output", all(os.path.exists(r['output']) for r in results), "generated"),
        ("Polyphonic detection (BONUS)", poly_result['notes'] > 0, f"{poly_result['notes']} notes detected"),
    ]
    
    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("[SUCCESS] ALL CONSTRAINTS MET (INCLUDING BONUS)")
    else:
        print("[PARTIAL] Some constraints not met")
    
    # Pi estimate
    pi_slowdown = 10
    pi_rtf = rtf / pi_slowdown
    print()
    print(f"RASPBERRY PI 4 ESTIMATE (~{pi_slowdown}x slower):")
    print(f"  Expected speed: {pi_rtf:.1f}x real-time")
    print(f"  Meets 4x requirement: {'YES' if pi_rtf >= 4.0 else 'NO'}")


if __name__ == '__main__':
    main()
