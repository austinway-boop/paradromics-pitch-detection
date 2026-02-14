#!/usr/bin/env python3
"""
PARADROMICS OPTIMIZATION - Method 7: Process Every 2nd Chunk

Skip processing every other chunk to double speed.
Interpolate pitch between processed chunks.

Target: 9x+ RT (up from 4.7x RT baseline)
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
    print("Warning: NumPy not available")

MIDI_NOTE_ON = 0x90
MIDI_NOTE_OFF = 0x80


class SkipChunkPitchDetector:
    """Pitch detector that processes every Nth chunk and interpolates."""
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_freq=80, max_freq=1000, skip_factor=2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_lag = max(1, int(sample_rate / max_freq))
        self.max_lag = min(chunk_size, int(sample_rate / min_freq))
        self.skip_factor = skip_factor  # Process every Nth chunk
        
        # Pre-allocate FFT arrays
        self.fft_size = 4096
        self._fft_buffer = np.zeros(self.fft_size, dtype=np.float32) if HAS_NUMPY else None
        
        self.current_note = None
        self.note_start_time = 0
        self.notes = []
        self.time_position = 0.0
        
        # Skip chunk state
        self.chunk_counter = 0
        self.last_detected_pitch = None
        self.next_detected_pitch = None
        self.pending_chunks = []  # Store (time_pos, duration) for skipped chunks
        
    def _detect_pitch_numpy(self, samples):
        """FFT-based autocorrelation pitch detection."""
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
    
    def _interpolate_pitch(self, pitch1, pitch2, ratio):
        """Interpolate between two MIDI pitches.
        
        For melodic content, we use nearest-neighbor interpolation
        (pick the closer pitch) since MIDI notes are discrete.
        This preserves note transitions better than linear interpolation.
        """
        if pitch1 is None:
            return pitch2
        if pitch2 is None:
            return pitch1
        
        # Simple strategy: use the pitch that's closer in the timeline
        # ratio = 0 means closer to pitch1, ratio = 1 means closer to pitch2
        if ratio < 0.5:
            return pitch1
        else:
            return pitch2
    
    def process_chunk(self, samples):
        """Process chunk with skip optimization."""
        chunk_duration = len(samples) / self.sample_rate
        self.chunk_counter += 1
        
        # Should we actually process this chunk?
        should_process = (self.chunk_counter % self.skip_factor) == 1
        
        if should_process:
            # Actually detect pitch
            if HAS_NUMPY:
                samples = np.asarray(samples, dtype=np.float32)
                midi_note = self._detect_pitch_numpy(samples)
            else:
                midi_note = None  # Pure python fallback not implemented for demo
            
            # Shift the detected pitches
            self.last_detected_pitch = self.next_detected_pitch
            self.next_detected_pitch = midi_note
            
            # Process any pending skipped chunks with interpolation
            if self.pending_chunks and self.last_detected_pitch is not None:
                total_pending = len(self.pending_chunks)
                for i, (t_pos, t_dur) in enumerate(self.pending_chunks):
                    ratio = (i + 1) / (total_pending + 1)
                    interp_note = self._interpolate_pitch(
                        self.last_detected_pitch, 
                        self.next_detected_pitch, 
                        ratio
                    )
                    self._update_note_state_at_time(interp_note, t_pos, t_dur)
            
            self.pending_chunks = []
            
            # Update with current detected note
            self._update_note_state(midi_note, chunk_duration)
        else:
            # Skip processing, queue for interpolation
            self.pending_chunks.append((self.time_position, chunk_duration))
        
        self.time_position += chunk_duration
        
        return self.next_detected_pitch if should_process else None
    
    def _update_note_state_at_time(self, new_note, time_pos, duration):
        """Update note state at a specific time (for interpolated chunks)."""
        # Save current state
        orig_time = self.time_position
        self.time_position = time_pos
        
        self._update_note_state(new_note, duration)
        
        # Restore
        self.time_position = orig_time
    
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
        # Process any remaining pending chunks
        if self.pending_chunks:
            for t_pos, t_dur in self.pending_chunks:
                # Use last known pitch for remaining chunks
                self._update_note_state_at_time(self.next_detected_pitch, t_pos, t_dur)
        
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


def process_wav_skip_chunks(input_path, output_path, chunk_size=2048, skip_factor=2):
    """Process WAV with skip-chunk optimization."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz (skip_factor={skip_factor})")
        
        detector = SkipChunkPitchDetector(
            sample_rate=sr, 
            chunk_size=chunk_size,
            skip_factor=skip_factor
        )
        
        chunks_processed = 0
        chunks_skipped = 0
        
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
            
            result = detector.process_chunk(samples)
            
            if result is not None:
                chunks_processed += 1
            else:
                chunks_skipped += 1
        
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
        'output': output_path,
        'chunks_processed': chunks_processed,
        'chunks_skipped': chunks_skipped,
        'skip_factor': skip_factor
    }


def generate_test_wav(filename, duration=60, sr=44100):
    """Generate test WAV with melody."""
    if HAS_NUMPY:
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
    else:
        return
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())
    
    print(f"  Generated: {os.path.basename(filename)} ({duration}s)")


def main():
    print("=" * 60)
    print("METHOD 7: SKIP CHUNK OPTIMIZATION")
    print("Process every 2nd chunk, interpolate pitch for skipped chunks")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate test files
    tests = [
        ('test_skip_30s.wav', 30),
        ('test_skip_60s.wav', 60),
        ('test_skip_45s.wav', 45),
    ]
    
    print("\n[1] Generating Test Audio")
    print("-" * 40)
    for name, dur in tests:
        generate_test_wav(os.path.join(base_dir, name), dur)
    
    # Test different skip factors
    skip_factors = [2, 3, 4]
    
    for skip_factor in skip_factors:
        print(f"\n[2] Processing with skip_factor={skip_factor}")
        print("-" * 40)
        
        results = []
        for name, _ in tests:
            inp = os.path.join(base_dir, name)
            out = os.path.join(base_dir, name.replace('.wav', f'_skip{skip_factor}.mid'))
            m = process_wav_skip_chunks(inp, out, skip_factor=skip_factor)
            results.append(m)
            print(f"    -> {m['rtf']:.1f}x RT | {m['memory_mb']:.1f}MB | {m['notes']} notes")
            print(f"       Processed: {m['chunks_processed']}, Skipped: {m['chunks_skipped']}")
        
        total_dur = sum(r['duration'] for r in results)
        total_time = sum(r['elapsed'] for r in results)
        max_mem = max(r['memory_mb'] for r in results)
        
        print(f"\n  SKIP FACTOR {skip_factor} RESULTS:")
        print(f"  Speed: {total_dur/total_time:.1f}x real-time")
        print(f"  Peak memory: {max_mem:.2f} MB")
        
        # Check if we hit 9x
        if total_dur/total_time >= 9.0:
            print(f"  ✅ TARGET HIT: {total_dur/total_time:.1f}x >= 9x RT")
        else:
            print(f"  ❌ Below target: {total_dur/total_time:.1f}x < 9x RT")
    
    print("\n" + "=" * 60)
    print("SUMMARY: Skip-chunk processing results")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
