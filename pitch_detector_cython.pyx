# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
Paradromics Pitch Detection - Cython Implementation
====================================================

Optimized for Raspberry Pi ARM architecture where:
- Numba has poor ARM support (no LLVM ARM backend in older versions)
- Cython compiles to native C code via GCC (excellent ARM support)
- Can leverage NEON SIMD intrinsics on Pi 4+

This implementation provides:
1. Pure C-level zero-crossing rate (ZCR) detection
2. Inline lowpass filtering to reduce noise
3. Memory-efficient streaming processing
4. Sub-sample interpolation for accuracy

Target: Beat 4.7x real-time on Raspberry Pi 4
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, log2, round, sqrt
from libc.stdlib cimport malloc, free

# NumPy type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.int32_t INT32_t


# =============================================================================
# METHOD 1: Pure C ZCR (Fastest, O(n))
# =============================================================================

cpdef int zcr_count_crossings(DTYPE_t[:] signal) noexcept nogil:
    """
    Count zero crossings in signal.
    Pure C implementation - no Python overhead.
    """
    cdef:
        int i, n, crossings
        DTYPE_t prev, curr
    
    n = signal.shape[0]
    if n < 2:
        return 0
    
    crossings = 0
    prev = signal[0]
    
    for i in range(1, n):
        curr = signal[i]
        if prev * curr < 0:
            crossings += 1
        prev = curr
    
    return crossings


cpdef double zcr_frequency(DTYPE_t[:] signal, int sample_rate=44100) noexcept:
    """
    Compute frequency from zero crossing rate.
    
    Frequency = crossings / (2 * duration)
    """
    cdef:
        int crossings
        double duration
    
    crossings = zcr_count_crossings(signal)
    duration = <double>signal.shape[0] / <double>sample_rate
    
    if duration <= 0:
        return 0.0
    
    return <double>crossings / (2.0 * duration)


# =============================================================================
# METHOD 2: ZCR with Inline Lowpass Filter (Best for noisy signals)
# =============================================================================

cpdef int zcr_filtered_count(DTYPE_t[:] signal, double alpha=0.15) noexcept nogil:
    """
    Zero crossing count with inline IIR lowpass filter.
    Single-pass O(n) - combines filtering and counting.
    
    IIR filter: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    
    alpha: filter coefficient (lower = more smoothing)
           0.10 = aggressive smoothing (low cutoff)
           0.15 = moderate smoothing (recommended)
           0.30 = light smoothing (high cutoff)
    """
    cdef:
        int i, n, crossings
        double prev_filtered, curr_filtered
        double one_minus_alpha
    
    n = signal.shape[0]
    if n < 2:
        return 0
    
    one_minus_alpha = 1.0 - alpha
    crossings = 0
    prev_filtered = signal[0]
    
    for i in range(1, n):
        # Apply lowpass filter inline
        curr_filtered = alpha * signal[i] + one_minus_alpha * prev_filtered
        
        # Check for zero crossing
        if prev_filtered * curr_filtered < 0:
            crossings += 1
        
        prev_filtered = curr_filtered
    
    return crossings


cpdef double zcr_filtered_frequency(DTYPE_t[:] signal, int sample_rate=44100, 
                                     double alpha=0.15) noexcept:
    """
    Frequency detection with inline lowpass filter.
    Best balance of speed and noise rejection.
    """
    cdef:
        int crossings
        double duration
    
    crossings = zcr_filtered_count(signal, alpha)
    duration = <double>signal.shape[0] / <double>sample_rate
    
    if duration <= 0:
        return 0.0
    
    return <double>crossings / (2.0 * duration)


# =============================================================================
# METHOD 3: Interpolated ZCR (Sub-sample accuracy)
# =============================================================================

cpdef double zcr_interpolated_frequency(DTYPE_t[:] signal, int sample_rate=44100,
                                         double alpha=0.15) noexcept:
    """
    Interpolated zero crossing with inline filtering.
    Uses linear interpolation for sub-sample crossing positions.
    More accurate for low frequencies.
    """
    cdef:
        int i, n, crossing_count
        double prev_filtered, curr_filtered, one_minus_alpha
        double s0, s1, crossing_pos
        double sum_intervals, last_crossing
        double avg_half_period
    
    n = signal.shape[0]
    if n < 2:
        return 0.0
    
    one_minus_alpha = 1.0 - alpha
    crossing_count = 0
    sum_intervals = 0.0
    last_crossing = -1.0
    prev_filtered = signal[0]
    
    for i in range(1, n):
        curr_filtered = alpha * signal[i] + one_minus_alpha * prev_filtered
        
        if prev_filtered * curr_filtered < 0:
            # Interpolate precise crossing position
            s0 = fabs(prev_filtered)
            s1 = fabs(curr_filtered)
            crossing_pos = <double>(i - 1) + s0 / (s0 + s1)
            
            # Track interval from last crossing
            if last_crossing >= 0:
                sum_intervals += crossing_pos - last_crossing
            
            last_crossing = crossing_pos
            crossing_count += 1
        
        prev_filtered = curr_filtered
    
    # Need at least 2 crossings for interval
    if crossing_count < 2:
        return 0.0
    
    # Average half-period from intervals
    avg_half_period = sum_intervals / <double>(crossing_count - 1)
    
    if avg_half_period <= 0:
        return 0.0
    
    # Frequency = sample_rate / (2 * half_period)
    return <double>sample_rate / (2.0 * avg_half_period)


# =============================================================================
# METHOD 4: Energy-gated ZCR (Ignores silence)
# =============================================================================

cpdef double compute_energy(DTYPE_t[:] signal) noexcept nogil:
    """Compute signal energy (sum of squares)."""
    cdef:
        int i, n
        double energy
    
    n = signal.shape[0]
    energy = 0.0
    
    for i in range(n):
        energy += signal[i] * signal[i]
    
    return energy


cpdef double zcr_energy_gated(DTYPE_t[:] signal, int sample_rate=44100,
                               double energy_threshold=0.0001,
                               double alpha=0.15) noexcept:
    """
    ZCR with energy gating - returns 0 if signal is too quiet.
    Prevents false detections on silence/noise floor.
    """
    cdef:
        double energy
    
    energy = compute_energy(signal)
    
    if energy < energy_threshold * signal.shape[0]:
        return 0.0
    
    return zcr_interpolated_frequency(signal, sample_rate, alpha)


# =============================================================================
# FREQUENCY TO MIDI CONVERSION
# =============================================================================

cpdef int freq_to_midi(double freq) noexcept nogil:
    """
    Convert frequency to MIDI note number.
    Returns -1 if out of piano range (21-108).
    
    MIDI formula: note = 69 + 12 * log2(freq / 440)
    """
    cdef:
        double midi_float
        int midi_note
    
    if freq <= 0:
        return -1
    
    midi_float = 69.0 + 12.0 * log2(freq / 440.0)
    midi_note = <int>round(midi_float)
    
    # Piano range: A0 (21) to C8 (108)
    if midi_note < 21 or midi_note > 108:
        return -1
    
    return midi_note


cpdef double midi_to_freq(int midi_note) noexcept nogil:
    """
    Convert MIDI note number to frequency.
    """
    if midi_note < 0:
        return 0.0
    return 440.0 * (2.0 ** ((<double>midi_note - 69.0) / 12.0))


# =============================================================================
# STREAMING PITCH DETECTOR CLASS
# =============================================================================

cdef class CythonPitchDetector:
    """
    Streaming pitch detector optimized for Raspberry Pi.
    
    Uses Cython-compiled ZCR with:
    - Inline lowpass filtering
    - Interpolated zero crossings
    - Energy gating
    - Note state tracking
    """
    cdef:
        int sample_rate
        int chunk_size
        double alpha
        double energy_threshold
        double time_position
        double note_start_time
        int current_note
        list notes
        double smoothing
        double last_freq
    
    def __init__(self, int sample_rate=44100, int chunk_size=2048,
                 double alpha=0.15, double energy_threshold=0.0001,
                 double smoothing=0.5):
        """
        Initialize detector.
        
        Args:
            sample_rate: Audio sample rate (default 44100)
            chunk_size: Samples per chunk (default 2048)
            alpha: Lowpass filter coefficient (default 0.15)
            energy_threshold: Energy gate threshold (default 0.0001)
            smoothing: Frequency smoothing factor 0-1 (default 0.5)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.energy_threshold = energy_threshold
        self.time_position = 0.0
        self.note_start_time = 0.0
        self.current_note = -1
        self.notes = []
        self.smoothing = smoothing
        self.last_freq = 0.0
    
    cpdef int process_chunk(self, DTYPE_t[:] samples):
        """
        Process audio chunk and detect pitch.
        Returns MIDI note number or -1 if no note detected.
        """
        cdef:
            double chunk_duration
            double freq
            int midi_note
        
        chunk_duration = <double>samples.shape[0] / <double>self.sample_rate
        
        # Detect frequency with energy gating
        freq = zcr_energy_gated(samples, self.sample_rate, 
                                self.energy_threshold, self.alpha)
        
        # Apply smoothing
        if self.last_freq > 0 and freq > 0:
            freq = self.smoothing * self.last_freq + (1.0 - self.smoothing) * freq
        
        if freq > 0:
            self.last_freq = freq
        
        # Convert to MIDI
        midi_note = freq_to_midi(freq)
        
        # Track note state
        self._update_note_state(midi_note, chunk_duration)
        self.time_position += chunk_duration
        
        return midi_note
    
    cdef void _update_note_state(self, int new_note, double duration):
        """Track note on/off events for MIDI generation."""
        cdef double note_duration
        
        if new_note != self.current_note:
            # End previous note
            if self.current_note >= 0:
                note_duration = self.time_position - self.note_start_time
                if note_duration > 0.03:  # Min 30ms note
                    self.notes.append((
                        self.current_note,
                        self.note_start_time,
                        note_duration
                    ))
            
            # Start new note
            self.current_note = new_note
            self.note_start_time = self.time_position
    
    cpdef list finalize(self):
        """Finalize and return all detected notes."""
        cdef double note_duration
        
        if self.current_note >= 0:
            note_duration = self.time_position - self.note_start_time
            if note_duration > 0.03:
                self.notes.append((
                    self.current_note,
                    self.note_start_time,
                    note_duration
                ))
        
        return self.notes
    
    cpdef void reset(self):
        """Reset detector state for new audio."""
        self.time_position = 0.0
        self.note_start_time = 0.0
        self.current_note = -1
        self.notes = []
        self.last_freq = 0.0


# =============================================================================
# BATCH PROCESSING (For maximum throughput)
# =============================================================================

cpdef np.ndarray[INT32_t, ndim=1] process_batch(
    DTYPE_t[:, :] chunks,
    int sample_rate=44100,
    double alpha=0.15,
    double energy_threshold=0.0001
):
    """
    Process multiple audio chunks at once.
    
    Args:
        chunks: 2D array of shape (n_chunks, chunk_size)
        sample_rate: Audio sample rate
        alpha: Lowpass filter coefficient
        energy_threshold: Energy gate threshold
    
    Returns:
        Array of MIDI note numbers (-1 for no detection)
    """
    cdef:
        int i, n_chunks
        double freq
        np.ndarray[INT32_t, ndim=1] results
    
    n_chunks = chunks.shape[0]
    results = np.empty(n_chunks, dtype=np.int32)
    
    for i in range(n_chunks):
        freq = zcr_energy_gated(chunks[i], sample_rate, energy_threshold, alpha)
        results[i] = freq_to_midi(freq)
    
    return results


# =============================================================================
# UTILITY: MIDI FILE WRITER (Pure Python for simplicity)
# =============================================================================

def write_midi(notes, filename, tempo_bpm=120):
    """
    Write notes to MIDI file.
    
    Args:
        notes: List of (midi_note, start_time, duration) tuples
        filename: Output MIDI filename
        tempo_bpm: Tempo in BPM (default 120)
    """
    import struct
    
    MIDI_NOTE_ON = 0x90
    MIDI_NOTE_OFF = 0x80
    ticks_per_beat = 480
    us_per_beat = int(60_000_000 / tempo_bpm)
    
    def vlq(value):
        """Variable-length quantity encoding."""
        result = [value & 0x7F]
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))
    
    def sec_to_ticks(seconds):
        return int(seconds * (tempo_bpm / 60.0) * ticks_per_beat)
    
    track = bytearray()
    
    # Tempo meta event
    track.extend(b'\x00\xFF\x51\x03')
    track.extend(us_per_beat.to_bytes(3, 'big'))
    
    # Build events
    events = []
    for note, start, dur in sorted(notes, key=lambda x: x[1]):
        start_tick = sec_to_ticks(start)
        end_tick = sec_to_ticks(start + dur)
        events.append((start_tick, True, note, 100))   # Note on
        events.append((end_tick, False, note, 0))       # Note off
    
    # Sort: time first, note-off before note-on at same time
    events.sort(key=lambda x: (x[0], not x[1]))
    
    # Write events
    tick = 0
    for t, on, n, v in events:
        track.extend(vlq(t - tick))
        tick = t
        track.extend(bytes([MIDI_NOTE_ON if on else MIDI_NOTE_OFF, n, v]))
    
    # End of track
    track.extend(b'\x00\xFF\x2F\x00')
    
    # Write file
    with open(filename, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, ticks_per_beat))
        f.write(b'MTrk' + struct.pack('>I', len(track)) + track)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_wav(input_path, output_path, chunk_size=2048):
    """
    Process WAV file and generate MIDI.
    
    Args:
        input_path: Path to input WAV file
        output_path: Path to output MIDI file
        chunk_size: Processing chunk size (default 2048)
    
    Returns:
        Dictionary with timing and performance metrics
    """
    import wave
    import struct as struct_mod
    import time
    import tracemalloc
    import os
    
    tracemalloc.start()
    t0 = time.perf_counter()
    
    with wave.open(input_path, 'rb') as wav:
        sr = wav.getframerate()
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        nf = wav.getnframes()
        
        print(f"  {os.path.basename(input_path)}: {nf/sr:.1f}s @ {sr}Hz/{sw*8}bit")
        
        detector = CythonPitchDetector(
            sample_rate=sr,
            chunk_size=chunk_size,
            alpha=0.15,
            energy_threshold=0.0001,
            smoothing=0.5
        )
        
        while True:
            raw = wav.readframes(chunk_size)
            if not raw:
                break
            
            ns = len(raw) // (sw * ch)
            if sw == 2:
                samples = struct_mod.unpack(f'<{ns * ch}h', raw)
            else:
                samples = [b - 128 for b in raw]
            
            # Mix stereo to mono
            if ch == 2:
                samples = [(samples[i] + samples[i+1]) / 2 for i in range(0, len(samples), 2)]
            
            # Normalize to [-1, 1]
            max_val = 2 ** (sw * 8 - 1)
            samples = np.array([s / max_val for s in samples], dtype=np.float64)
            
            detector.process_chunk(samples)
        
        notes = detector.finalize()
    
    write_midi(notes, output_path)
    
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
