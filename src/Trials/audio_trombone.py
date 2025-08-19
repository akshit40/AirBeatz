import pygame
import numpy as np
import os
import sys
import traceback
from scipy.io import wavfile
from scipy import signal

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

# Sound directory paths - try multiple possible locations
# Get the absolute path of the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # VirtualInstrument directory

# Try different possible paths to find the sounds directory
POSSIBLE_SOUND_DIRS = [
    os.path.join(BASE_DIR, "sounds"),  # VirtualInstrument/sounds
    os.path.join(os.getcwd(), "sounds"),  # Current working directory/sounds
    "sounds",  # Relative to where the script is run from
]

# Try to find Trombone_Base.wav
TROMBONE_BASE = None
for sound_dir in POSSIBLE_SOUND_DIRS:
    if os.path.exists(sound_dir):
        print(f"Found sounds directory at: {sound_dir}")
        possible_file = os.path.join(sound_dir, "Trombone_Base.wav")
        if os.path.exists(possible_file):
            TROMBONE_BASE = possible_file
            print(f"Found Trombone_Base.wav at: {TROMBONE_BASE}")
            break

# If file still not found, use the first valid sounds directory
if TROMBONE_BASE is None:
    for sound_dir in POSSIBLE_SOUND_DIRS:
        if os.path.exists(sound_dir):
            TROMBONE_BASE = os.path.join(sound_dir, "Trombone_Base.wav")
            print(f"Will try to load from: {TROMBONE_BASE} (file may not exist)")
            break

# Note frequencies (Hz) for pitch shifting
BASE_FREQ = 233.08  # Bb3 as reference frequency
NOTE_FREQS = {
    "Bb3": 233.08,
    "C4": 261.63,
    "D4": 293.66,
    "Eb4": 311.13,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
}

# Sound caching
sound_cache = {}
current_sound = None

def load_base_sound():
    """
    Load the base trombone sound for pitch shifting.
    Returns the sample rate and sound data.
    """
    try:
        print(f"Attempting to load: {TROMBONE_BASE}")
        if TROMBONE_BASE and os.path.exists(TROMBONE_BASE):
            sample_rate, data = wavfile.read(TROMBONE_BASE)
            print(f"Successfully loaded sound file: {TROMBONE_BASE}")
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1).astype(data.dtype)
            
            # Convert to int16 if needed
            if data.dtype != np.int16:
                data = (data * (32767 / np.max(np.abs(data)))).astype(np.int16)
                
            return sample_rate, data
        else:
            print(f"File not found: {TROMBONE_BASE}")
            raise FileNotFoundError(f"Trombone_Base.wav not found at {TROMBONE_BASE}")
            
    except Exception as e:
        print(f"Error loading trombone sound: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Generate a fallback tone
        print("Generating fallback sine wave tone")
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a more trombone-like synthetic sound
        fundamental = np.sin(2 * np.pi * BASE_FREQ * t)
        h1 = 0.5 * np.sin(2 * np.pi * (2 * BASE_FREQ) * t)
        h2 = 0.7 * np.sin(2 * np.pi * (3 * BASE_FREQ) * t)
        data = fundamental + h1 + h2
        
        # Normalize and convert to int16
        data = data / np.max(np.abs(data))
        data = (data * 32767).astype(np.int16)
        
        return sample_rate, data

def pitch_shift(data, sample_rate, semitones):
    """
    Shift the pitch of audio data by the specified number of semitones.
    """
    # Calculate pitch shift factor
    factor = 2 ** (semitones / 12.0)
    
    # For small pitch shifts, just use resample for efficiency
    output_len = int(len(data) / factor)
    return signal.resample(data, output_len).astype(np.int16)

def get_semitones_from_note(note_name):
    """
    Calculate semitones difference between base note (Bb3) and target note.
    """
    if note_name not in NOTE_FREQS:
        return 0
        
    freq_ratio = NOTE_FREQS[note_name] / BASE_FREQ
    semitones = 12 * np.log2(freq_ratio)
    return semitones

def create_pitch_shifted_sound(note_name):
    """
    Create a pitch-shifted version of the base trombone sound.
    """
    try:
        # Load the base sound (only once)
        if 'base_rate' not in sound_cache or 'base_data' not in sound_cache:
            sample_rate, data = load_base_sound()
            sound_cache['base_rate'] = sample_rate
            sound_cache['base_data'] = data.copy()  # Store a copy to avoid modifying original
        else:
            sample_rate = sound_cache['base_rate']
            data = sound_cache['base_data'].copy()
        
        # Calculate semitones difference and shift pitch
        semitones = get_semitones_from_note(note_name)
        shifted_data = pitch_shift(data, sample_rate, semitones)
        
        # Create pygame Sound object
        return pygame.mixer.Sound(buffer=shifted_data.tobytes())
    
    except Exception as e:
        print(f"Error creating pitch-shifted sound: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a simple beep for emergencies
        return create_emergency_sound(note_name)

def create_emergency_sound(note_name):
    """Create an emergency beep sound if everything else fails"""
    sample_rate = 44100
    duration = 0.5
    freq = NOTE_FREQS.get(note_name, 440)  # Default to A4 if not found
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t) * 32767
    data = data.astype(np.int16)
    
    return pygame.mixer.Sound(buffer=data.tobytes())

def play_note(note_name):
    """
    Plays the pitch-shifted trombone note.
    """
    global current_sound
    
    try:
        # Stop any currently playing note
        stop_note()
        
        if note_name not in NOTE_FREQS:
            print(f"Note '{note_name}' not recognized.")
            return
        
        # Check if the sound is cached, if not generate and cache it
        if note_name not in sound_cache:
            sound_cache[note_name] = create_pitch_shifted_sound(note_name)
        
        # Play the sound
        current_sound = sound_cache[note_name]
        current_sound.play(loops=-1)  # Loop indefinitely for continuous sound
    
    except Exception as e:
        print(f"Error playing note: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def stop_note():
    """
    Stops any currently playing note.
    """
    global current_sound
    
    if current_sound:
        current_sound.stop()
        current_sound = None

def preload_notes():
    """
    Pre-generates all trombone notes for smoother playback.
    """
    print("Preloading trombone notes...")
    try:
        for note in NOTE_FREQS:
            if note not in sound_cache:
                sound_cache[note] = create_pitch_shifted_sound(note)
        print("Preloading complete.")
    except Exception as e:
        print(f"Error during preloading: {e}")
        print(f"Traceback: {traceback.format_exc()}")

# Display sound file information
print("\n===== VIRTUAL TROMBONE AUDIO SETUP =====")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Base directory: {BASE_DIR}")
print(f"Sound file path: {TROMBONE_BASE}")
print(f"File exists: {TROMBONE_BASE and os.path.exists(TROMBONE_BASE)}")
print("========================================\n")

# Preload notes when module is imported
preload_notes()