"""
Simple test script to verify that the trombone audio is working correctly.
This script attempts to:
1. Find and load the Trombone_Base.wav file
2. Play it in its original form 
3. Play a few pitch-shifted versions

Run with: python src/test_audio.py
"""

import os
import sys
import time
import pygame
import numpy as np
from scipy.io import wavfile

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

def find_trombone_file():
    """Try to find the Trombone_Base.wav file in various locations"""
    print("Searching for Trombone_Base.wav...")
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Possible locations
    possible_paths = [
        os.path.join(base_dir, "sounds", "Trombone_Base.wav"),
        os.path.join(script_dir, "../sounds", "Trombone_Base.wav"),
        os.path.join(os.getcwd(), "sounds", "Trombone_Base.wav"),
        "sounds/Trombone_Base.wav",
        "../sounds/Trombone_Base.wav",
    ]
    
    # Print out the current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found at: {path}")
            return path
        else:
            print(f"Not found at: {path}")
    
    # If not found, create a test directory with a synthetic sound
    print("Trombone_Base.wav not found. Creating a test sound...")
    return None

def create_test_sound():
    """Create a simple synthetic trombone-like sound for testing"""
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    frequency = 233.08  # Bb3
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a simple harmonic series
    wave = np.sin(2 * np.pi * frequency * t)
    wave += 0.5 * np.sin(2 * np.pi * 2 * frequency * t)
    wave += 0.3 * np.sin(2 * np.pi * 3 * frequency * t)
    
    # Apply envelope
    envelope = np.ones_like(t)
    attack = int(0.1 * sample_rate)
    release = int(0.3 * sample_rate)
    
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    wave = wave * envelope
    
    # Normalize and convert to int16
    wave = wave / np.max(np.abs(wave))
    wave = (wave * 32767).astype(np.int16)
    
    return sample_rate, wave

def play_sound(sound_data, sample_rate, pitch_shift=0):
    """Play a sound with optional pitch shifting"""
    # If pitch shift is requested, do it
    if pitch_shift != 0:
        # Calculate pitch shift factor
        factor = 2 ** (pitch_shift / 12.0)
        
        # Resample for pitch shift
        output_len = int(len(sound_data) / factor)
        from scipy import signal
        shifted_data = signal.resample(sound_data, output_len).astype(np.int16)
    else:
        shifted_data = sound_data
    
    # Create and play the sound
    sound = pygame.mixer.Sound(buffer=shifted_data.tobytes())
    sound.play()
    
    # Wait for it to finish
    time.sleep(2)
    sound.stop()

def main():
    """Main function to test trombone audio"""
    print("===== TROMBONE AUDIO TEST =====")
    
    # Find the trombone sound file
    sound_path = find_trombone_file()
    
    if sound_path and os.path.exists(sound_path):
        # Load the found sound file
        print(f"Loading sound file: {sound_path}")
        try:
            sample_rate, sound_data = wavfile.read(sound_path)
            print(f"Sample rate: {sample_rate}Hz, Data shape: {sound_data.shape}")
            
            # Convert to mono if stereo
            if len(sound_data.shape) > 1:
                sound_data = np.mean(sound_data, axis=1).astype(sound_data.dtype)
                print("Converted stereo to mono")
                
            # Normalize if needed
            if sound_data.dtype != np.int16:
                sound_data = (sound_data * (32767 / np.max(np.abs(sound_data)))).astype(np.int16)
                print("Normalized and converted to int16")
                
        except Exception as e:
            print(f"Error loading sound file: {e}")
            print("Creating synthetic sound instead")
            sample_rate, sound_data = create_test_sound()
    else:
        # Create a synthetic sound
        sample_rate, sound_data = create_test_sound()
    
    # Play original sound
    print("\nPlaying original sound...")
    play_sound(sound_data, sample_rate)
    
    # Play pitch-shifted versions
    for semitones, note in [(0, "Bb3"), (2, "C4"), (5, "Eb4"), (9, "G4")]:
        print(f"\nPlaying {note} (shifted by {semitones} semitones)...")
        play_sound(sound_data, sample_rate, pitch_shift=semitones)
    
    print("\nAudio test complete. If you heard sounds, audio is working correctly!")

if __name__ == "__main__":
    main()