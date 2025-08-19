"""
Setup script to create the sounds directory and generate a test trombone sound file
if it doesn't already exist.

Run with: python src/setup_sounds.py
"""

import os
import sys
import numpy as np
from scipy.io import wavfile

def create_sounds_directory():
    """Create the sounds directory if it doesn't exist"""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Define the sounds directory path
    sounds_dir = os.path.join(base_dir, "sounds")
    
    # Create if it doesn't exist
    if not os.path.exists(sounds_dir):
        print(f"Creating sounds directory: {sounds_dir}")
        os.makedirs(sounds_dir)
    else:
        print(f"Sounds directory already exists: {sounds_dir}")
    
    return sounds_dir

def create_trombone_sound(output_path):
    """Create a synthetic trombone sound and save it to the given path"""
    print(f"Creating synthetic trombone sound at: {output_path}")
    
    # Define parameters
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    base_freq = 233.08  # Bb3 (standard trombone note)
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a trombone-like sound with harmonics
    # Real trombones have strong odd harmonics
    fundamental = np.sin(2 * np.pi * base_freq * t)
    h1 = 0.5 * np.sin(2 * np.pi * (2 * base_freq) * t)  # 2nd harmonic
    h2 = 0.7 * np.sin(2 * np.pi * (3 * base_freq) * t)  # 3rd harmonic (strong)
    h3 = 0.2 * np.sin(2 * np.pi * (4 * base_freq) * t)  # 4th harmonic
    h4 = 0.4 * np.sin(2 * np.pi * (5 * base_freq) * t)  # 5th harmonic (strong)
    
    # Combine harmonics
    sound = fundamental + h1 + h2 + h3 + h4
    
    # Apply envelope for natural sound
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)  # 100ms attack
    decay_samples = int(0.2 * sample_rate)   # 200ms decay
    release_samples = int(0.3 * sample_rate) # 300ms release
    
    # Attack phase - linear ramp
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay and sustain
    if decay_samples > 0:
        decay_end = attack_samples + decay_samples
        decay_curve = np.linspace(0, 1, decay_samples)
        decay_curve = 1 - (1 - 0.7) * decay_curve**2  # Curve to 0.7 sustain level
        envelope[attack_samples:decay_end] = decay_curve
    
    # Release phase
    if release_samples > 0:
        release_start = len(envelope) - release_samples
        envelope[release_start:] = np.linspace(0.7, 0, release_samples)
    
    # Apply the envelope
    sound = sound * envelope
    
    # Normalize and convert to 16-bit PCM
    sound = sound / np.max(np.abs(sound))
    sound = (sound * 32767).astype(np.int16)
    
    # Save to WAV file
    try:
        wavfile.write(output_path, sample_rate, sound)
        print(f"Successfully created: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating sound file: {e}")
        return False

def main():
    print("===== SETTING UP SOUNDS DIRECTORY =====")
    
    # Create sounds directory
    sounds_dir = create_sounds_directory()
    
    # Define path for trombone sound
    trombone_path = os.path.join(sounds_dir, "Trombone_Base.wav")
    
    # Check if it already exists
    if os.path.exists(trombone_path):
        print(f"Trombone_Base.wav already exists at: {trombone_path}")
        response = input("Do you want to replace it? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing file.")
            return
    
    # Create the sound
    success = create_trombone_sound(trombone_path)
    
    if success:
        print("\nSetup complete! You now have a trombone sound file at:")
        print(trombone_path)
        print("\nYou can now run:")
        print("  python src/test_audio.py    (to test audio)")
        print("  python src/trombone_main.py (to run the full application)")
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()