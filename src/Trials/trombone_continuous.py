import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import os
from scipy.io import wavfile
from scipy import signal

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=2)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Audio setup
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
pygame.mixer.set_num_channels(8)  # Increase number of channels for smoother transitions

# Find trombone sound file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
SOUND_DIR = os.path.join(BASE_DIR, "sounds")
TROMBONE_BASE = os.path.join(SOUND_DIR, "Trombone_Base.wav")

if not os.path.exists(TROMBONE_BASE):
    print(f"WARNING: Trombone sound file not found at {TROMBONE_BASE}")
    print("Running setup to create a synthetic trombone sound...")
    
    # Create sounds directory if it doesn't exist
    os.makedirs(SOUND_DIR, exist_ok=True)
    
    # Create a synthetic trombone sound
    sample_rate = 44100
    duration = 2.0
    base_freq = 233.08  # Bb3 (standard trombone note)
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a trombone-like sound with harmonics
    fundamental = np.sin(2 * np.pi * base_freq * t)
    h1 = 0.5 * np.sin(2 * np.pi * (2 * base_freq) * t)
    h2 = 0.7 * np.sin(2 * np.pi * (3 * base_freq) * t)
    h3 = 0.2 * np.sin(2 * np.pi * (4 * base_freq) * t)
    h4 = 0.4 * np.sin(2 * np.pi * (5 * base_freq) * t)
    
    sound = fundamental + h1 + h2 + h3 + h4
    
    # Apply envelope
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)
    decay_samples = int(0.2 * sample_rate)
    release_samples = int(0.3 * sample_rate)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if release_samples > 0:
        release_start = len(envelope) - release_samples
        envelope[release_start:] = np.linspace(1, 0, release_samples)
    
    sound = sound * envelope
    sound = sound / np.max(np.abs(sound))
    sound = (sound * 32767).astype(np.int16)
    
    wavfile.write(TROMBONE_BASE, sample_rate, sound)
    print(f"Created synthetic trombone sound at {TROMBONE_BASE}")

# Load the base trombone sound
try:
    sample_rate, trombone_data = wavfile.read(TROMBONE_BASE)
    # Convert to mono if stereo
    if len(trombone_data.shape) > 1:
        trombone_data = np.mean(trombone_data, axis=1).astype(np.int16)
    print(f"Loaded trombone sound: {TROMBONE_BASE}")
except Exception as e:
    print(f"Error loading sound: {e}")
    # Create a simple fallback sound
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    trombone_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

# Trombone parameters
MIN_DISTANCE = 0.1   # Minimum hand distance (normalized)
MAX_DISTANCE = 0.8   # Maximum hand distance (normalized)
MIN_FREQ = 233.08    # Bb3
MAX_FREQ = 466.16    # Bb4 (one octave higher)
is_playing = False
current_sound = None
current_freq = None

# Cache for pitch-shifted sounds
sound_cache = {}

def pitch_shift_realtime(data, sample_rate, freq_ratio):
    """Shift the pitch of audio data by the specified frequency ratio"""
    # Calculate output length
    output_len = int(len(data) / freq_ratio)
    
    # Use resampling for pitch shifting
    return signal.resample(data, output_len).astype(np.int16)

def get_sound_for_frequency(freq):
    """Get a pitch-shifted version of the trombone sound for the specified frequency"""
    # Round to nearest 0.5 Hz to reduce cache size but keep smooth transitions
    freq = round(freq * 2) / 2
    
    if freq in sound_cache:
        return sound_cache[freq]
    
    # Calculate frequency ratio
    freq_ratio = freq / MIN_FREQ
    
    # Pitch shift the base sound
    shifted_data = pitch_shift_realtime(trombone_data, sample_rate, freq_ratio)
    
    # Create pygame Sound object
    sound = pygame.mixer.Sound(buffer=shifted_data.tobytes())
    
    # Cache for future use
    sound_cache[freq] = sound
    
    return sound

def play_frequency(freq):
    """Play the specified frequency"""
    global current_sound, current_freq, is_playing
    
    # If already playing the same frequency, do nothing
    if is_playing and current_freq == freq:
        return
    
    # Stop current sound if playing
    if is_playing:
        current_sound.stop()
    
    # Get sound for this frequency
    current_sound = get_sound_for_frequency(freq)
    current_freq = freq
    
    # Play sound
    current_sound.play(loops=-1)  # Loop indefinitely
    is_playing = True

def stop_sound():
    """Stop any currently playing sound"""
    global current_sound, is_playing
    
    if is_playing and current_sound:
        current_sound.stop()
        is_playing = False

def is_fist(hand_landmarks):
    """Check if the hand is making a fist gesture"""
    # For a fist, all fingertips should be below their PIP joints
    # (higher y value means lower position in image)
    
    # Check each finger (index, middle, ring, pinky)
    finger_bent_count = 0
    
    for fingertip, pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        # If fingertip is below (greater y) its PIP joint, it's bent
        if hand_landmarks.landmark[fingertip].y > hand_landmarks.landmark[pip].y:
            finger_bent_count += 1
    
    # More lenient fist detection - consider it a fist if most fingers are bent
    return finger_bent_count >= 3

def is_index_extended(hand_landmarks):
    """Check if index finger is extended - used as trigger"""
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    
    # Index is extended if tip is higher than PIP joint
    return index_tip_y < index_pip_y

def calculate_hand_distance(landmarks1, landmarks2):
    """Calculate distance between two hand wrists"""
    wrist1 = np.array([
        landmarks1.landmark[mp_hands.HandLandmark.WRIST].x,
        landmarks1.landmark[mp_hands.HandLandmark.WRIST].y
    ])
    
    wrist2 = np.array([
        landmarks2.landmark[mp_hands.HandLandmark.WRIST].x,
        landmarks2.landmark[mp_hands.HandLandmark.WRIST].y
    ])
    
    return np.linalg.norm(wrist1 - wrist2)

def map_distance_to_frequency(distance):
    """Map the distance between hands to a continuous frequency"""
    # Clamp distance to min/max range
    clamped_distance = max(MIN_DISTANCE, min(distance, MAX_DISTANCE))
    
    # Normalize to 0-1 range
    normalized = (clamped_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    
    # Map to frequency range (inverted: closer hands = higher pitch)
    normalized = 1 - normalized
    
    # Logarithmic mapping for more natural pitch perception
    # A_log = A₁ * (A₂/A₁)^t where t is normalized [0-1]
    freq = MIN_FREQ * math.pow(MAX_FREQ/MIN_FREQ, normalized)
    
    return freq

def frequency_to_note_name(freq):
    """Convert a frequency to the closest note name"""
    # A4 is 440 Hz, and each semitone is a factor of 2^(1/12)
    A4 = 440.0
    
    # Calculate number of semitones from A4
    semitones = 12 * math.log2(freq / A4)
    
    # Round to nearest semitone
    semitones_rounded = round(semitones)
    
    # Calculate cents (deviation from equal temperament)
    cents = 100 * (semitones - semitones_rounded)
    
    # Note names (starting from A)
    note_names = ["A", "A#/Bb", "B", "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab"]
    
    # Calculate octave and note index
    octave = 4 + (semitones_rounded + 9) // 12
    note_idx = (semitones_rounded + 9) % 12
    
    note = f"{note_names[note_idx]}{octave}"
    
    # Add cents if deviation is significant
    if abs(cents) > 10:
        note += f" {'+' if cents > 0 else ''}{cents:.0f}¢"
    
    return note

def draw_trombone_visualization(image, distance_percentage, frequency, is_playing):
    """Draw a visual representation of a trombone with slide position"""
    height, width = image.shape[:2]
    
    # Draw trombone body (fixed part)
    body_start = (width - 300, 100)
    body_end = (width - 100, 150)
    cv2.rectangle(image, body_start, body_end, (100, 100, 200), -1)
    
    # Draw mouthpiece
    mouthpiece_center = (width - 300, 125)
    cv2.circle(image, mouthpiece_center, 20, (150, 150, 220), -1)
    
    # Calculate slide position
    slide_length = int(distance_percentage * 300)
    slide_start = (width - 100, 125)
    slide_end = (width - 100 - slide_length, 125)
    
    # Draw slide
    cv2.line(image, slide_start, slide_end, (200, 100, 100), 10)
    cv2.circle(image, slide_end, 15, (220, 150, 150), -1)
    
    # Draw bell
    bell_center = (width - 100, 125)
    cv2.ellipse(image, bell_center, (30, 50), 0, -90, 90, (150, 150, 220), -1)
    
    # Draw frequency display
    freq_text = f"{frequency:.1f} Hz"
    cv2.putText(image, freq_text, (width - 280, 190), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # If playing, draw sound waves
    if is_playing:
        wave_center = (width - 50, 125)
        for i in range(3):
            radius = 20 + i * 15
            cv2.circle(image, wave_center, radius, (0, 255, 0), 2)

# Test sound at startup
print("Testing audio system...")
test_sound = get_sound_for_frequency(MIN_FREQ)
test_sound.play()
time.sleep(0.5)
test_sound.stop()
print("Audio test complete")

# Main loop
print("Starting Virtual Trombone with continuous pitch control...")
print("Show both fists and extend index finger to play")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Reset playing state if no hands detected
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        if is_playing:
            stop_sound()
        
        if results.multi_hand_landmarks:
            # Still draw the single hand
            mp_drawing.draw_landmarks(
                image, 
                results.multi_hand_landmarks[0], 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Display instructions if no hands detected
        cv2.putText(image, "Show both fists", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Extend index finger to play", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, 0.5, MIN_FREQ, False)
    else:
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Calculate distance between hands
        distance = calculate_hand_distance(results.multi_hand_landmarks[0], 
                                          results.multi_hand_landmarks[1])
        
        # Map distance to continuous frequency
        frequency = map_distance_to_frequency(distance)
        
        # Get equivalent note name
        note_name = frequency_to_note_name(frequency)
        
        # Check if both hands are making fists
        fists_count = sum(1 for hand in results.multi_hand_landmarks if is_fist(hand))
        both_fists = (fists_count == 2)
        
        # Check if index finger is extended on at least one hand
        any_index_extended = any(is_index_extended(hand) for hand in results.multi_hand_landmarks)
        
        # Calculate normalized distance for visualization
        distance_percentage = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        distance_percentage = max(0, min(distance_percentage, 1))
        
        # Display info
        cv2.putText(image, f"Distance: {distance:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Frequency: {frequency:.1f} Hz", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Note: {note_name}", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Fists: {fists_count}/2", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Visual slider showing position
        slider_length = 400
        slider_pos = int((1 - ((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))) 
                        * slider_length)
        slider_pos = max(0, min(slider_pos, slider_length))
        
        cv2.rectangle(image, (10, 220), (10 + slider_length, 240), (100, 100, 100), -1)
        cv2.rectangle(image, (10, 220), (10 + slider_pos, 240), (0, 255, 0), -1)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, distance_percentage, frequency, is_playing)
        
        # Play or change note if requirements are met
        if both_fists and any_index_extended:
            cv2.putText(image, "PLAYING", (10, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Play continuous frequency
            play_frequency(frequency)
        else:
            if is_playing:
                stop_sound()

    cv2.imshow("Virtual Trombone - Continuous Pitch", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

# Clean up
if is_playing:
    stop_sound()
    
cap.release()
cv2.destroyAllWindows()