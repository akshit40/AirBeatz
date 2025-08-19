import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading
import queue
import sys

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=2)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Audio parameters
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
MAX_VOLUME = 0.5  # Adjust volume (0.0 to 1.0)

# Trombone parameters
MIN_DISTANCE = 0.1   # Minimum hand distance (normalized)
MAX_DISTANCE = 0.8   # Maximum hand distance (normalized)
MIN_FREQ = 233.08    # Bb3
MAX_FREQ = 466.16    # Bb4 (one octave higher)

# Shared variables for audio thread
current_frequency = MIN_FREQ
is_playing = False
audio_queue = queue.Queue()
audio_thread_running = True

# Setup PyGame for audio
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=BUFFER_SIZE)
pygame.mixer.set_num_channels(1)
channel = pygame.mixer.Channel(0)

def generate_trombone_tone(frequency, duration, volume=MAX_VOLUME):
    """Generate a synthetic trombone-like tone at the specified frequency"""
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Create a waveform with harmonics to approximate a trombone timbre
    # Trombones have strong odd harmonics
    fundamental = np.sin(2 * np.pi * frequency * t)
    h1 = 0.5 * np.sin(2 * np.pi * 2 * frequency * t)  # 2nd harmonic
    h2 = 0.7 * np.sin(2 * np.pi * 3 * frequency * t)  # 3rd harmonic (strong)
    h3 = 0.2 * np.sin(2 * np.pi * 4 * frequency * t)  # 4th harmonic
    h4 = 0.4 * np.sin(2 * np.pi * 5 * frequency * t)  # 5th harmonic (strong)
    h5 = 0.1 * np.sin(2 * np.pi * 6 * frequency * t)  # 6th harmonic
    
    # Combine the harmonics to create the trombone-like sound
    waveform = fundamental + h1 + h2 + h3 + h4 + h5
    
    # Apply an envelope to smooth attack and release
    attack_time = 0.02  # seconds
    release_time = 0.02  # seconds
    
    attack_samples = int(attack_time * SAMPLE_RATE)
    release_samples = int(release_time * SAMPLE_RATE)
    
    envelope = np.ones(num_samples)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0 and num_samples > release_samples:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    # Apply envelope and volume
    waveform = waveform * envelope * volume
    
    # Normalize to prevent clipping
    waveform = waveform / max(1.0, np.max(np.abs(waveform)))
    
    # Convert to 16-bit signed integers
    waveform = (waveform * 32767).astype(np.int16)
    
    return pygame.mixer.Sound(waveform.tobytes())

def audio_thread_function():
    """Thread function that continuously updates the playing tone"""
    global audio_thread_running, is_playing
    
    current_sound = None
    current_freq = 0
    last_update_time = 0
    
    print("Audio thread started")
    
    while audio_thread_running:
        # Check for new commands in the queue
        try:
            cmd = audio_queue.get(block=False)
            if cmd["type"] == "play":
                is_playing = True
                # Initial sound generation on play command
                if current_sound is None or abs(current_freq - cmd["frequency"]) > 0.5:
                    current_freq = cmd["frequency"]
                    current_sound = generate_trombone_tone(current_freq, 0.5)
                    channel.play(current_sound, loops=-1)
                last_update_time = time.time()
            elif cmd["type"] == "stop":
                is_playing = False
                channel.stop()
                current_sound = None
            elif cmd["type"] == "update" and is_playing:
                current_freq = cmd["frequency"]
                last_update_time = time.time()
        except queue.Empty:
            pass
        
        # If playing, check if we need to update the frequency
        if is_playing and current_sound is not None:
            current_time = time.time()
            # Update sound every 50ms if frequency has changed
            if current_time - last_update_time > 0.05 and abs(current_freq - current_frequency) > 0.5:
                current_freq = current_frequency
                new_sound = generate_trombone_tone(current_freq, 0.5)
                # Use fadeout/fadein for smoother transition
                channel.fadeout(20)
                channel.play(new_sound, loops=-1, fade_ms=20)
                current_sound = new_sound
                last_update_time = current_time
                
        # Sleep to reduce CPU usage
        time.sleep(0.01)
    
    # Clean up
    if channel.get_busy():
        channel.stop()
    print("Audio thread stopped")

def start_audio():
    """Start playing the tone"""
    audio_queue.put({"type": "play", "frequency": current_frequency})

def stop_audio():
    """Stop playing the tone"""
    audio_queue.put({"type": "stop"})

def update_frequency(freq):
    """Update the playing frequency"""
    global current_frequency
    current_frequency = freq
    audio_queue.put({"type": "update", "frequency": freq})

def is_fist(hand_landmarks):
    """Check if the hand is making a fist gesture"""
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
    
    # Consider it a fist if most fingers are bent
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
            # Make waves pulsate
            scale = 0.8 + 0.4 * math.sin(time.time() * 10 + i)
            scaled_radius = int(radius * scale)
            cv2.circle(image, wave_center, scaled_radius, (0, 255, 0), 2)

# Start the audio thread
audio_thread = threading.Thread(target=audio_thread_function)
audio_thread.daemon = True
audio_thread.start()

# Test sound at startup
print("Testing audio system...")
test_freq = 440
test_sound = generate_trombone_tone(test_freq, 0.5)
channel.play(test_sound)
time.sleep(0.5)
channel.stop()
print("Audio test complete")

print("Starting Virtual Trombone with continuous synthetic sound...")
print("Show both fists and extend index finger to play")
print("Press 'q' or ESC to quit")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add title
        cv2.putText(image, "SYNTHETIC TROMBONE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Reset playing state if no hands detected
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
            if is_playing:
                stop_audio()
            
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
            
            # Update frequency for audio thread
            update_frequency(frequency)
            
            # Check if both hands are making fists
            fists_count = sum(1 for hand in results.multi_hand_landmarks if is_fist(hand))
            both_fists = (fists_count >= 1)  # Make it more lenient
            
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
            should_play = both_fists and any_index_extended
            
            if should_play:
                cv2.putText(image, "PLAYING", (10, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                if not is_playing:
                    start_audio()
            else:
                if is_playing:
                    stop_audio()

        cv2.imshow("Synthetic Trombone", image)
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break

finally:
    # Clean up
    print("Cleaning up...")
    if is_playing:
        stop_audio()
    
    # Signal audio thread to stop
    audio_thread_running = False
    if audio_thread.is_alive():
        audio_thread.join(timeout=1.0)
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Program terminated")