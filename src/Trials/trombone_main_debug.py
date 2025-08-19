import cv2
import mediapipe as mp
import numpy as np
from audio_trombone import play_note, stop_note
import sys
import time

# Print debugging information
print("Starting Virtual Trombone with debugging...")
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"MediaPipe version: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Webcam setup
print("Setting up webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Webcam resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Trombone parameters
MIN_DISTANCE = 0.1  # Minimum hand distance (normalized)
MAX_DISTANCE = 0.8  # Maximum hand distance (normalized)
is_playing = False
current_note = None

# Test sound at startup
print("Testing audio by playing a note...")
play_note("C4")
time.sleep(1)
stop_note()
print("Audio test complete")

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

def is_fist(hand_landmarks):
    """Check if the hand is making a fist gesture"""
    # For a fist, all fingertips should be close to the base of the hand
    # Use a simpler, more reliable detection method
    
    # Get palm positions
    palm_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    palm_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    
    # Check each fingertip relative to its base
    fingers_bent = 0
    total_fingers = 4  # Excluding thumb as it's unreliable
    
    for fingertip, mcp in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]:
        # If fingertip is below (greater y) its mcp, it's bent
        if hand_landmarks.landmark[fingertip].y > hand_landmarks.landmark[mcp].y:
            fingers_bent += 1
    
    # Consider it a fist if most fingers are bent
    return fingers_bent >= 3

def is_index_extended(hand_landmarks):
    """Check if index finger is extended - used as trigger"""
    # Make detection more reliable
    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    
    # Index is extended if it's significantly higher than its PIP joint
    # AND higher than the middle finger (to avoid false positives)
    index_extended = (index_tip_y < index_pip_y - 0.03) and (index_tip_y < middle_tip_y)
    
    return index_extended

def map_distance_to_note(distance):
    """Map the distance between hands to a musical note"""
    # Clamp distance to min/max range
    clamped_distance = max(MIN_DISTANCE, min(distance, MAX_DISTANCE))
    
    # Normalize to 0-1 range
    normalized = (clamped_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    
    # Map to notes (inverted: closer hands = higher pitch)
    normalized = 1 - normalized
    
    # 7 notes from Bb3 to A4 (typical trombone first position notes)
    notes = ["Bb3", "C4", "D4", "Eb4", "F4", "G4", "A4"]
    index = int(normalized * (len(notes) - 1))
    
    return notes[index]

def draw_trombone_visualization(image, distance_percentage, is_playing):
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
    
    # If playing, draw sound waves
    if is_playing:
        wave_center = (width - 50, 125)
        for i in range(3):
            radius = 20 + i * 15
            cv2.circle(image, wave_center, radius, (0, 255, 0), 2)

# Detection status and debug variables
frame_count = 0
last_status_print = time.time()
hands_detected = 0
fists_detected = 0
index_extended_detected = 0
play_attempts = 0

print("Starting main loop...")

while cap.isOpened():
    frame_count += 1
    
    # Read frame
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera")
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process with MediaPipe
    results = hands.process(image)
    
    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Add debug overlay
    current_time = time.time()
    if current_time - last_status_print > 1.0:  # Print debug stats every second
        print(f"FPS: ~{frame_count / (current_time - last_status_print):.1f} | "
              f"Hands: {hands_detected/frame_count:.1%} | "
              f"Fists: {fists_detected/max(1,hands_detected):.1%} | "
              f"Index: {index_extended_detected/max(1,hands_detected):.1%} | "
              f"Play attempts: {play_attempts}")
        frame_count = 0
        hands_detected = 0
        fists_detected = 0
        index_extended_detected = 0
        play_attempts = 0
        last_status_print = current_time

    # Reset playing state if no hands detected
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        if is_playing:
            print("Stopping sound - not enough hands detected")
            stop_note()
            is_playing = False
        
        if results.multi_hand_landmarks:
            hands_detected += 1
            # Still draw the single hand
            mp_drawing.draw_landmarks(
                image, 
                results.multi_hand_landmarks[0], 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Display instructions
        cv2.putText(image, "SHOW BOTH HANDS AND MAKE FISTS", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "EXTEND INDEX FINGER TO PLAY", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, 0.5, False)
    else:
        # Both hands detected
        hands_detected += 1
        
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
        
        # Map distance to note
        note = map_distance_to_note(distance)
        
        # Check if both hands are making fists
        fists_count = sum(1 for hand in results.multi_hand_landmarks if is_fist(hand))
        if fists_count > 0:
            fists_detected += 1
        
        both_fists = (fists_count == 2)
        
        # Check if index finger is extended on at least one hand (trigger)
        any_index_extended = any(is_index_extended(hand) for hand in results.multi_hand_landmarks)
        if any_index_extended:
            index_extended_detected += 1
        
        # Calculate normalized distance for visualization
        distance_percentage = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        distance_percentage = max(0, min(distance_percentage, 1))
        
        # Display info
        cv2.putText(image, f"Distance: {distance:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Note: {note}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Fists: {fists_count}/2", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Index extended: {any_index_extended}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Visual slider showing position
        slider_length = 400
        slider_pos = int((1 - ((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))) 
                        * slider_length)
        slider_pos = max(0, min(slider_pos, slider_length))
        
        cv2.rectangle(image, (10, 220), (10 + slider_length, 240), (100, 100, 100), -1)
        cv2.rectangle(image, (10, 220), (10 + slider_pos, 240), (0, 255, 0), -1)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, distance_percentage, is_playing)
        
        # Add debugging text to show if we're detecting the play gesture
        play_gesture_detected = both_fists and any_index_extended
        cv2.putText(image, f"Play gesture: {play_gesture_detected}", (10, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # SIMPLIFIED PLAYING CONDITION FOR DEBUGGING
        # Just using any index finger extension to trigger
        if any_index_extended:
            play_attempts += 1
            cv2.putText(image, "PLAYING", (10, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            if not is_playing or current_note != note:
                print(f"Playing note: {note} (was_playing: {is_playing}, current_note: {current_note})")
                if is_playing:
                    stop_note()
                play_note(note)
                current_note = note
                is_playing = True
        else:
            if is_playing:
                print("Stopping note - gesture released")
                stop_note()
                is_playing = False
                current_note = None

    # Add a simple debug overlay
    cv2.putText(image, "DEBUG MODE", (image.shape[1] - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display image
    cv2.imshow("Virtual Trombone (Debug)", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

# Clean up
print("Cleaning up...")
if is_playing:
    stop_note()
    
cap.release()
cv2.destroyAllWindows()
print("Program terminated")