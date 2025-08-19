import cv2
import mediapipe as mp
import numpy as np
from audio_trombone import play_note, stop_note
import time  # For debugging

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=2)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced from 1920 for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Reduced from 1080 for better performance

# Trombone parameters
MIN_DISTANCE = 0.1  # Minimum hand distance (normalized)
MAX_DISTANCE = 0.8  # Maximum hand distance (normalized)
is_playing = False
current_note = None

# Test sound at startup to verify audio works
print("Testing audio system...")
play_note("C4")
time.sleep(0.5)
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

def is_index_extended(hand_landmarks):
    """
    Check if index finger is extended - now more lenient
    Returns True if index finger is likely extended
    """
    # Get y-coordinates of index finger joints
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    
    # Also get x-coordinates to detect sideways extension
    index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    
    # Two conditions for extension:
    # 1. Finger tip is higher than PIP joint (vertical extension)
    # 2. Finger tip is horizontally distant from PIP joint (horizontal extension)
    vertical_extension = index_tip < index_pip
    horizontal_extension = abs(index_tip_x - index_pip_x) > 0.05
    
    return vertical_extension or horizontal_extension

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
            stop_note()
            is_playing = False
        
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
        cv2.putText(image, "Show both hands", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Extend index finger to play", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, 0.5, False)
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
        
        # Map distance to note
        note = map_distance_to_note(distance)
        
        # Check if index finger is extended on at least one hand
        any_index_extended = any(is_index_extended(hand) for hand in results.multi_hand_landmarks)
        
        # Calculate normalized distance for visualization
        distance_percentage = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        distance_percentage = max(0, min(distance_percentage, 1))
        
        # Display info
        cv2.putText(image, f"Distance: {distance:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Note: {note}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Visual slider showing position
        slider_length = 400
        slider_pos = int((1 - ((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))) 
                        * slider_length)
        slider_pos = max(0, min(slider_pos, slider_length))
        
        cv2.rectangle(image, (10, 180), (10 + slider_length, 200), (100, 100, 100), -1)
        cv2.rectangle(image, (10, 180), (10 + slider_pos, 200), (0, 255, 0), -1)
        
        # Draw trombone visualization
        draw_trombone_visualization(image, distance_percentage, is_playing)
        
        # Play or change note if index is extended - SIMPLIFIED CONDITION
        if any_index_extended:
            cv2.putText(image, "PLAYING", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            if not is_playing or current_note != note:
                if is_playing:
                    stop_note()
                play_note(note)
                current_note = note
                is_playing = True
        else:
            if is_playing:
                stop_note()
                is_playing = False
                current_note = None

    cv2.imshow("Virtual Trombone", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

# Clean up
if is_playing:
    stop_note()
    
cap.release()
cv2.destroyAllWindows()