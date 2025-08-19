"""
Simplified Virtual Trombone
---------------------------
This version has more lenient gesture detection and simplified controls
to help diagnose issues with the original version.

- Only requires one hand (instead of two)
- Plays as long as index finger is extended (no fist requirement)
- Hand position determines pitch (y-position)
- More verbose debugging
"""

import cv2
import mediapipe as mp
import numpy as np
from audio_trombone import play_note, stop_note
import sys
import time

# Print debugging information
print("Starting Simplified Virtual Trombone...")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Webcam setup
print("Setting up webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    sys.exit(1)

# Test sound at startup
print("Testing audio...")
play_note("C4")
time.sleep(1)
stop_note()
print("Audio test complete")

# Trombone parameters
is_playing = False
current_note = None

def is_index_extended(hand_landmarks):
    """Check if index finger is extended"""
    # Get y-coordinates of index finger joints
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    
    # If tip is higher (smaller y) than pip, finger is extended
    return index_tip < index_pip

def get_note_from_position(hand_landmarks):
    """Determine note based on hand position"""
    # Using y-position of hand to determine note (higher hand = higher pitch)
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    
    # Map y-position to notes (screen space is 0-1, with 0 at top)
    if wrist_y < 0.3:  # Top of screen
        return "A4"
    elif wrist_y < 0.4:
        return "G4"
    elif wrist_y < 0.5:
        return "F4"
    elif wrist_y < 0.6:
        return "Eb4"
    elif wrist_y < 0.7:
        return "D4"
    elif wrist_y < 0.8:
        return "C4"
    else:  # Bottom of screen
        return "Bb3"

while cap.isOpened():
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
    
    # Add title
    cv2.putText(image, "SIMPLIFIED TROMBONE", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Reset playing state if no hands detected
    if not results.multi_hand_landmarks:
        if is_playing:
            print("Stopping sound - no hands detected")
            stop_note()
            is_playing = False
            current_note = None
        
        # Display instructions
        cv2.putText(image, "Show one hand and extend index finger", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Move hand up/down to change pitch", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Just use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Check if index finger is extended
        index_extended = is_index_extended(hand_landmarks)
        
        # Get note based on hand position
        note = get_note_from_position(hand_landmarks)
        
        # Display info
        cv2.putText(image, f"Note: {note}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Index extended: {index_extended}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw notes on the side of the screen
        notes = ["A4", "G4", "F4", "Eb4", "D4", "C4", "Bb3"]
        for i, note_name in enumerate(notes):
            y_pos = int(30 + i * (image.shape[0] / len(notes)))
            cv2.putText(image, note_name, (image.shape[1] - 100, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Play note if index is extended
        if index_extended:
            cv2.putText(image, "PLAYING", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            if not is_playing or current_note != note:
                print(f"Playing note: {note}")
                if is_playing:
                    stop_note()
                play_note(note)
                current_note = note
                is_playing = True
        else:
            if is_playing:
                print("Stopping note - index finger not extended")
                stop_note()
                is_playing = False
                current_note = None

    # Display image
    cv2.imshow("Simplified Virtual Trombone", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

# Clean up
print("Cleaning up...")
if is_playing:
    stop_note()
    
cap.release()
cv2.destroyAllWindows()
print("Program terminated")