import cv2
import mediapipe as mp
from gestures import detect_strum, detect_chord
from audio import play_chord

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.6,
    max_num_hands=2  # Allow detection of up to 2 hands
)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

last_y = None
cooldown = 0
chord_name = None  # Keep track of the current chord

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image for more intuitive interaction
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    left_hand = None
    right_hand = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        # First, identify which detected hand is left and which is right
        for idx, hand_handedness in enumerate(results.multi_handedness):
            if idx >= len(results.multi_hand_landmarks):  # Safety check
                continue
                
            # The handedness prediction gives us a label (Left or Right) and a score
            handedness = hand_handedness.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[idx]
            
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Store the hand data based on handedness
            if handedness == "Left":  # This will be the chord hand (appears as right due to mirroring)
                left_hand = hand_landmarks
            elif handedness == "Right":  # This will be the strum hand (appears as left due to mirroring)
                right_hand = hand_landmarks
        
        # Process chord detection with the left hand
        if left_hand:
            new_chord = detect_chord(left_hand)
            if new_chord:
                chord_name = new_chord  # Update the chord only if a valid chord is detected
        
        # Process strum detection with the right hand
        if right_hand and chord_name and cooldown <= 0:
            wrist_y = right_hand.landmark[mp_hands.HandLandmark.WRIST].y
            
            # Only register downward strum (positive y-change since y increases downward in image coordinates)
            if last_y is not None and (wrist_y - last_y) > 0.05:  # Detect downward movement only
                print(f"Downward strum detected! Playing {chord_name}")
                play_chord(chord_name)
                cooldown = 20
                
            last_y = wrist_y
    
    # Display the current chord on screen
    if chord_name:
        cv2.putText(image, f"Chord: {chord_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Draw hand role labels
    if left_hand:
        wrist_pos = left_hand.landmark[mp_hands.HandLandmark.WRIST]
        cv2.putText(image, "Chord Hand", (int(wrist_pos.x * frame.shape[1]), 
                    int(wrist_pos.y * frame.shape[0] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if right_hand:
        wrist_pos = right_hand.landmark[mp_hands.HandLandmark.WRIST]
        cv2.putText(image, "Strum Hand", (int(wrist_pos.x * frame.shape[1]), 
                    int(wrist_pos.y * frame.shape[0] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if cooldown > 0:
        cooldown -= 1

    cv2.imshow("Gesture Guitar", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()