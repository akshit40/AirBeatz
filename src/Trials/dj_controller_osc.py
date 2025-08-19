import cv2
import mediapipe as mp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
import math

# === OSC Setup ===
osc_ip = "127.0.0.1"
osc_port = 57120
osc = SimpleUDPClient(osc_ip, osc_port)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.6,
                       max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === Control Variables ===
volume = 5.0       # 0.0 – 10.0
rate = 1.0         # 0.1x – 2.0x
pitch = 220.0      # 20Hz – 600Hz

# === Utility Functions ===
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_pinch_distance(hand_landmarks):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return distance(thumb, index)

# === Video Capture ===
cap = cv2.VideoCapture(0)
print("Hand DJ started!")
print("Controls: [Q]uit | [R]eset | [N]ext | [P]revious")

# Start playback initially
osc.send_message("/hand_dj", ["play"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_info, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = hand_info.classification[0].label
            if label == "Left":
                left_hand = landmarks
            elif label == "Right":
                right_hand = landmarks

    # === Gesture Mapping ===

    # Volume control: Distance between wrists
    if left_hand and right_hand:
        lw = left_hand.landmark[mp_hands.HandLandmark.WRIST]
        rw = right_hand.landmark[mp_hands.HandLandmark.WRIST]
        dist = distance(lw, rw)
        volume = min(10.0, max(0.0, dist * 25))
        osc.send_message("/hand_dj", ["vol", volume])

    # Speed control: Left hand pinch
    if left_hand:
        pinch = get_pinch_distance(left_hand)
        rate = max(0.1, min(2.0, 2.0 - pinch * 4))  # Inverse mapping
        osc.send_message("/hand_dj", ["rate", rate])

    # Pitch control: Right hand pinch
    if right_hand:
        pinch = get_pinch_distance(right_hand)
        pitch = max(20, min(600, 600 - pinch * 1200))  # Inverse mapping
        osc.send_message("/hand_dj", ["freq", pitch])

    # === Drawing & HUD ===
    for hand in results.multi_hand_landmarks or []:
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Volume: {volume:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed: {rate:.2f}x", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f}Hz", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "[Q]uit  [R]eset  [N]ext  [P]revious", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    cv2.imshow("Hand DJ OSC", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        osc.send_message("/hand_dj", ["reset"])
    elif key == ord('n'):
        osc.send_message("/hand_dj", ["next"])
        osc.send_message("/hand_dj", ["play"])
    elif key == ord('p'):
        osc.send_message("/hand_dj", ["prev"])
        osc.send_message("/hand_dj", ["play"])

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")
