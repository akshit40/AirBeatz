import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=2)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Trombone synth params
SAMPLE_RATE = 44100
MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.5
MIN_FREQ = 233.08  # Bb3
MAX_FREQ = 466.16  # Bb4

# Shared state
current_freq = MIN_FREQ
is_playing = False
stream_running = True

# Thread-safe frequency lock
freq_lock = threading.Lock()

# Generate waveform with harmonics
def trombone_wave(t, freq):
    return (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * 2 * freq * t) +
        0.7 * np.sin(2 * np.pi * 3 * freq * t) +
        0.2 * np.sin(2 * np.pi * 4 * freq * t) +
        0.4 * np.sin(2 * np.pi * 5 * freq * t)
    )

# Real-time audio callback
def audio_callback(outdata, frames, time_info, status):
    global current_freq
    t = (np.arange(frames) + audio_callback.phase) / SAMPLE_RATE
    with freq_lock:
        freq = current_freq
    wave = trombone_wave(t, freq)
    wave *= 0.3  # volume
    outdata[:] = wave.reshape(-1, 1).astype(np.float32)
    audio_callback.phase += frames

audio_callback.phase = 0

# Start audio stream
stream = sd.OutputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
stream.start()

# Gesture logic
def both_index_extended(hands):
    return sum(
        hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        for hand in hands
    ) == 2

def index_tip_distance(h1, h2):
    p1 = np.array([
        h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    ])
    p2 = np.array([
        h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    ])
    return np.linalg.norm(p1 - p2)

def map_distance_to_freq(dist):
    dist = max(MIN_DISTANCE, min(MAX_DISTANCE, dist))
    norm = 1 - ((dist - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))
    return MIN_FREQ * math.pow(MAX_FREQ / MIN_FREQ, norm)

print("Real-time Trombone Synth started.")
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            h1, h2 = results.multi_hand_landmarks
            if both_index_extended([h1, h2]):
                dist = index_tip_distance(h1, h2)
                freq = map_distance_to_freq(dist)
                with freq_lock:
                    current_freq = freq
                is_playing = True
                cv2.putText(frame, f"Playing: {freq:.1f} Hz", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                is_playing = False
                with freq_lock:
                    current_freq = 0
        else:
            is_playing = False
            with freq_lock:
                current_freq = 0

        for hand in results.multi_hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Real-time Trombone Synth", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    stream_running = False
    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")
