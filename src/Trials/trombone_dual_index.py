import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading
import queue

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
MAX_VOLUME = 0.5
MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.5
MIN_FREQ = 233.08
MAX_FREQ = 466.16

current_frequency = MIN_FREQ
is_playing = False
audio_queue = queue.Queue()
audio_thread_running = True

pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=BUFFER_SIZE)
pygame.mixer.set_num_channels(1)
channel = pygame.mixer.Channel(0)

def generate_trombone_tone(frequency, duration, volume=MAX_VOLUME):
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    waveform = (
        np.sin(2 * np.pi * frequency * t) +
        0.5 * np.sin(2 * np.pi * 2 * frequency * t) +
        0.7 * np.sin(2 * np.pi * 3 * frequency * t) +
        0.2 * np.sin(2 * np.pi * 4 * frequency * t) +
        0.4 * np.sin(2 * np.pi * 5 * frequency * t) +
        0.1 * np.sin(2 * np.pi * 6 * frequency * t)
    )
    envelope = np.ones(num_samples)
    attack = int(0.02 * SAMPLE_RATE)
    release = int(0.02 * SAMPLE_RATE)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    waveform = waveform * envelope * volume
    waveform = waveform / max(1.0, np.max(np.abs(waveform)))
    waveform = (waveform * 32767).astype(np.int16)
    return pygame.mixer.Sound(waveform.tobytes())

def audio_thread_function():
    global audio_thread_running, is_playing
    current_sound = None
    current_freq = 0
    last_update_time = 0
    while audio_thread_running:
        try:
            cmd = audio_queue.get(block=False)
            if cmd["type"] == "play":
                is_playing = True
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
        if is_playing and current_freq != current_frequency:
            current_freq = current_frequency
            new_sound = generate_trombone_tone(current_freq, 0.5)
            channel.fadeout(20)
            channel.play(new_sound, loops=-1, fade_ms=20)
        time.sleep(0.01)
    if channel.get_busy():
        channel.stop()

def start_audio():
    audio_queue.put({"type": "play", "frequency": current_frequency})

def stop_audio():
    audio_queue.put({"type": "stop"})

def update_frequency(freq):
    global current_frequency
    current_frequency = freq
    audio_queue.put({"type": "update", "frequency": freq})

def both_index_fingers_extended(hands):
    count = 0
    for hand in hands:
        tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        pip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        if tip < pip:
            count += 1
    return count == 2

def index_finger_distance(h1, h2):
    p1 = np.array([
        h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        h1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    ])
    p2 = np.array([
        h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        h2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    ])
    return np.linalg.norm(p1 - p2)

def map_distance_to_frequency(distance):
    distance = max(MIN_DISTANCE, min(distance, MAX_DISTANCE))
    normalized = 1 - ((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))
    return MIN_FREQ * math.pow(MAX_FREQ / MIN_FREQ, normalized)

audio_thread = threading.Thread(target=audio_thread_function)
audio_thread.daemon = True
audio_thread.start()

print("Starting Dual Index Trombone...")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1, hand2 = results.multi_hand_landmarks
        if both_index_fingers_extended([hand1, hand2]):
            dist = index_finger_distance(hand1, hand2)
            freq = map_distance_to_frequency(dist)
            update_frequency(freq)
            if not is_playing:
                start_audio()
            cv2.putText(image, f"Playing: {freq:.1f} Hz", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if is_playing:
                stop_audio()
    else:
        if is_playing:
            stop_audio()

    for hand_landmarks in results.multi_hand_landmarks or []:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow("Dual Index Trombone", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

if is_playing:
    stop_audio()

audio_thread_running = False
if audio_thread.is_alive():
    audio_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
