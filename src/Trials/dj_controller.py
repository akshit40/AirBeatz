import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
import math
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

SONG_FOLDER = "songs"
songs = [os.path.join(SONG_FOLDER, f) for f in os.listdir(SONG_FOLDER) if f.lower().endswith('.wav')]
current_song_index = 0
volume = 0.5
speed = 1.0
pitch_shift = 0.0

playback_thread = None
playback_running = False
waveform_data = None

# Helper functions
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_pinch_distance(hand_landmarks):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return distance(thumb, index)

def load_waveform(path):
    global waveform_data
    try:
        rate, data = wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data / np.max(np.abs(data))  # normalize
        waveform_data = data[::rate // 100]  # 100 samples per second
    except Exception as e:
        print("Failed to load waveform:", e)
        waveform_data = None

def draw_waveform(image, data):
    if data is None:
        return
    h, w = image.shape[:2]
    center_y = int(h * 0.9)
    scale = int(h * 0.1)
    x_spacing = int(w / len(data))
    prev = (0, center_y)
    for i in range(1, len(data)):
        x = i * x_spacing
        y = int(center_y - data[i] * scale)
        cv2.line(image, prev, (x, y), (255, 255, 0), 1)
        prev = (x, y)

def draw_slider(image, value, min_val, max_val, center, length, label):
    norm = (value - min_val) / (max_val - min_val)
    start = (int(center[0] - length / 2), center[1])
    end = (int(center[0] + length / 2), center[1])
    marker = (int(start[0] + norm * length), center[1])
    cv2.line(image, start, end, (255, 255, 255), 2)
    cv2.circle(image, marker, 8, (0, 0, 255), -1)
    cv2.putText(image, f"{label}: {value:.2f}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def audio_player(song_path):
    global volume, speed, pitch_shift, playback_running
    print(f"Loading: {os.path.basename(song_path)}")
    song = AudioSegment.from_wav(song_path)
    playback_running = True
    while playback_running:
        modified = song._spawn(song.raw_data, overrides={
            "frame_rate": int(song.frame_rate * (2.0 ** (pitch_shift / 12.0)))
        }).set_frame_rate(song.frame_rate)
        modified = modified.set_frame_rate(int(modified.frame_rate * speed))
        modified = modified + (20 * math.log10(max(volume, 0.001)))
        play_obj = play_audio(modified)
        while play_obj.is_playing() and playback_running:
            time.sleep(0.1)
        break

def play_song(path):
    global playback_thread
    load_waveform(path)
    if playback_thread and playback_thread.is_alive():
        stop_playback()
    playback_thread = threading.Thread(target=audio_player, args=(path,))
    playback_thread.start()

def stop_playback():
    global playback_running
    playback_running = False
    time.sleep(0.2)

if songs:
    play_song(songs[current_song_index])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        frame = np.full((480, 640, 3), (0, 255, 255), dtype=np.uint8)
        cv2.putText(frame, "Camera error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("DJ Controller", frame)
        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
            break
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_data = list(zip(results.multi_handedness, results.multi_hand_landmarks))
        left_hand = None
        right_hand = None
        for hand_info, landmarks in hand_data:
            label = hand_info.classification[0].label
            if label == "Left":
                left_hand = landmarks
            elif label == "Right":
                right_hand = landmarks

        if left_hand and right_hand:
            lw = left_hand.landmark[mp_hands.HandLandmark.WRIST]
            rw = right_hand.landmark[mp_hands.HandLandmark.WRIST]
            dist = distance(lw, rw)
            volume = min(1.0, max(0.0, 2 * dist))

        if left_hand:
            speed = 1.0 + (0.5 - get_pinch_distance(left_hand))
            speed = max(0.5, min(1.5, speed))

        if right_hand:
            pitch_shift = (0.5 - get_pinch_distance(right_hand)) * 20

        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    h, w = frame.shape[:2]
    draw_waveform(frame, waveform_data)
    draw_slider(frame, volume, 0.0, 1.0, (w // 2, int(h * 0.15)), 200, "VOLUME")
    draw_slider(frame, speed, 0.5, 1.5, (int(w * 0.25), int(h * 0.5)), 120, "SPEED")
    draw_slider(frame, pitch_shift, -10, 10, (int(w * 0.75), int(h * 0.5)), 120, "PITCH")
    cv2.putText(frame, f"Now Playing: {os.path.basename(songs[current_song_index])}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q or ESC to exit", (40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    cv2.imshow("DJ Controller", frame)
    key = cv2.waitKey(5) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('n'):
        current_song_index = (current_song_index + 1) % len(songs)
        play_song(songs[current_song_index])

stop_playback()
cap.release()
cv2.destroyAllWindows()
