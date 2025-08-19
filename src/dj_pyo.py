import os
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from pyo import *
import queue
import threading

player = None
hands_detected = False
pitch_shifter = None  # Added for pitch control
audio_buffer = queue.Queue(maxsize=100)  # Buffer to store audio samples for visualization
is_recording = True  # Flag to control the recording thread
restart_visualizer = False  # Flag to indicate visualizer should reconnect
recorder = None  # Global variable for audio recorder

# Function to continuously record audio data for visualization
def record_audio_data():
    global is_recording, restart_visualizer
    restart_visualizer = False
    recorder = None
    
    while is_recording:
        try:
            # If we need to restart the visualizer (e.g., after track change)
            if restart_visualizer or recorder is None:
                # Create a new recorder object connected to the current pitch_shifter
                if pitch_shifter is not None:
                    recorder = PeakAmp(pitch_shifter, function=lambda x, y: audio_buffer_update(max(x, y)))
                    restart_visualizer = False
                    print("Audio visualizer connected")
                else:
                    time.sleep(0.1)  # Wait for pitch_shifter to be created
                    continue
            
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
        except Exception as e:
            print(f"Audio visualization error: {e}")
            recorder = None
            time.sleep(0.1)  # Wait before retrying
            
def audio_buffer_update(amp):
    # Add amplitude to buffer, remove old values if full
    try:
        if audio_buffer.full():
            audio_buffer.get_nowait()  # Remove oldest value
        audio_buffer.put_nowait(amp)  # Add new value
    except:
        pass  # Ignore if buffer operations fail

# === Setup Pyo Server ===
s = Server().boot()
s.start()

# === Playlist ===
song_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../songs'))
song_files = [os.path.join(song_dir, f) for f in os.listdir(song_dir) if f.endswith('.wav')]
current_track_index = 0

# === Load Track Function (plays immediately) ===
def load_track(index):
    global player, pitch_shifter, recorder
    if player is not None:
        player.stop()
    
    # Create player with initial settings - starts with volume at 0.5 now
    player = SfPlayer(song_files[index], speed=1, loop=True, mul=0.5)
    
    # Create pitch shifter using a better approach for pitch-only changes
    # Use the Harmonizer with appropriate settings to modify pitch without affecting speed
    pitch_shifter = Harmonizer(player, transpo=0, feedback=0, winsize=0.1, mul=1).out()
    
    # Reset audio buffer for visualization
    while not audio_buffer.empty():
        try:
            audio_buffer.get_nowait()
        except:
            break
    
    # Restart audio recording for the new track
    try:
        # Audio visualizer may need to be reconnected
        global restart_visualizer
        restart_visualizer = True
    except:
        pass

load_track(current_track_index)

# Start the audio recording thread
recording_thread = threading.Thread(target=record_audio_data)
recording_thread.daemon = True  # Thread will exit when main program exits
recording_thread.start()

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.6,
                       max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === Control Parameters ===
volume = 0.5       # Default volume
speed = 1.0        # Default speed
pitch_hz = 440     # Default pitch (Hz)
pitch_transpo = 0  # Default transposition value in semitones

# === Utility Functions ===
def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_pinch_distance(hand):
    thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return distance(thumb, index)

# Function to detect hand twist by comparing wrist-middle_finger with wrist-pinky orientations
def detect_hand_twist(hand):
    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    pinky = hand.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate vectors
    wrist_to_middle = np.array([middle_finger.x - wrist.x, middle_finger.y - wrist.y])
    wrist_to_pinky = np.array([pinky.x - wrist.x, pinky.y - wrist.y])
    
    # Normalize vectors
    wrist_to_middle = wrist_to_middle / np.linalg.norm(wrist_to_middle)
    wrist_to_pinky = wrist_to_pinky / np.linalg.norm(wrist_to_pinky)
    
    # Calculate cross product to determine twist direction (positive or negative)
    cross_product = np.cross(np.append(wrist_to_middle, 0), np.append(wrist_to_pinky, 0))[2]
    
    # Calculate dot product to get similarity (angle)
    dot_product = np.dot(wrist_to_middle, wrist_to_pinky)
    
    # If dot product is too high, vectors are too similar (not twisted)
    if dot_product > 0.85:  # Threshold for minimum twist
        return 0
    
    # Return twist direction (-1 or 1) based on cross product
    return np.sign(cross_product)

# Function to convert Hz to semitones (relative to A4 = 440Hz)
def hz_to_transpo(hz):
    # Convert Hz to semitones relative to A4 (440Hz)
    return 12 * math.log2(hz / 440.0)

# === Webcam ===
cap = cv2.VideoCapture(0)

# Set resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual dimensions (camera might not support exactly 1280x720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution set to: {frame_width}x{frame_height}")
print("Hand DJ (Pyo) Started - [Q]uit, [N]ext, [P]revious, [R]eset")

# Variables to track twist gestures and prevent multiple triggers
left_twist_cooldown = 0
right_twist_cooldown = 0
twist_cooldown_frames = 30  # Wait this many frames before detecting another twist

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = None
    right_hand = None
    left_twist = 0
    right_twist = 0
    
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True
        for info, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = info.classification[0].label
            if label == "Left":
                left_hand = landmarks
                # Only check for twist if cooldown has expired
                if left_twist_cooldown <= 0:
                    left_twist = detect_hand_twist(landmarks)
            elif label == "Right":
                right_hand = landmarks
                # Only check for twist if cooldown has expired
                if right_twist_cooldown <= 0:
                    right_twist = detect_hand_twist(landmarks)
    
    # Handle twist gestures for track control
    if left_twist < -0.3:  # Left hand twist threshold
        current_track_index = (current_track_index - 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
        left_twist_cooldown = twist_cooldown_frames  # Set cooldown
        print(f"Left hand twist detected: Previous track ({current_track_index})")
        
    if right_twist > 0.3:  # Right hand twist threshold
        current_track_index = (current_track_index + 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
        right_twist_cooldown = twist_cooldown_frames  # Set cooldown
        print(f"Right hand twist detected: Next track ({current_track_index})")
        
    # Decrease cooldown counters
    if left_twist_cooldown > 0:
        left_twist_cooldown -= 1
    if right_twist_cooldown > 0:
        right_twist_cooldown -= 1

    # Volume - distance between pinch centers (scaled to 0-10)
    # Only when both hands are present
    if left_hand and right_hand:
        # Get pinch points
        left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate center of each pinch
        left_pinch_center = np.array([(left_thumb.x + left_index.x) / 2, (left_thumb.y + left_index.y) / 2])
        right_pinch_center = np.array([(right_thumb.x + right_index.x) / 2, (right_thumb.y + right_index.y) / 2])
        
        # Calculate distance between pinch centers
        pinch_dist = np.linalg.norm(right_pinch_center - left_pinch_center) * frame_width
        
        # More sensitive volume mapping - reaches max volume with less distance
        # Adjust these values to fine-tune sensitivity
        min_dist = 50  # Distance (in pixels) for minimum volume
        max_dist = 300  # Distance (in pixels) for maximum volume
        
        # Linear mapping from distance to volume range
        volume_display = float(min(10.0, max(0.0, 
            10.0 * (pinch_dist - min_dist) / (max_dist - min_dist)
        )))
        
        volume = volume_display / 10.0  # Convert to 0-1 range for player
        
        if pitch_shifter:
            pitch_shifter.setMul(volume)  # Set volume on the pitch shifter

    # Speed - left pinch
    if left_hand:
        pinch = get_pinch_distance(left_hand)
        speed = float(max(0.1, min(2.0, 2.0 - pinch * 4)))
        player.setSpeed(speed)

    # Pitch - right pinch (implemented using Harmonizer)
    if right_hand and pitch_shifter:
        pinch = get_pinch_distance(right_hand)
        
        # Calculate pitch in Hz (20Hz to 600Hz)
        pitch_hz = max(20, min(600, 600 - pinch * 1200))
        
        # Convert Hz to semitones for transposition
        # For Harmonizer, we need semitones relative to original pitch
        pitch_transpo = hz_to_transpo(pitch_hz) - hz_to_transpo(440)
        
        # Apply transposition to the Harmonizer
        pitch_shifter.setTranspo(pitch_transpo)

    # Draw visual indicators based on available hands
    if left_hand or right_hand:
        # Single hand mode - only show for the hand that's visible
        
        if left_hand:
            # Get left hand pinch points
            left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert to pixel coordinates
            left_thumb_px = (int(left_thumb.x * frame_width), int(left_thumb.y * frame_height))
            left_index_px = (int(left_index.x * frame_width), int(left_index.y * frame_height))
            
            # Calculate pinch center
            left_pinch_center = (
                (left_thumb_px[0] + left_index_px[0]) // 2,
                (left_thumb_px[1] + left_index_px[1]) // 2
            )
            
            # Draw circles at pinch points
            circle_radius = 12
            circle_color = (255, 255, 255)  # White circles
            circle_thickness = 2
            
            cv2.circle(frame, left_thumb_px, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, left_index_px, circle_radius, circle_color, circle_thickness)
            
            # Draw filled circle at pinch center
            cv2.circle(frame, left_pinch_center, 8, (255, 255, 255), -1)  # White filled dot
            
            # Draw vertical connection line for pinch
            cv2.line(frame, left_thumb_px, left_index_px, (255, 255, 255), 2)  # White line
            
            # Display SPEED label and value below hand
            cv2.putText(frame, "SPEED", 
                      (left_pinch_center[0] - 30, left_pinch_center[1] + 35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{speed:.1f}x", 
                      (left_pinch_center[0] - 20, left_pinch_center[1] + 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show twist detection if active
            if left_twist_cooldown > 0:
                cv2.putText(frame, "PREV TRACK", 
                          (left_pinch_center[0] - 50, left_pinch_center[1] - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        if right_hand:
            # Get right hand pinch points
            right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert to pixel coordinates
            right_thumb_px = (int(right_thumb.x * frame_width), int(right_thumb.y * frame_height))
            right_index_px = (int(right_index.x * frame_width), int(right_index.y * frame_height))
            
            # Calculate pinch center
            right_pinch_center = (
                (right_thumb_px[0] + right_index_px[0]) // 2,
                (right_thumb_px[1] + right_index_px[1]) // 2
            )
            
            # Draw circles at pinch points
            circle_radius = 12
            circle_color = (255, 255, 255)  # White circles
            circle_thickness = 2
            
            cv2.circle(frame, right_thumb_px, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, right_index_px, circle_radius, circle_color, circle_thickness)
            
            # Draw filled circle at pinch center
            cv2.circle(frame, right_pinch_center, 8, (255, 255, 255), -1)  # White filled dot
            
            # Draw vertical connection line for pinch
            cv2.line(frame, right_thumb_px, right_index_px, (255, 255, 255), 2)  # White line
            
            # Display PITCH label and value below hand
            cv2.putText(frame, "PITCH", 
                      (right_pinch_center[0] - 30, right_pinch_center[1] + 35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{pitch_hz:.0f}Hz", 
                      (right_pinch_center[0] - 35, right_pinch_center[1] + 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show twist detection if active
            if right_twist_cooldown > 0:
                cv2.putText(frame, "NEXT TRACK", 
                          (right_pinch_center[0] - 50, right_pinch_center[1] - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # If both hands are present, draw the connection and volume visualization
        if left_hand and right_hand:
            # Get pinch points (normalized coordinates)
            left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to pixel values
            left_thumb_px = (int(left_thumb.x * frame_width), int(left_thumb.y * frame_height))
            left_index_px = (int(left_index.x * frame_width), int(left_index.y * frame_height))
            right_thumb_px = (int(right_thumb.x * frame_width), int(right_thumb.y * frame_height))
            right_index_px = (int(right_index.x * frame_width), int(right_index.y * frame_height))
            
            # Calculate center of pinch lines in pixels
            left_pinch_center = (
                (left_thumb_px[0] + left_index_px[0]) // 2,
                (left_thumb_px[1] + left_index_px[1]) // 2
            )
            right_pinch_center = (
                (right_thumb_px[0] + right_index_px[0]) // 2,
                (right_thumb_px[1] + right_index_px[1]) // 2
            )
            
            # Draw the white connecting line between pinch centers
            cv2.line(frame, left_pinch_center, right_pinch_center, (255, 255, 255), 2)
            
            # Draw circles at pinch points (thumb and index of each hand)
            circle_radius = 12
            circle_color = (255, 255, 255)  # White circles
            circle_thickness = 2
            
            cv2.circle(frame, left_thumb_px, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, left_index_px, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, right_thumb_px, circle_radius, circle_color, circle_thickness)
            cv2.circle(frame, right_index_px, circle_radius, circle_color, circle_thickness)
            
            # Draw filled circles at pinch centers
            cv2.circle(frame, left_pinch_center, 8, (255, 255, 255), -1)  # White filled dot
            cv2.circle(frame, right_pinch_center, 8, (255, 255, 255), -1)  # White filled dot
            
            # Draw vertical connection lines for each pinch
            cv2.line(frame, left_thumb_px, left_index_px, (255, 255, 255), 2)  # White line for left pinch
            cv2.line(frame, right_thumb_px, right_index_px, (255, 255, 255), 2)  # White line for right pinch
            
            # Calculate line properties for volume visualization
            line_length = int(np.sqrt(
                (right_pinch_center[0] - left_pinch_center[0])**2 + 
                (right_pinch_center[1] - left_pinch_center[1])**2
            ))
            angle = np.arctan2(
                right_pinch_center[1] - left_pinch_center[1],
                right_pinch_center[0] - left_pinch_center[0]
            )
            
            # Calculate midpoint for label placement
            midpoint = (
                (left_pinch_center[0] + right_pinch_center[0]) // 2,
                (left_pinch_center[1] + right_pinch_center[1]) // 2
            )
            
            # Draw volume bars spanning the entire connection line
            bar_width = 6
            bar_spacing = 6  # Increased spacing for better visibility
            num_bars = max(10, min(30, line_length // (bar_width + bar_spacing)))
            
            # Get or generate audio data for visualization
            try:
                samples = list(audio_buffer.queue)
                if len(samples) < num_bars:
                    samples = [0] * (num_bars - len(samples)) + samples
                recent_samples = samples[-num_bars:]
                
                # If the samples are very low, boost them for better visibility
                max_sample = max(recent_samples) if recent_samples else 0
                if max_sample < 0.1:  # If samples are very small
                    # Use volume-based visualization as fallback
                    recent_samples = []
                    for i in range(num_bars):
                        pos = abs((i - (num_bars // 2)) / (num_bars // 2))
                        amp = volume * (1 - pos * 0.6)
                        recent_samples.append(amp)
            except:
                # Fallback visualization based on volume
                recent_samples = []
                for i in range(num_bars):
                    pos = abs((i - (num_bars // 2)) / (num_bars // 2))
                    amp = volume * (1 - pos * 0.6)
                    recent_samples.append(amp)
            
            # Draw each volume bar along the line
            for i in range(num_bars):
                # Calculate position along the line
                pos = i / (num_bars - 1)  # 0 to 1
                x = int(left_pinch_center[0] + pos * (right_pinch_center[0] - left_pinch_center[0]))
                y = int(left_pinch_center[1] + pos * (right_pinch_center[1] - left_pinch_center[1]))
                
                # Calculate bar height 
                amp = recent_samples[i]
                max_bar_height = 60  # Maximum bar height
                bar_height = int(max_bar_height * min(1.0, amp * 2.0))
                
                # Calculate color gradient (blue to green)
                b = max(0, 255 - (i * 255 // num_bars))
                g = min(255, i * 255 // num_bars)
                
                # Calculate perpendicular direction for the bar
                perp_x = int(np.sin(angle) * bar_height)
                perp_y = int(-np.cos(angle) * bar_height)
                
                # Make sure we actually draw something even with zero amplitude
                min_bar_height = 5
                if bar_height < min_bar_height:
                    perp_x = int(np.sin(angle) * min_bar_height)
                    perp_y = int(-np.cos(angle) * min_bar_height)
                
                # Draw the volume bar perpendicular to the line
                cv2.line(frame, 
                        (x, y), 
                        (x + perp_x, y + perp_y), 
                        (b, g, 255), bar_width)
            
            # Draw SPEED label and value BELOW left hand
            cv2.putText(frame, "SPEED", 
                       (left_pinch_center[0] - 30, left_pinch_center[1] + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{speed:.1f}x", 
                       (left_pinch_center[0] - 20, left_pinch_center[1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw PITCH label and value BELOW right hand
            cv2.putText(frame, "PITCH", 
                       (right_pinch_center[0] - 30, right_pinch_center[1] + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{pitch_hz:.0f}Hz", 
                       (right_pinch_center[0] - 35, right_pinch_center[1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw VOLUME label and value at center, above the line
            text_offset = 35  # Offset for text above the line
            perp_text_x = int(np.sin(angle) * text_offset)
            perp_text_y = int(-np.cos(angle) * text_offset)
            
            vol_text_pos = (midpoint[0] + perp_text_x - 30, midpoint[1] + perp_text_y - 10)
            vol_value_pos = (midpoint[0] + perp_text_x - 12, midpoint[1] + perp_text_y + 15)
            
            cv2.putText(frame, "VOLUME", vol_text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{volume_display:.1f}", vol_value_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display twist instruction if detected
            if left_twist_cooldown > 0:
                cv2.putText(frame, "PREV TRACK", 
                           (left_pinch_center[0] - 50, left_pinch_center[1] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                           
            if right_twist_cooldown > 0:
                cv2.putText(frame, "NEXT TRACK", 
                           (right_pinch_center[0] - 50, right_pinch_center[1] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display only track info at the bottom (minimal display)
    track_info = f"{os.path.basename(song_files[current_track_index])}"
    cv2.putText(frame, track_info, 
               (10, frame_height - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    # Create display window with proper size
    cv2.namedWindow("DJ Controller (Pyo)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DJ Controller (Pyo)", 1280, 720)
    cv2.imshow("DJ Controller (Pyo)", frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_track_index = (current_track_index + 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
    elif key == ord('p'):
        current_track_index = (current_track_index - 1) % len(song_files)
        load_track(current_track_index)
        hands_detected = False
    elif key == ord('r'):
        volume = 0.5
        volume_display = 5.0
        speed = 1.0
        pitch_hz = 440
        pitch_transpo = 0
        if pitch_shifter:
            pitch_shifter.setMul(float(volume))
        player.setSpeed(float(speed))
        if pitch_shifter:
            pitch_shifter.setTranspo(pitch_transpo)

cap.release()
cv2.destroyAllWindows()
is_recording = False  # Stop the recording thread
time.sleep(0.5)  # Give time for the thread to exit
s.stop()
s.shutdown()
print("Exited cleanly.")