from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

def detect_strum(last_y, current_y, cooldown):
    """
    Detects a downward strumming gesture based on Y-position change.
    """
    if last_y is not None and cooldown <= 0:
        if last_y - current_y > 0.1:
            return True
    return False

def get_extended_fingers(hand_landmarks: NormalizedLandmarkList):
    """
    Determines which fingers are extended based on tip vs base joint Y-positions.
    Returns a list of finger tip landmark IDs that are extended.
    """
    fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    base_joints = [6, 10, 14, 18]

    extended = []
    for tip, base in zip(fingertips, base_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            extended.append(tip)
    return extended

def detect_chord(hand_landmarks):
    """
    Detects a chord based on simple finger extension patterns.
    Returns the name of the detected chord or None.
    """
    finger_ids = set(get_extended_fingers(hand_landmarks))

    if finger_ids == {8, 12}:  # Index & Middle
        return "C_major"
    elif finger_ids == {8, 12, 16}:  # Index, Middle & Ring
        return "G_major"
    elif finger_ids == {8, 12, 16, 20}:  # Index, Middle, Ring & Pinky
        return "D_major"
    else:
        return None
