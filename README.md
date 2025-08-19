# Virtual DJ Hand Controller

A computer vision-based application that allows you to control audio playback using hand gestures, turning your webcam into a virtual DJ controller.

## Features

- **Intuitive Gesture Controls**: Manipulate volume, speed, and pitch using natural hand movements
- **Real-time Audio Processing**: Instantly hear changes as you move your hands
- **Visual Feedback**: On-screen visualization of audio parameters and waveforms
- **Track Navigation**: Switch between tracks with hand twist gestures

![Virtual DJ Hand Controller Demo](docs/demo_screenshot.png)

## How It Works

The application uses your webcam to track hand movements through MediaPipe's hand landmark detection. Different gestures control various aspects of audio playback:

- **Volume**: Distance between left and right hands
- **Speed/Tempo**: Left hand pinch gesture (thumb to index finger)
- **Pitch**: Right hand pinch gesture (thumb to index finger)
- **Track Navigation**: Twist either hand to move forward/backward in the playlist

## Requirements

- Python 3.11.x (recommended). Python 3.10 also works. (Python 3.13 is NOT yet supported by the pinned MediaPipe version.)
- Webcam
- Audio files in WAV format

## Installation

### 1. Install Python 3.11 (if you don't have it yet)
Download the latest 3.11.x (e.g. 3.11.9) from https://www.python.org/downloads/ . During installation on Windows:
- Check "Install for all users" (optional but helps)
- Ensure the "py launcher" option is selected
- (Optional) You may skip adding to PATH; the launcher `py` is enough

Verify versions:
```
py -0p
```
You should see an entry for 3.11 next to any other installed versions (e.g. 3.13).

### 2. Clone / open the project directory
In your terminal (PowerShell or Git Bash) navigate to the project root.

### 3. Create a Python 3.11 virtual environment
PowerShell:
```
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```
Git Bash (note: use `Scripts`, not `bin`):
```
py -3.11 -m venv .venv311
source .venv311/Scripts/activate
```
macOS / Linux (if applicable):
```
python3.11 -m venv .venv311
source .venv311/bin/activate
```
Confirm:
```
python -V  # should print Python 3.11.x
```

If you previously created a venv with another version, remove it first:
```
Remove-Item -Recurse -Force .\venv  # PowerShell (ignore errors)
rm -rf venv .venv .venv311          # Bash
```

### 4. Upgrade pip and install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If `mediapipe` fails on a newer Python (e.g. 3.13), ensure you're inside the 3.11 environment.

### 5. Create directories for audio assets (if they do not exist)
```
mkdir -p songs sounds
```
On Windows (PowerShell) the equivalent is:
```
New-Item -ItemType Directory -Force -Path songs,sounds | Out-Null
```

### 6. Add your WAV audio files to the `songs` directory
Place any additional sound effect WAVs in `sounds` if you plan to trigger them.

### 7. (Optional) Verify MediaPipe availability
```
pip index versions mediapipe | findstr 0.10.21  # PowerShell
pip index versions mediapipe | grep 0.10.21     # Bash
```
If no versions appear, you are likely not using Python 3.10/3.11.

## Usage

1. Run the main application:
- python src/dj_pyo.py
2. Position yourself in front of the webcam, ensuring your hands are visible.

3. Control the audio with the following gestures:
- **Volume**: Move hands closer/further apart
- **Speed**: Pinch your left hand (thumb to index finger)
- **Pitch**: Pinch your right hand (thumb to index finger)
- **Previous Track**: Twist left hand
- **Next Track**: Twist right hand

4. Press 'Q' to quit, 'N' for next track, 'P' for previous track, and 'R' to reset parameters.

## Controls Overview

| Control | Gesture | Effect |
|---------|---------|--------|
| Volume | Distance between hands | 0.0 - 1.0 gain |
| Speed | Left hand pinch | 0.5x - 1.5x playback speed |
| Pitch | Right hand pinch | Adjusts pitch without affecting tempo |
| Next Track | Right hand twist | Moves to next track in playlist |
| Previous Track | Left hand twist | Returns to previous track in playlist |
| Reset | 'R' key | Returns all parameters to default values |

## Configuration

Audio files should be placed in the `songs` directory. The application supports WAV files by default.

## Troubleshooting

- **No camera detected**: Ensure your webcam is connected and not being used by another application
- **No hands detected**: Adjust lighting and position your hands clearly in the camera's view
- **Audio doesn't play**: Check that your audio files are in the correct format and location
- **mediapipe install fails**: You are probably on Python 3.12+ or 3.13; recreate the venv with `py -3.11 -m venv .venv311` and reinstall requirements
- **Activation script not found in Git Bash**: Use `source .venv311/Scripts/activate` (Windows venv uses `Scripts/` not `bin/`)
- **Multiple Python versions**: Use the launcher: `py -3.11` (3.11) or `py -3.13` (3.13) explicitly

## Development

The project is structured as follows:

- `src/dj_pyo.py`: Main application file
- `src/Trials/`: Alternative implementations and experimental features
- `songs/`: Directory for audio files
- `sounds/`: Directory for sound effects

## Acknowledgments

- MediaPipe for the hand tracking library
- Pyo for the audio processing engine
- OpenCV for camera capture and display