import pygame
import os

pygame.mixer.init()
SOUND_DIR = os.path.join(os.path.dirname(__file__), "../sounds")

# Preload chord sounds
chords = {
  "C_major": pygame.mixer.Sound(os.path.join(SOUND_DIR, "C_major.wav")),
  "G_major": pygame.mixer.Sound(os.path.join(SOUND_DIR, "G_major.wav")),
  "D_major": pygame.mixer.Sound(os.path.join(SOUND_DIR, "D_major.wav")),
}

def play_chord(name):
  """
  Plays a chord sound by name.
  """
  if name in chords:
    chords[name].play()
  else:
    print(f"Chord '{name}' not found.")
