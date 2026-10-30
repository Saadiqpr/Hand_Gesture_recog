# Hand Gesture Dino Runner

A webcam-controlled endless runner. MediaPipe detects a hand, a CNN classifies
its gesture, and Pygame uses the prediction to move the player between lanes,
jump, and duck.

## Features

- Real-time hand detection and gesture recognition from a webcam
- Five gesture controls with confidence display and smoothed predictions
- Keyboard controls as an alternative to gestures
- Separate webcam diagnostic and dataset collection utilities

## Requirements

- Python 3.12 (the pinned dependencies are tested with this version)
- Windows, macOS, or Linux with a working webcam
- A display capable of opening OpenCV and Pygame windows

The trained CNN models and MediaPipe hand-landmarker task file are included in
the repository. Keep these files in the project root; the scripts load them
using relative paths.

## Setup

Open PowerShell in the project directory and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, allow scripts for the
current terminal with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
the environment again.

## Play

Start the game from the project directory:

```powershell
python dino_game.py
```

Allow webcam access when prompted. The game opens in its menu; show **Pointing
Up** or press **Enter** to start. Make gestures clearly and keep your hand
visible in the camera preview.

| Gesture | Action |
| --- | --- |
| Open Palm | Jump |
| Fist | Duck |
| L-Sign | Move left one lane |
| V-Sign | Move right one lane |
| Pointing Up | Start or restart |

Keyboard controls while playing: **Space** to jump, **Down Arrow** to duck,
**Left/Right Arrow** to change lanes, **Enter** to start or restart, and **Q**
or **Escape** to quit.

## Other scripts

- `main.py` runs a full-frame CNN diagnostic using `gesture_cnn.keras` and
  `class_mapping.json`.
- `test_finetuned.py` displays predictions from the hand-cropped
  `best_finetuned_gesture.keras` model.
- `collect_gestures.py` captures up to 100 images for each gesture into
  `webcam_dataset/`. Select a class with **1–5**, press **Space** to start or
  pause capture, and press **Q** to exit.
- `inspect_dataset.py` displays sample images from `webcam_dataset/` and saves
  a preview to `webcam_samples.png`.

The diagnostic and utility scripts also expect to be run from the project
directory. The local webcam dataset and generated preview are intentionally
ignored by Git.

## Project files

| File | Purpose |
| --- | --- |
| `dino_game.py` | Gesture-controlled Pygame runner |
| `main.py` | Full-frame CNN webcam diagnostic |
| `test_finetuned.py` | Hand-crop CNN webcam diagnostic |
| `collect_gestures.py` | Webcam image collection utility |
| `inspect_dataset.py` | Dataset preview utility |
| `best_finetuned_gesture.keras` | Gesture model used by the game |
| `gesture_cnn.keras` | Model used by `main.py` |
| `hand_landmarker.task` | MediaPipe hand detector model |
| `class_mapping.json` | Class labels used by `main.py` |
| `requirements.txt` | Python dependency versions |

## Troubleshooting

- **Camera could not be opened:** close other apps using the webcam, check
  operating-system camera permissions, and make sure the intended camera is
  available as camera index `0`.
- **Predictions are uncertain:** improve lighting, keep the full hand in frame,
  and hold the gesture steady.
- **A model or task file is missing:** run the script from the project
  directory and check that the required asset is present there.
