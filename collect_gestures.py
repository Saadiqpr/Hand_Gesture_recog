
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("webcam_dataset")
TARGET_IMAGES = 100
CAPTURE_INTERVAL = 0.35
IMG_SIZE = 128

GESTURES = {
    ord("1"): "Fist",
    ord("2"): "L-Sign",
    ord("3"): "Open Palm_Open Hand",
    ord("4"): "Pointing Up",
    ord("5"): "V-Sign"
}

# Create folders
for gesture in GESTURES.values():
    (DATA_DIR / gesture).mkdir(parents=True, exist_ok=True)

# ==========================================
# MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# HAND IMAGE PREPROCESSING
# ==========================================

def prepare_hand_image(frame, landmarks):
    height, width = frame.shape[:2]

    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    padding = max(
        int(max(x2 - x1, y2 - y1) * 0.45),
        20
    )

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    # Square padding with black background,
    # matching the earlier webcam preprocessing.
    h, w = crop.shape[:2]
    side = max(h, w)

    square = np.zeros((side, side, 3), dtype=np.uint8)

    y_offset = (side - h) // 2
    x_offset = (side - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = crop

    # CNN training input: RGB, 128x128, normalized.
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    normalized = resized.astype(np.float32) / 255.0

    return normalized, (x1, y1, x2, y2)


# ==========================================
# CAMERA INITIALIZATION
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Check camera permissions."
    )

print("\n===================================")
print(" HAND GESTURE DATA COLLECTION")
print("===================================")
print("1 = Fist")
print("2 = L-Sign")
print("3 = Open Palm")
print("4 = Pointing Up")
print("5 = V-Sign")
print("SPACE = Start/Pause collection")
print("Q = Exit")
print("\nTarget: 100 images per gesture")
print("===================================\n")

selected_gesture = None
collecting = False
last_capture_time = 0
last_timestamp = -1

# ==========================================
# DATA COLLECTION
# ==========================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(
                cv2.getTickCount()
                / cv2.getTickFrequency()
                * 1000
            )

            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            hand_image = None
            box = None

            if result.hand_landmarks:
                hand_image, box = prepare_hand_image(
                    frame,
                    result.hand_landmarks[0]
                )

                if box:
                    x1, y1, x2, y2 = box

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

            # Current gesture count
            count = 0

            if selected_gesture:
                folder = DATA_DIR / selected_gesture
                count = len(list(folder.glob("*.jpg")))

            # Automatic image saving
            current_time = time.time()

            if (
                collecting
                and selected_gesture
                and hand_image is not None
                and count < TARGET_IMAGES
                and current_time - last_capture_time >= CAPTURE_INTERVAL
            ):
                filename = (
                    DATA_DIR
                    / selected_gesture
                    / f"{int(current_time * 1000)}.jpg"
                )

                # Save the RGB image in standard JPEG format.
                save_image = cv2.cvtColor(
                    (hand_image * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )

                cv2.imwrite(str(filename), save_image)

                last_capture_time = current_time
                count += 1

                print(
                    f"{selected_gesture}: "
                    f"{count}/{TARGET_IMAGES}"
                )

                if count >= TARGET_IMAGES:
                    collecting = False
                    print(
                        f"{selected_gesture} collection completed!"
                    )

            # Display instructions
            cv2.putText(
                display,
                "1:Fist  2:L  3:Open  4:Point  5:V",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            if selected_gesture:
                status = (
                    f"{selected_gesture}: "
                    f"{count}/{TARGET_IMAGES}"
                )
            else:
                status = "Select a gesture using keys 1-5"

            cv2.putText(
                display,
                status,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            collection_status = (
                "COLLECTING"
                if collecting
                else "PAUSED"
            )

            cv2.putText(
                display,
                collection_status,
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if collecting else (255, 255, 255),
                2
            )

            cv2.imshow(
                "Hand Gesture Data Collection",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key in GESTURES:
                selected_gesture = GESTURES[key]
                collecting = False
                print(
                    f"\nSelected: {selected_gesture}"
                )

            elif key == 32:
                if selected_gesture is None:
                    print("Select a gesture first.")
                else:
                    folder = DATA_DIR / selected_gesture
                    count = len(list(folder.glob("*.jpg")))

                    if count >= TARGET_IMAGES:
                        print("This gesture already has 100 images.")
                    else:
                        collecting = not collecting
                        print(
                            "Collection started."
                            if collecting
                            else "Collection paused."
                        )

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\nData collection session ended.")
print("Dataset folder:", DATA_DIR.resolve())
