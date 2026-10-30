
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ==========================================
# 1. LOAD TRAINED CNN
# ==========================================
print("Loading CNN model...")

model = tf.keras.models.load_model("gesture_cnn.keras")

with open("class_mapping.json", "r") as file:
    class_mapping = json.load(file)

IMG_SIZE = 128

print("Model loaded successfully!")
print("Gesture classes:", class_mapping)


# ==========================================
# 2. CONFIGURE MEDIAPIPE
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
# 3. OPEN WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nWebcam started.")
print("Show your hand clearly.")
print("Press Q to exit.")

last_timestamp = -1


# ==========================================
# 4. LIVE CNN PREDICTION
# ==========================================
try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            # Mirror the webcam.
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe hand detection.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                cv2.getTickCount()
                / cv2.getTickFrequency()
                * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp + 1
            )
            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                # Draw detected hand landmarks.
                for hand in result.hand_landmarks:
                    for point in hand:
                        x = int(point.x * frame.shape[1])
                        y = int(point.y * frame.shape[0])

                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 255, 0),
                            -1
                        )

                # IMPORTANT:
                # Use the entire webcam frame for CNN input.
                resized = cv2.resize(
                    rgb_frame,
                    (IMG_SIZE, IMG_SIZE)
                )

                normalized = (
                    resized.astype(np.float32) / 255.0
                )

                probabilities = model.predict(
                    np.expand_dims(normalized, axis=0),
                    verbose=0
                )[0]

                class_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[class_index]
                )

                gesture = class_mapping[
                    str(class_index)
                ]

                # Display the top prediction.
                cv2.putText(
                    frame,
                    f"FULL FRAME: {gesture}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence * 100:.1f}%",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # Display confidence for every class.
                y_text = 105

                for index, probability in enumerate(
                    probabilities
                ):
                    class_name = class_mapping[str(index)]
                    percentage = float(probability) * 100

                    color = (
                        (0, 255, 255)
                        if index == class_index
                        else (255, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        f"{class_name}: {percentage:.1f}%",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        color,
                        2
                    )

                    y_text += 27

            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Full Frame CNN Diagnostic",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Diagnostic test completed.")
