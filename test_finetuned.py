
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

MODEL_PATH = "best_finetuned_gesture.keras"

class_names = [
    "Fist",
    "L-Sign",
    "Open Palm",
    "Pointing Up",
    "V-Sign"
]

model = tf.keras.models.load_model(MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(
        cv2.getTickCount() / cv2.getTickFrequency() * 1000
    )

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        x_coords = [int(p.x * w) for p in hand]
        y_coords = [int(p.y * h) for p in hand]

        x1 = max(0, min(x_coords) - 30)
        y1 = max(0, min(y_coords) - 30)
        x2 = min(w, max(x_coords) + 30)
        y2 = min(h, max(y_coords) + 30)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            crop = cv2.resize(crop, (128, 128))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype("float32") / 255.0

            prediction = model.predict(
                np.expand_dims(crop, axis=0),
                verbose=0
            )[0]

            # Display all five class probabilities
            panel_x = 15
            panel_y = 25

            cv2.putText(
                frame, "GESTURE PROBABILITIES",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2
            )

            for i, name in enumerate(class_names):
                percentage = float(prediction[i] * 100)

                y = 65 + i * 42

                cv2.putText(
                    frame,
                    f"{name}: {percentage:.1f}%",
                    (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if i == np.argmax(prediction)
                    else (255, 255, 255),
                    2
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.imshow("All Gesture Probabilities", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
