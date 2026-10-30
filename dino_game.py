
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "best_finetuned_gesture.keras"
LANDMARKER_PATH = "hand_landmarker.task"

CLASS_NAMES = [
    "Fist",
    "L-Sign",
    "Open Palm",
    "Pointing Up",
    "V-Sign"
]

# Gesture controls:
# Open Palm  = Jump
# Fist       = Duck
# Pointing Up = Start / Restart
# L-Sign     = Move Left
# V-Sign     = Move Right

CONFIDENCE_THRESHOLD = 0.65
CAMERA_INDEX = 0

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

GROUND_Y = 480
LANES = [350, 500, 650]


# ============================================================
# 2. LOAD CNN MODEL
# ============================================================

print("Loading gesture recognition model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# ============================================================
# 3. INITIALIZE MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# 4. INITIALIZE WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Webcam could not be opened. Check your camera."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# 5. INITIALIZE PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption("Hand Gesture Dino Runner")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 25, bold=True)
small_font = pygame.font.SysFont("Arial", 17)
big_font = pygame.font.SysFont("Arial", 46, bold=True)

WHITE = (245, 245, 245)
BLACK = (30, 35, 42)
GREEN = (45, 175, 95)
BLUE = (55, 125, 205)
YELLOW = (245, 190, 55)
SKY = (218, 239, 250)
GREY = (115, 130, 140)


# ============================================================
# 6. GAME VARIABLES
# ============================================================

MENU = "menu"
PLAYING = "playing"
GAME_OVER = "game_over"

game_state = MENU

score = 0
best_score = 0

game_speed = 6.0

player_lane = 1
player_y = GROUND_Y - 60

vertical_velocity = 0.0

is_jumping = False
is_ducking = False

obstacles = []
spawn_timer = 0.0


# ============================================================
# 7. GESTURE VARIABLES
# ============================================================

current_gesture = "No hand"
current_confidence = 0.0

prediction_history = deque(maxlen=5)

last_inference_time = 0.0
last_action_time = 0.0

camera_preview = None

INFERENCE_INTERVAL = 0.10
ACTION_COOLDOWN = 0.60


# ============================================================
# 8. TEXT DISPLAY FUNCTION
# ============================================================

def draw_text(
    text,
    x,
    y,
    color=BLACK,
    selected_font=font,
    center=False
):

    rendered = selected_font.render(
        text,
        True,
        color
    )

    rect = rendered.get_rect()

    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)

    screen.blit(rendered, rect)


# ============================================================
# 9. RESET GAME
# ============================================================

def reset_game():

    global game_state
    global score
    global game_speed
    global player_lane
    global player_y
    global vertical_velocity
    global is_jumping
    global is_ducking
    global obstacles
    global spawn_timer

    game_state = PLAYING

    score = 0
    game_speed = 6.0

    player_lane = 1

    player_y = GROUND_Y - 60

    vertical_velocity = 0.0

    is_jumping = False
    is_ducking = False

    obstacles = []
    spawn_timer = 0.0


# ============================================================
# 10. WEBCAM AND GESTURE RECOGNITION
# ============================================================

def process_camera():

    global current_gesture
    global current_confidence
    global last_inference_time
    global camera_preview

    success, frame = cap.read()

    if not success:
        current_gesture = "Camera Error"
        return

    frame = cv2.flip(frame, 1)

    # Keep clean frame for CNN
    original_frame = frame.copy()

    rgb = cv2.cvtColor(
        original_frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.monotonic() * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    display_frame = frame.copy()

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = original_frame.shape

        x_coords = [
            int(point.x * w)
            for point in hand
        ]

        y_coords = [
            int(point.y * h)
            for point in hand
        ]

        x1 = max(0, min(x_coords) - 30)
        y1 = max(0, min(y_coords) - 30)

        x2 = min(w, max(x_coords) + 30)
        y2 = min(h, max(y_coords) + 30)

        current_time = time.monotonic()

        if current_time - last_inference_time >= INFERENCE_INTERVAL:

            crop = original_frame[y1:y2, x1:x2]

            if crop.size > 0:

                # Preprocessing used by the working webcam test
                crop = cv2.resize(
                    crop,
                    (128, 128)
                )

                crop = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB
                )

                crop = crop.astype("float32") / 255.0

                prediction = model.predict(
                    np.expand_dims(crop, axis=0),
                    verbose=0
                )[0]

                prediction_history.append(prediction)

                # Smooth predictions across recent frames
                average_prediction = np.mean(
                    np.array(prediction_history),
                    axis=0
                )

                predicted_index = int(
                    np.argmax(average_prediction)
                )

                current_confidence = float(
                    average_prediction[predicted_index]
                )

                if current_confidence >= CONFIDENCE_THRESHOLD:

                    current_gesture = CLASS_NAMES[
                        predicted_index
                    ]

                else:
                    current_gesture = "Uncertain"

            last_inference_time = current_time

        # Simple bounding box; no landmark dots or lines
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2
        )

    else:

        current_gesture = "No hand"
        current_confidence = 0.0

        prediction_history.clear()

    camera_preview = display_frame


# ============================================================
# 11. GESTURE CONTROLS
# ============================================================

def perform_gesture_action():

    global game_state
    global last_action_time
    global player_lane
    global vertical_velocity
    global is_jumping
    global is_ducking

    current_time = time.monotonic()

    if current_time - last_action_time < ACTION_COOLDOWN:
        return

    # Start game
    if game_state == MENU:

        if current_gesture == "Pointing Up":

            reset_game()
            last_action_time = current_time

        return

    # Restart game
    if game_state == GAME_OVER:

        if current_gesture == "Pointing Up":

            reset_game()
            last_action_time = current_time

        return

    if game_state == PLAYING:

        if current_gesture == "Open Palm":

            if not is_jumping:

                vertical_velocity = -16.5

                is_jumping = True
                is_ducking = False

                last_action_time = current_time

        elif current_gesture == "Fist":

            is_ducking = True
            last_action_time = current_time

        elif current_gesture == "L-Sign":

            player_lane = max(
                0,
                player_lane - 1
            )

            last_action_time = current_time

        elif current_gesture == "V-Sign":

            player_lane = min(
                2,
                player_lane + 1
            )

            last_action_time = current_time

        elif current_gesture != "Fist":

            is_ducking = False


# ============================================================
# 12. UPDATE GAME
# ============================================================

def update_game(dt):

    global score
    global best_score
    global game_speed
    global spawn_timer
    global player_y
    global vertical_velocity
    global is_jumping
    global game_state

    if game_state != PLAYING:
        return

    # Jump physics
    if is_jumping:

        player_y += vertical_velocity

        vertical_velocity += 0.75

        if player_y >= GROUND_Y - 60:

            player_y = GROUND_Y - 60

            vertical_velocity = 0.0

            is_jumping = False

    # Spawn obstacles
    spawn_timer += dt

    if spawn_timer > max(
        0.8,
        1.5 - game_speed * 0.04
    ):

        obstacles.append({
            "x": SCREEN_WIDTH + 30,
            "lane": random.randrange(3),
            "kind": random.choice(["cactus", "bird"]),
            "passed": False
        })

        spawn_timer = 0.0

    # Player collision rectangle
    player_x = LANES[player_lane]

    player_height = (
        32 if is_ducking and not is_jumping
        else 60
    )

    player_top = (
        GROUND_Y - player_height
        if is_ducking and not is_jumping
        else player_y
    )

    player_rect = pygame.Rect(
        player_x - 22,
        int(player_top),
        44,
        player_height
    )

    # Move obstacles and check collisions
    for obstacle in obstacles:

        obstacle["x"] -= game_speed

        obstacle_height = (
            25 if obstacle["kind"] == "bird"
            else 45
        )

        obstacle_width = (
            45 if obstacle["kind"] == "bird"
            else 28
        )

        obstacle_y = GROUND_Y - obstacle_height

        obstacle_rect = pygame.Rect(
            int(obstacle["x"]),
            obstacle_y,
            obstacle_width,
            obstacle_height
        )

        if (
            obstacle["lane"] == player_lane
            and player_rect.colliderect(obstacle_rect)
        ):

            game_state = GAME_OVER

            best_score = max(
                best_score,
                score
            )

        if (
            not obstacle["passed"]
            and obstacle["x"] < player_x - 30
        ):

            obstacle["passed"] = True
            score += 1

    obstacles[:] = [
        obstacle
        for obstacle in obstacles
        if obstacle["x"] > -70
    ]

    # Gradually increase difficulty
    game_speed = min(
        12,
        game_speed + dt * 0.10
    )


# ============================================================
# 13. DRAW PLAYER
# ============================================================

def draw_player():

    x = LANES[player_lane]

    if is_ducking and not is_jumping:

        pygame.draw.rect(
            screen,
            GREEN,
            (x - 27, GROUND_Y - 32, 54, 28),
            border_radius=8
        )

        pygame.draw.circle(
            screen,
            GREEN,
            (x + 18, GROUND_Y - 30),
            14
        )

    else:

        pygame.draw.rect(
            screen,
            GREEN,
            (x - 22, int(player_y), 44, 60),
            border_radius=8
        )

        pygame.draw.circle(
            screen,
            GREEN,
            (x + 16, int(player_y) + 10),
            14
        )

        pygame.draw.circle(
            screen,
            BLACK,
            (x + 21, int(player_y) + 7),
            3
        )


# ============================================================
# 14. DRAW OBSTACLES
# ============================================================

def draw_obstacles():

    for obstacle in obstacles:

        x = int(obstacle["x"])

        if obstacle["kind"] == "cactus":

            pygame.draw.rect(
                screen,
                GREEN,
                (x, GROUND_Y - 45, 27, 45),
                border_radius=5
            )

            pygame.draw.rect(
                screen,
                GREEN,
                (x - 8, GROUND_Y - 30, 10, 18),
                border_radius=4
            )

            pygame.draw.rect(
                screen,
                GREEN,
                (x + 24, GROUND_Y - 36, 10, 18),
                border_radius=4
            )

        else:

            pygame.draw.ellipse(
                screen,
                BLACK,
                (x, GROUND_Y - 60, 45, 24)
            )


# ============================================================
# 15. WEBCAM PREVIEW
# ============================================================

def draw_camera_panel():

    panel = pygame.Rect(
        SCREEN_WIDTH - 270,
        18,
        250,
        185
    )

    pygame.draw.rect(
        screen,
        BLACK,
        panel,
        border_radius=10
    )

    draw_text(
        "WEBCAM",
        panel.x + 12,
        panel.y + 8,
        WHITE,
        small_font
    )

    if camera_preview is not None:

        preview = cv2.resize(
            camera_preview,
            (225, 125)
        )

        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB
        )

        preview_surface = pygame.surfarray.make_surface(
            np.rot90(preview)
        )

        screen.blit(
            preview_surface,
            (panel.x + 12, panel.y + 32)
        )

    draw_text(
        f"{current_gesture}: {current_confidence * 100:.0f}%",
        panel.x + 12,
        panel.y + 162,
        YELLOW,
        small_font
    )


# ============================================================
# 16. DRAW GAME SCREEN
# ============================================================

def draw_game():

    screen.fill(SKY)

    # Clouds
    pygame.draw.circle(
        screen, WHITE, (130, 100), 28
    )

    pygame.draw.circle(
        screen, WHITE, (165, 95), 35
    )

    pygame.draw.circle(
        screen, WHITE, (200, 100), 25
    )

    # Lanes
    for lane in LANES:

        pygame.draw.line(
            screen,
            (195, 215, 225),
            (lane, 210),
            (lane, GROUND_Y),
            2
        )

    # Ground
    pygame.draw.line(
        screen,
        GREY,
        (0, GROUND_Y),
        (SCREEN_WIDTH, GROUND_Y),
        5
    )

    if game_state == PLAYING:

        draw_obstacles()
        draw_player()

    draw_text(
        f"SCORE: {score}",
        22,
        18
    )

    draw_text(
        f"BEST: {best_score}",
        22,
        50,
        selected_font=small_font
    )

    # Controls guide
    controls = [
        "OPEN PALM: JUMP",
        "FIST: DUCK",
        "L-SIGN: LEFT",
        "V-SIGN: RIGHT",
        "POINTING UP: START / RESTART"
    ]

    for i, text in enumerate(controls):

        draw_text(
            text,
            22,
            390 + i * 23,
            selected_font=small_font
        )

    # Menu screen
    if game_state == MENU:

        draw_text(
            "HAND GESTURE DINO RUNNER",
            SCREEN_WIDTH // 2,
            250,
            selected_font=big_font,
            center=True
        )

        draw_text(
            "Show POINTING UP to start",
            SCREEN_WIDTH // 2,
            315,
            BLUE,
            center=True
        )

    # Game over screen
    elif game_state == GAME_OVER:

        overlay = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            pygame.SRCALPHA
        )

        overlay.fill(
            (15, 20, 30, 165)
        )

        screen.blit(
            overlay,
            (0, 0)
        )

        draw_text(
            "GAME OVER",
            SCREEN_WIDTH // 2,
            245,
            WHITE,
            big_font,
            True
        )

        draw_text(
            f"Final score: {score}",
            SCREEN_WIDTH // 2,
            305,
            WHITE,
            center=True
        )

        draw_text(
            "Show POINTING UP to restart",
            SCREEN_WIDTH // 2,
            355,
            YELLOW,
            center=True
        )

    draw_camera_panel()

    pygame.display.flip()


# ============================================================
# 17. MAIN GAME LOOP
# ============================================================

running = True

previous_time = time.monotonic()

try:

    while running:

        current_time = time.monotonic()

        dt = min(
            current_time - previous_time,
            0.05
        )

        previous_time = current_time

        # Handle keyboard and window events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q
                ):

                    running = False

                elif (
                    event.key == pygame.K_RETURN
                    and game_state in (MENU, GAME_OVER)
                ):

                    reset_game()

                elif (
                    event.key == pygame.K_SPACE
                    and game_state == PLAYING
                    and not is_jumping
                ):

                    vertical_velocity = -16.5
                    is_jumping = True
                    is_ducking = False

                elif (
                    event.key == pygame.K_DOWN
                    and game_state == PLAYING
                ):

                    is_ducking = True

                elif (
                    event.key == pygame.K_LEFT
                    and game_state == PLAYING
                ):

                    player_lane = max(
                        0,
                        player_lane - 1
                    )

                elif (
                    event.key == pygame.K_RIGHT
                    and game_state == PLAYING
                ):

                    player_lane = min(
                        2,
                        player_lane + 1
                    )

            elif (
                event.type == pygame.KEYUP
                and event.key == pygame.K_DOWN
            ):

                is_ducking = False

        # Run webcam recognition
        process_camera()

        # Apply recognized gesture
        perform_gesture_action()

        # Update game
        update_game(dt)

        # Draw everything
        draw_game()

        clock.tick(FPS)

finally:

    cap.release()

    landmarker.close()

    pygame.quit()

    cv2.destroyAllWindows()

    print("Game closed successfully.")
