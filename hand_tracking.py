
import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

SENSITIVITY = 2.8
SMOOTHING = 0.55

PINCH_THRESHOLD = 0.35
HOLD_TIME = 0.22
COOLDOWN = 0.5
CONFIRM_FRAMES = 2

smooth_x = screen_w / 2
smooth_y = screen_h / 2

pinch_start = None
pinch_latched = False
last_click = 0
pinch_frames = 0


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_angle(a, b, c):
    """Calculate the angle at landmark b, in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cosine = (
        ba[0] * bc[0] + ba[1] * bc[1]
    ) / (mag_ba * mag_bc)

    cosine = max(-1, min(1, cosine))
    return math.degrees(math.acos(cosine))


try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        status = "Show one hand"
        normalized_pinch = 0.0
        index_angle = 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            lm = hand.landmark
            now = time.time()

            tip = lm[8]

            target_x = int(
                screen_w / 2
                + (tip.x - 0.5) * screen_w * SENSITIVITY
            )
            target_y = int(
                screen_h / 2
                + (tip.y - 0.5) * screen_h * SENSITIVITY
            )

            target_x = max(0, min(screen_w - 1, target_x))
            target_y = max(0, min(screen_h - 1, target_y))

            pinch_distance = distance(lm[4], lm[8])
            hand_size = distance(lm[5], lm[17])

            normalized_pinch = (
                pinch_distance / max(hand_size, 0.001)
            )

            index_angle = finger_angle(lm[5], lm[6], lm[7])

            pinch_candidate = normalized_pinch < PINCH_THRESHOLD

            if pinch_candidate:
                pinch_frames += 1
            else:
                pinch_frames = 0

            pinching = pinch_frames >= CONFIRM_FRAMES

            if not pinching:
                smooth_x += (
                    target_x - smooth_x
                ) * SMOOTHING

                smooth_y += (
                    target_y - smooth_y
                ) * SMOOTHING

                try:
                    pyautogui.moveTo(
                        int(smooth_x), int(smooth_y)
                    )
                except pyautogui.FailSafeException:
                    break

            if pinching:
                status = "Pinch detected"

                if pinch_start is None:
                    pinch_start = now

                if (
                    now - pinch_start >= HOLD_TIME
                    and not pinch_latched
                    and now - last_click >= COOLDOWN
                ):
                    pyautogui.click(button="left")
                    last_click = now
                    pinch_latched = True
                    status = "LEFT CLICK!"

            else:
                pinch_start = None
                pinch_latched = False
                status = "Cursor active"

            cv2.putText(
                frame, status, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Pinch ratio: {normalized_pinch:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Index angle: {index_angle:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2
            )

        else:
            pinch_start = None
            pinch_latched = False
            pinch_frames = 0

        cv2.imshow("AirControl - Gesture Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
