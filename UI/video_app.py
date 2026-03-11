import cv2
import numpy as np
from collections import deque
from enum import Enum
import time
import platform

os = platform.system()
print(os)

#mode for the prediction
class Mode(Enum):
    FER_ONLY = 1 #only facial emotion recognition
    SER_ONLY = 2 #only speech emotion recognition
    FUSED = 3 #fusion of facial emotion and speech emotion recognition

#color palette
THEME = {"bg": (18, 18, 24), 
    "panel": (35, 36, 48),
    "panel2": (45, 48, 66),
    "text": (240, 240, 245),
    "muted": (180, 185, 195),
    "accent": (255, 170, 60),
    "cyan": (80, 220, 255),
    "green": (80, 220, 120),
    "red": (90, 90, 255),
    "yellow": (80, 220, 255),}

def emotion_color(emotion):
    colors = {"happy": (80, 220, 120),
        "sad": (255, 140, 90),
        "angry": (80, 80, 255),
        "surprise": (80, 220, 255),
        "fear": (180, 120, 255),
        "neutral": (200, 200, 210),}
    return colors.get(emotion.lower(), (255, 255, 255))

def draw_conf_bar(img, x, y, w, h, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 62, 75), -1)
    fill_w = int(w * max(0, min(1, value)))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (95, 98, 120), 1)


def draw_card(img, x1, y1, x2, y2, title, label, conf, accent):
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 36, 48), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 72, 90), 1)
    cv2.rectangle(img, (x1, y1), (x1 + 8, y2), accent, -1)
    cv2.putText(img, title, (x1 + 18, y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 245, 250), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 42), (x2 - 16, y1 + 42), (70, 72, 90), 1)
    cv2.putText(img, f"Label: {label.title()}", (x1 + 18, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 212, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"Confidence: {conf:.2f}", (x1 + 18, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 212, 220), 1, cv2.LINE_AA)
    draw_conf_bar(img, x1 + 18, y1 + 115, (x2 - x1) - 36, 14, conf, accent)


def draw_history(img, x, y, history):
    cv2.putText(img, "Recent History", (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (245,245,250), 1, cv2.LINE_AA)
    y += 20
    for i, item in enumerate(list(history)[:8]):
        yy = y + i * 34
        c = emotion_color(item)

        cv2.rectangle(img, (x, yy), (x + 210, yy + 24), (45,48,66), -1)
        cv2.rectangle(img, (x, yy), (x + 210, yy + 24), (80,84,104), 1)

        cv2.circle(img, (x + 14, yy + 12), 7, c, -1)

        cv2.putText(img, item.title(), (x + 30, yy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220,220,228), 1, cv2.LINE_AA)


def put_hud(img, text, label, subtext=""):
    h, w = img.shape[:2]
    bar_h = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    color = emotion_color(label)
    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    if subtext:
        cv2.putText(img, subtext, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

def put_footer(img, text):
    h, w = img.shape[:2]
    cv2.putText(img, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

def fake_fer():
    return ("neutral", 0.55)

def fake_ser():
    return ("happy", 0.61)

def late_fusion(fer, ser):
    return fer if fer[1] >= ser[1] else ser


def open_camera():
    for idx in [0, 1, 2, 3]:
        print(f"[INFO] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[INFO] Using camera index {idx}")
            return cap

        cap.release()

    return None

def main():
    print("[INFO] Starting...")

    cap = open_camera()

    if cap is None:
        print("[ERROR] Could not open a working webcam.")
        return

    print("[INFO] cap.isOpened() =", cap.isOpened())

    mode = Mode.FER_ONLY
    show_history = True
    history = deque(maxlen=10)

    last_print = time.time()

    mode = Mode.FER_ONLY
    show_history = True
    history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        print("[DEBUG] ret =", ret, "shape =", None if frame is None else frame.shape)

        if not ret or frame is None:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        fer = fake_fer()
        ser = fake_ser()
        fused = late_fusion(fer, ser)

        if mode == Mode.FER_ONLY:
            main_pred = fer
            mode_name = "MODE: FER ONLY (press 2 for SER, 3 for FUSED)"
        elif mode == Mode.SER_ONLY:
            main_pred = ser
            mode_name = "MODE: SER ONLY (press 1 for FER, 3 for FUSED)"
        else:
            main_pred = fused
            mode_name = "MODE: FUSED (press 1 for FER, 2 for SER)"

        history.appendleft(main_pred[0])

        h, w = frame.shape[:2]
        panel_w = 420
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = (30, 30, 30)

        left_view = canvas[:, :w]
        label, conf = main_pred
        put_hud(left_view, f"Prediction: {label.upper()} ({conf:.2f})", label, mode_name)

        px1, px2 = w + 20, w + panel_w - 20
        draw_card(canvas, px1, 20,  px2, 150, "FER", fer[0], fer[1], (80,220,255))
        draw_card(canvas, px1, 170, px2, 300, "SER", ser[0], ser[1], (80,220,120))
        draw_card(canvas, px1, 320, px2, 450, "FUSED", fused[0], fused[1], (255,170,60))

        if show_history:
            draw_history(canvas, px1, 470, history)

        put_footer(left_view, "Keys: 1=FER  2=SER  3=FUSED  H=history  Q/Esc=quit")
        cv2.imshow("Emotion Interface", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[INFO] Quit key pressed.")
            break
        elif key == ord("1"):
            mode = Mode.FER_ONLY
        elif key == ord("2"):
            mode = Mode.SER_ONLY
        elif key == ord("3"):
            mode = Mode.FUSED
        elif key in (ord("h"), ord("H")):
            show_history = not show_history

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()