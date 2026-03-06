import cv2
import numpy as np
from collections import deque
from enum import Enum
import time
import platform

#mode for the prediction
class Mode(Enum):
    FER_ONLY = 1 #only FER used for the prediction 
    SER_ONLY = 2 #only SER used for the prediction 
    FUSED = 3 #FER and SER used for the prediction

def draw_panel(img, x1, y1, x2, y2, title, lines): #draws UI panel
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(img, title, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y = y1 + 65
    for line in lines:
        cv2.putText(img, line, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28

def emotion_color(emotion):
    colors = {
        "happy": (0,255,0),
        "sad": (255,0,0),
        "angry": (0,0,255),
        "surprise": (0,255,255),
        "fear": (255,255,0),
        "neutral": (200,200,200)}

    return colors.get(emotion.lower(), (255,255,255))

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


def main():
    print("[INFO] Starting...")

    # use the exact camera call that worked in your simple test
    cap = cv2.VideoCapture(1)
    print("[INFO] cap.isOpened() =", cap.isOpened())

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

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
        draw_panel(canvas, px1, 20,  px2, 130, "FER",   [f"label: {fer[0]}",   f"conf: {fer[1]:.2f}"])
        draw_panel(canvas, px1, 160, px2, 270, "SER",   [f"label: {ser[0]}",   f"conf: {ser[1]:.2f}"])
        draw_panel(canvas, px1, 300, px2, 410, "FUSED", [f"label: {fused[0]}", f"conf: {fused[1]:.2f}"])

        if show_history:
            y0 = 450
            cv2.putText(canvas, "History (latest first):", (px1, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y = y0 + 30
            for i, item in enumerate(list(history)[:10]):
                cv2.putText(canvas, f"{i+1}. {item}", (px1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
                y += 26

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