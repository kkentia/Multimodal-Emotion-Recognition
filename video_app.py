import cv2 
import numpy as np
from collections import deque
from enum import Enum

#Modes 
class Mode(Enum):
    FER_ONLY = 1 #only FER for prediction 
    SER_ONLY = 2 #only SER for prediction
    FUSED = 3 #both used for prediction with late fusion

def draw_panel(img, x1, y1, x2, y2, title, lines):
    """Draws a boxed panel with title and multiple lines of text."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(img, title, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    y = y1 + 65
    for line in lines:
        cv2.putText(img, line, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28


def put_hud(img, text, subtext=""):
    """Top HUD bar."""
    h, w = img.shape[:2]
    bar_h = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)  # black bar
    alpha = 0.55
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if subtext:
        cv2.putText(img, subtext, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)


def put_footer(img, text):
    h, w = img.shape[:2]
    cv2.putText(img, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


# -------- Placeholder “model outputs” ----------
def fake_fer():
    # Replace with your FER inference later
    return ("neutral", 0.55)

def fake_ser():
    # Member 4 will replace this with microphone thread output
    return ("happy", 0.61)

def late_fusion(fer, ser):
    """
    Late fusion stub: choose the higher-confidence prediction.
    Later your team can replace with learned fusion or weighted averaging.
    """
    (f_label, f_conf) = fer
    (s_label, s_conf) = ser
    if f_conf >= s_conf:
        return (f_label, f_conf)
    return (s_label, s_conf)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try index 0/1/2)")

    mode = Mode.FER_ONLY
    show_history = True

    # Keep a short history for UI (last 10 predictions)
    history = deque(maxlen=10)

    cv2.namedWindow("Emotion Interface", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        # --- Your model results (currently stubbed) ---
        fer = fake_fer()
        ser = fake_ser()
        fused = late_fusion(fer, ser)

        # Pick what to show as the “main” prediction based on mode
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

        # --- Create a bigger canvas: camera + right panel ---
        h, w = frame.shape[:2]
        panel_w = 420
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

        # Put camera frame on left
        canvas[:, :w] = frame

        # Right panel background (dark gray)
        canvas[:, w:] = (30, 30, 30)

        # --- HUD on top of the camera area only ---
        # draw on a view slice so HUD doesn't cover the side panel unless you want it to
        left_view = canvas[:, :w]
        label, conf = main_pred
        put_hud(left_view, f"Prediction: {label.upper()}  ({conf:.2f})", mode_name)

        # --- Right panel: FER, SER, FUSED boxes ---
        px1, px2 = w + 20, w + panel_w - 20
        draw_panel(canvas, px1, 20,  px2, 130, "FER",   [f"label: {fer[0]}",   f"conf: {fer[1]:.2f}"])
        draw_panel(canvas, px1, 160, px2, 270, "SER",   [f"label: {ser[0]}",   f"conf: {ser[1]:.2f}"])
        draw_panel(canvas, px1, 300, px2, 410, "FUSED", [f"label: {fused[0]}", f"conf: {fused[1]:.2f}"])

        if show_history:
            # History panel
            y0 = 450
            cv2.putText(canvas, "History (latest first):", (px1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y = y0 + 30
            for i, item in enumerate(list(history)[:10]):
                cv2.putText(canvas, f"{i+1}. {item}", (px1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)
                y += 26

        put_footer(left_view, "Keys: 1=FER  2=SER  3=FUSED  H=history  Q/Esc=quit")

        cv2.imshow("Emotion Interface", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            mode = Mode.FER_ONLY
        elif key == ord('2'):
            mode = Mode.SER_ONLY
        elif key == ord('3'):
            mode = Mode.FUSED
        elif key in (ord('h'), ord('H')):
            show_history = not show_history

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()