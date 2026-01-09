"""
Autorzy:
- Patryk Kośmider
- Ziemowit Orlikowski

Prototyp maszyny do gry w "Baba Jaga patrzy"
System wykorzystuje kamerę internetową lub plik wideo do analizy obrazu w czasie rzeczywistym.
Określa, czy osoby na obrazie są graczami (ubranie zielone) czy strażnikami (ubranie czerwone).
Gracze mogą się poddać, podnosząc ręce nad głowę, co jest rozpoznawane przez system.
Jeśli gracz się poruszy, zostaje "wyeliminowany" (oznaczone na czerwono z celownikiem na głowie).

Przygotowanie do uruchomienia - wymagania:

Instalacja pakietów:
  pip install numpy opencv-python, argparse, mediapipe

Przykładowe uruchomienia:
    python computer_vision.py --video 0 (dla inputu z kamery)
    python computer_vision.py --video sample_video.mp4 (dla inputu z pliku wideo)

"""

import cv2
import numpy as np
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_heavy.task"
TARGET_WIDTH = 1280
persistence = {}


def get_color_status(frame, x1, y1, x2, y2):
    """
    Określa, czy postać jest graczem (zielony), strażnikiem (czerwony) czy nieznana (inny kolor).
    Zwraca "player", "guard" lub "unknown".
    :param frame: Klatka obrazu
    :param x1, y1, x2, y2: Współrzędne prostokąta postaci
    :return: status koloru postaci
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_area = roi.shape[0] * roi.shape[1]

    # Maska zielona (Gracze)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    # Maska czerwona (Straznicy)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([8, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_count = np.count_nonzero(green_mask)
    red_count = np.count_nonzero(red_mask)

    # Procentowy udzial koloru
    green_pct = green_count / roi_area
    red_pct = red_count / roi_area

    if red_pct > 0.10 and red_pct > green_pct:
        return "guard"
    if green_pct > 0.08:
        return "player"

    return "unknown"


def draw_crosshair(frame, cx, cy, size=20, color=(0, 0, 255)):
    """
    Rysuje celownik na głowie gracza.
    :param frame: Klatka obrazu
    :param cx, cy: Współrzędne środka celownika
    :param size: Rozmiar celownika
    :param color: Kolor celownika
    :return: None
    """

    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)


def initialize_source(source):
    """
    Inicjalizuje kamerę/wideo oraz model MediaPipe.
    :param source: Źródło wideo (numer kamery lub ścieżka do pliku)
    :return: obiekt VideoCapture oraz model PoseLandmarker
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Ustawienia modelu MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    # Konfiguracja PoseLandmarker (model, ilosc szukanych postaci, progi pewności)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=50,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return cap, landmarker

def run_vision(source):
    """
    Główna pętla przetwarzania wideo.
    :param source: Źródło wideo (numer kamery lub ścieżka do pliku)
    :return: None
    """
    cap, landmarker = initialize_source(source)
    ret, frame = cap.read()

    # Przygotowanie do detekcji ruchu
    frame = cv2.resize(
        frame, (TARGET_WIDTH, int(frame.shape[0] * TARGET_WIDTH / frame.shape[1]))
    )
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(
            frame, (TARGET_WIDTH, int(frame.shape[0] * TARGET_WIDTH / frame.shape[1]))
        )
        h_img, w_img = frame.shape[:2]
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Konwersja obrazu do formatu MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_res = landmarker.detect_for_video(mp_image, ts)

        # Detekcja ruchu - roznica klatek
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(
            cv2.GaussianBlur(prev_gray, (11, 11), 0),
            cv2.GaussianBlur(curr_gray, (11, 11), 0),
        )
        _, m_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        if pose_res.pose_landmarks:
            for idx, landmarks in enumerate(pose_res.pose_landmarks):
                if idx not in persistence:
                    persistence[idx] = {"surrender": 0, "motion": 0}

                # Pobranie punktow landmarkowych - nos i nadgarstki
                # Nadgarstki (powyzej nosa - poddanie sie)
                nose = landmarks[0]
                lw, rw = landmarks[15], landmarks[16]

                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1, y1 = int(min(xs) * w_img), int(min(ys) * h_img)
                x2, y2 = int(max(xs) * w_img), int(max(ys) * h_img)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                body_h = y2 - y1

                # Rozpoznawanie roli
                current_role = get_color_status(frame, x1, y1, x2, y2)
                if current_role == "guard":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        frame, "STRAZNIK", (x1, y1 - 10), 0, 0.7, (0, 165, 255), 2
                    )
                    continue

                is_player = current_role == "player"
                is_unknown = current_role == "unknown"

                nose_y = nose.y * h_img
                margin = body_h * 0.1

                if (lw.y * h_img < (nose_y - margin)) and (
                    rw.y * h_img < (nose_y - margin)
                ):
                    # ile klatek status sie utrzymuje
                    persistence[idx]["surrender"] = 2
                else:
                    if persistence[idx]["surrender"] > 0:
                        persistence[idx]["surrender"] -= 1

                body_roi_m = m_mask[y1:y2, x1:x2]
                is_moving = False

                if body_roi_m.size > 0:
                    motion_factor = np.count_nonzero(body_roi_m) / body_roi_m.size
                    # Ruch jest wykryty jesli przekroczy prog 0.03
                    if motion_factor > 0.03:
                        persistence[idx]["motion"] = 2
                    else:
                        if persistence[idx]["motion"] > 0:
                            persistence[idx]["motion"] -= 1

                is_moving = persistence[idx]["motion"] > 0

                # GRACZ PODDANY
                if persistence[idx]["surrender"] > 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        "PODDAL SIE (OK)",
                        (x1, y1 - 10),
                        0,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # ELIMINACJA ZA RUCH
                elif is_moving and is_player:
                    hx, hy = int(nose.x * w_img), int(nose.y * h_img)
                    draw_crosshair(frame, hx, hy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "ELIMINACJA! (RUCH)",
                        (x1, y1 - 10),
                        0,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # ZWYKŁY GRACZ
                elif is_player:
                    hx, hy = int(nose.x * w_img), int(nose.y * h_img)
                    draw_crosshair(frame, hx, hy, color=(0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, "GRACZ", (x1, y1 - 10), 0, 0.7, (0, 255, 0), 2)
                
                # NIEZNANY
                elif is_unknown:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(
                        frame, "NIEZNANY", (x1, y1 - 10), 0, 0.7, (255, 0, 0), 2
                    )

        cv2.imshow("Green light - red light", frame)
        prev_gray = curr_gray.copy()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="0")
    args = parser.parse_args()
    run_vision(args.video)
