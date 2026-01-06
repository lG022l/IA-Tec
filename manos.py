import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = frame.shape

    manoX1 = manoY1 = manoX2 = manoY2 = None
    puntoDeseado = 8

    if results.multi_hand_landmarks:
        count = 0
        for hand_landmarks in results.multi_hand_landmarks:
            count += 1
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx, pointMark in enumerate(hand_landmarks.landmark):
                if idx == puntoDeseado:
                    x_px = int(pointMark.x * w)
                    y_px = int(pointMark.y * h)
                    if count == 1:
                        manoX1, manoY1 = x_px, y_px
                    elif count == 2:
                        manoX2, manoY2 = x_px, y_px

    if manoX1 is not None and manoX2 is not None:
        dist = math.dist([manoX1, manoY1], [manoX2, manoY2])
        angulo = math.atan2(manoY2 - manoY1, manoX2 - manoX1)
        grados = math.degrees(angulo)

        cx = int((manoX1 + manoX2) / 2)
        cy = int((manoY1 + manoY2) / 2)

        ancho = int(dist)
        alto = int(dist / 2)

        rect_pts = [
            (-ancho//2, -alto//2),
            ( ancho//2, -alto//2),
            ( ancho//2,  alto//2),
            (-ancho//2,  alto//2)
        ]

        # Rotar manualmente los puntos y trasladar al centro
        pts_rotados = []
        for (x, y) in rect_pts:
            xr = x * math.cos(angulo) - y * math.sin(angulo)
            yr = x * math.sin(angulo) + y * math.cos(angulo)
            pts_rotados.append((int(cx + xr), int(cy + yr)))

        pts_cv = np.array(pts_rotados, np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(frame, [pts_cv], isClosed=True, color=(0, 255, 0), thickness=2)



    cv2.imshow("Rectangulo rotado", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
