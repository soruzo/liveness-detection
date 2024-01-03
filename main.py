import cv2
import dlib
import numpy as np
import random
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


if __name__ == '__main__':
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 4
    blink_goal = random.randint(2, 6)
    blink_count = 0
    frame_counter = 0

    cap = cv2.VideoCapture(0)

    # Mensagem inicial
    for _ in range(90):  # Aprox. 3 segundos
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Prepare-se para piscar {blink_goal} vezes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(33)

    # Contagem regressiva
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Iniciando em {i}", (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1000)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            # Coordenadas dos olhos
            leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                frame_counter += 1
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_count += 1
                    frame_counter = 0

        time_elapsed = time.time() - start_time
        if time_elapsed > 10:
            break

        cv2.putText(frame, f"Piscadas: {blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # Feedback final
    if blink_count == blink_goal:
        message = "Usuario validado com sucesso!."
        color = (255, 0, 0)  # Azul
    else:
        message = "Usuario invalido."
        color = (0, 0, 255)  # Vermelho

    cv2.putText(frame, message, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(2000)  # Aguarda 2 segundos

    cv2.destroyAllWindows()
