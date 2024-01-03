import cv2
import dlib
import numpy as np
import random

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


if __name__ == '__main__':
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    blink_goal = random.randint(1, 5)  # Número aleatório de piscadas
    blink_count = 0
    frame_counter = 0

    cap = cv2.VideoCapture(0)

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

            # Feedback visual
            cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if blink_count >= blink_goal:
                cv2.putText(frame, "Blink Goal Achieved!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
