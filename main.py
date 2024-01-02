import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Função para calcular a razão de aspecto do olho
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


if __name__ == '__main__':
    # Limiar para determinar se o olho está fechado
    EYE_AR_THRESH = 0.3

    # Capturar vídeo da webcam
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

            # Calcular a razão de aspecto dos olhos
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Média da razão de aspecto para ambos os olhos
            ear = (leftEAR + rightEAR) / 2.0

            # Verificar se a razão de aspecto está abaixo do limiar
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Olho Fechado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Olho Aberto", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
