import cv2
import dlib
import numpy as np
import random
import time

# Carregar o detector de rostos e o predictor de marcos faciais da dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Função para calcular a razão de aspecto do olho
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# Função simplificada para detectar movimento da cabeça
def detect_head_movement(landmarks, prev_position):
    try:
        nose = landmarks.part(30)  # Ponto do nariz
        if prev_position is None:
            return False, (nose.x, nose.y)

        movement_threshold = 10
        if abs(nose.x - prev_position[0]) > movement_threshold or \
                abs(nose.y - prev_position[1]) > movement_threshold:
            return True, (nose.x, nose.y)

        return False, (nose.x, nose.y)
    except Exception as e:
        print(f"Erro na detecção de movimento da cabeça: {e}")
        return False, prev_position


if __name__ == '__main__':
    # Inicializações
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    TIME_LIMIT = 10  # Limite de tempo em segundos para cada etapa

    blink_goal = random.randint(1, 5)  # Número aleatório de piscadas
    blink_count = 0
    frame_counter = 0
    start_time = time.time()
    stage = 1  # Começa no estágio de piscadas
    prev_position = None  # Para armazenar a posição anterior do nariz

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        time_remaining = TIME_LIMIT - (time.time() - start_time)  # Tempo restante

        try:
            for face in faces:
                landmarks = predictor(gray, face)

                # Coordenadas dos olhos
                leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                # Etapa 1: Detecção de Piscadas
                if stage == 1:
                    cv2.putText(frame, f"Pisque {blink_goal} vezes (restam {time_remaining:.2f}s)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if ear < EYE_AR_THRESH:
                        frame_counter += 1
                    elif frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_count += 1
                        frame_counter = 0

                    if blink_count >= blink_goal:
                        stage = 2  # Passa para a etapa de movimento da cabeça
                        start_time = time.time()
                        prev_position = (landmarks.part(30).x, landmarks.part(30).y)

                    if time_remaining <= 0:
                        stage = -1

                # Etapa 2: Detecção de Movimento da Cabeça
                elif stage == 2:
                    cv2.putText(frame, f"Mova a cabeça (restam {time_remaining:.2f}s)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                    head_moved, prev_position = detect_head_movement(landmarks, prev_position)
                    if head_moved:
                        stage = 3

                if time_remaining <= 0:
                    stage = -1

            # Feedback final e saída do loop
            if stage == -1:
                cv2.putText(frame, "Falha na validação", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time.sleep(2)
                break
            elif stage == 3:
                cv2.putText(frame, "Validação bem-sucedida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(2)
                break

        except Exception as e:
            print(f"Erro durante o processamento: {e}")
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
