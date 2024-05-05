import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Capture video 
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #  landmarks 
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Calculate FPS.
    end = time.time()
    totalTime = end - start
    fps_text = f'FPS: {int(1 / totalTime)}' if totalTime > 0 else 'FPS: Max'

    # Display FPS 
    cv2.putText(image, fps_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display 
    cv2.imshow('Face Landmarks', image)

    # Exit with ESC key 
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
