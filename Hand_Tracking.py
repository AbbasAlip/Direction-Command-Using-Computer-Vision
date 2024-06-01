import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
prev_x = None
movement = ""
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x = hand_landmarks.landmark[8].x


            if prev_x is not None:
                if x < prev_x - 0.01:
                    movement = "left"
                elif x > prev_x + 0.01:
                    movement = "right"
                else:
                    movement = ""


            prev_x = x


            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    if movement:
        cv2.putText(frame, movement, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Hand Movement Detection', frame)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()


