"""

Hands demo with colorful lines;
https://google.github.io/mediapipe/solutions/hands.html
"""

import cv2
import mediapipe as mp


def gesture_detection(hand_landmarks_positions):
    thumb_tip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_pip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_finger_tip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_pip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_finger_tip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_pip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_finger_tip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_pip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_tip = hand_landmarks_positions.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (index_finger_tip.y > index_finger_pip.y
            and middle_finger_tip.y > middle_finger_pip.y
            and ring_finger_tip.y > ring_finger_pip.y
            and pinky_tip.y > pinky_pip.y
            and thumb_tip.x < index_finger_tip.x):
        state = "Pause the music"
        return state

    elif (index_finger_tip.y < index_finger_pip.y
            and pinky_tip.y < pinky_pip.y
            and middle_finger_tip.y > middle_finger_pip.y
            and ring_finger_tip.y > ring_finger_pip.y
            and thumb_tip.x < index_finger_tip.x):
        state = "Start the music"
        return state

    elif (index_finger_tip.y > index_finger_pip.y
            and middle_finger_tip.y > middle_finger_pip.y
            and ring_finger_tip.y > ring_finger_pip.y
            and pinky_tip.y > pinky_pip.y
            and thumb_tip.x > index_finger_tip.x):
        state = "Next song"
        return state

    elif (index_finger_tip.y < index_finger_pip.y
          and pinky_tip.y > pinky_pip.y
          and middle_finger_tip.y < middle_finger_pip.y
          and ring_finger_tip.y > ring_finger_pip.y
          and thumb_tip.x < index_finger_tip.x):
        state = "Previous song"
        return state



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    actual_state = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if not results.multi_hand_landmarks:
            continue

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            state = gesture_detection(hand_landmarks_positions=hand_landmarks)

            if state is not None and state is not actual_state:
                actual_state = state
                print(state)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


