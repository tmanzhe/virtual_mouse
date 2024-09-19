import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller

# create mouse controller instance
mouse = Controller()

# get screen dimensions
screen_width, screen_height = pyautogui.size()

# initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)


def get_index_finger_tip(processed):
    # check if hands are detected
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # get the first detected hand
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def reposition_mouse(index_finger_tip):
    # move mouse based on index finger tip position
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def check_left_click(landmark_list, thumb_index_dist):
    # determine if a left click gesture is made
    return (
        util.calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )


def check_right_click(landmark_list, thumb_index_dist):
    # determine if a right click gesture is made
    return (
        util.calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        util.calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )


def check_double_click(landmark_list, thumb_index_dist):
    # determine if a double click gesture is made
    return (
        util.calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )


def check_screenshot(landmark_list, thumb_index_dist):
    # determine if a screenshot gesture is made
    return (
        util.calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )


def identify_gesture(frame, landmark_list, processed):
    # identify gestures based on landmark positions
    if len(landmark_list) >= 21:

        index_finger_tip = get_index_finger_tip(processed)
        thumb_index_dist = util.calculate_distance([landmark_list[4], landmark_list[5]])

        if util.calculate_distance([landmark_list[4], landmark_list[5]]) < 50 and util.calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            reposition_mouse(index_finger_tip)
        elif check_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif check_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif check_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif check_screenshot(landmark_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def main():
    # main function to run the application
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror the frame
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB for mediapipe
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # get the first detected hand
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)  # draw hand landmarks
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))  # store landmark positions

            identify_gesture(frame, landmark_list, processed)  # identify gestures

            cv2.imshow('Frame', frame)  # display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # quit on pressing 'q'
                break
    finally:
        cap.release()  # release the camera
        cv2.destroyAllWindows()  # close all OpenCV windows


if __name__ == '__main__':
    main()
