import streamlit as st
import cv2
import numpy as np
# import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av

import mediapipe as mp
from pynput.keyboard import Key, Controller
import time

st.set_page_config(layout="wide")
st.title("Human Body based Virtual Controller")
st.write("TAI (Tekken AI)")
col1, col2 = st.columns([1, 1])

# Helper functions

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

moves = ["right_punch", "lseft_punch", "left_kick", "right_kick"]
move_mapping = {
    "right_punch": "a",
    "left_punch": "s",
    "left_kick": "z",
    "right_kick": "x"
}

keyboard = Controller()
delay = 0.2

move = 'still'
stage = 'down'
# prev_key = ""


def press_key(s):
    #     if prev_key != s:

    print('key pressed', s)
    keyboard.press(s)
#     else: pass
    # print('Pressed', s)
    time.sleep(delay)
    keyboard.release(s)
#     prev_key = s
    time.sleep(delay)


def calculate_dist(a, b):
    a = np.array(a)
    b = np.array(b)

    dist = np.sqrt(np.square(b[1] - a[1]) + np.square(b[0] - a[0]))
    return dist


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


################################

# st.write("Hello world")
with col1:
    st.components.v1.iframe(
        "https://www.retrogames.cc/embed/40238-tekken-3.html", width=600, height=450, scrolling=False)


def transform(frame: av.VideoFrame):
    global move, stage
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    img = frame.to_ndarray(format="bgr24")

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)
    hand_results = hands.process(image)
    # print(hand_results)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Left Arm
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Right Arm
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Left Leg
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Right Leg
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Upper body
        # mouth_left = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
        #               landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
        # mouth_right = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
        #                landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
#             nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # mid_shoulder = [(right_shoulder[0] + left_shoulder[0]) / 2,
        #                 (right_shoulder[1] + left_shoulder[1]) / 2]

        # Hand models
        # right hand

        # Calulate distance
        right_shoulder_left_wrist_dist = calculate_dist(
            right_shoulder, left_wrist)
        left_shoulder_right_wrist_dist = calculate_dist(
            left_shoulder, right_wrist)

        # Calculate angle
        angle_left_arm = calculate_angle(
            left_shoulder, left_elbow, left_wrist)
        angle_right_arm = calculate_angle(
            right_shoulder, right_elbow, right_wrist)
        angle_left_leg = calculate_angle(left_hip, left_knee, left_ankle)
        angle_right_leg = calculate_angle(
            right_hip, right_knee, right_ankle)

        # angle_mid_left_shoulder = calculate_angle(
        #     mouth_left, mid_shoulder, left_shoulder)
        # angle_mid_right_shoulder = calculate_angle(
        #     mouth_right, mid_shoulder, right_shoulder)

#             angle_mid_left_shoulder = calculate_angle(nose, mid_shoulder, left_shoulder)
#             angle_mid_right_shoulder = calculate_angle(nose, mid_shoulder, right_shoulder)

        # Visualize angle
        cv2.putText(image, str(round(angle_left_arm, 2)), tuple(np.multiply(left_elbow, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(angle_right_arm, 2)), tuple(np.multiply(right_elbow, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(angle_left_leg, 2)), tuple(np.multiply(left_knee, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(angle_right_leg, 2)), tuple(np.multiply(right_knee, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # cv2.putText(image, str(round(angle_mid_left_shoulder, 2)), tuple(np.multiply(left_shoulder, [
        #             640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(image, str(round(angle_mid_right_shoulder, 2)), tuple(np.multiply(right_shoulder, [
        # 640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, str(round(right_shoulder_left_wrist_dist, 2)), tuple(np.multiply(left_wrist, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(round(left_shoulder_right_wrist_dist, 2)), tuple(np.multiply(right_wrist, [
                    640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Move Logic
        if angle_left_arm > 180 and angle_right_arm > 180 and angle_left_leg > 180 and angle_right_leg > 180:
            move = 'still'
            stage = 'down'

        # Upper Body (Head movement)
#         elif angle_mid_left_shoulder < 70:
#             stage = "head_left_tilt"

#         elif angle_mid_left_shoulder >= 80 and stage == "head_left_tilt":
#             move = "left_head_tilt"
#             key = Key.left
# #                 print("left arrow")
#             press_key(key)
#             stage = "down"

#         elif angle_mid_right_shoulder < 70:
#             stage = "head_right_tilt"

#         elif angle_mid_right_shoulder >= 80 and stage == "head_right_tilt":
#             move = "right_head_tilt"
#             key = Key.right
# #                 print("right arrow")
#             press_key(key)
#             stage = "down"

        # Upper body (jump) drpawancse@gmail.com
        elif left_shoulder_right_wrist_dist < 0.20 and right_shoulder_left_wrist_dist < 0.20:
            key = Key.up
            press_key(key)
#                 print("jump")

        # Squat (do not uncomnent this)
        # elif angle_left_leg < 155 and angle_right_leg < 155:
        #     stage = "squat"
        #     move = "squat"
        #     press_key(Key.down)

        # elif angle_left_leg > 155 or angle_right_leg > 155:
        #     stage = "still"

        # Arms
        elif angle_left_arm < 45:
            stage = 'left_arm_up'

        elif angle_left_arm > 120 and stage == "left_arm_up":
            move = "left_punch"
            key = move_mapping[move]
            press_key(key)
            stage = "down"

        elif angle_right_arm < 45:
            stage = 'right_arm_up'

        elif angle_right_arm > 120 and stage == "right_arm_up":
            move = "right_punch"
            key = move_mapping[move]
            press_key(key)
            stage = "down"

        # legs
        elif angle_left_leg < 150:
            stage = 'left_leg_up'

        elif angle_left_leg > 170 and stage == "left_leg_up":
            move = "left_kick"
            key = move_mapping[move]
            press_key(key)
            stage = "down"

        elif angle_right_leg < 150:
            stage = 'right_leg_up'

        elif angle_right_leg > 170 and stage == "right_leg_up":
            move = "right_kick"
            key = move_mapping[move]
            press_key(key)
            stage = "down"
    except:
        pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    # print('hand_results', hand_results.multi_hand_landmarks)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            #         handIndex = hand_results.multi_hand_landmarks.index(hand_landmarks)
            #         handLabel = hand_results.multi_handedness[handIndex].classification[0].label
            #         print(handLabel)

            #         handLandmarks = []

            #         left_hand_angle = 0
            #         # right_hand_angle = 0

            #         # Fill list with x and y positions of each landmark
            #         for landmarks in hand_landmarks.landmark:
            #             handLandmarks.append([landmarks.x, landmarks.y])

            #         # print(handLandmarks)

            #         if handLabel == "Right":
            #             left_hand_angle = calculate_angle(handLandmarks[14], handLandmarks[15], handLandmarks[16])
            #             print('left', left_hand_angle)

            #             if left_hand_angle > 140:
            #                 press_key(Key.right)

            # if handLabel == "Right":
            #     right_hand_angle = calculate_angle(handLandmarks[14], handLandmarks[15], handLandmarks[16])

            #     if right_hand_angle > 80:
            #         press_key(Key.right)
            #     left_hand_fingers = left_hand_fingers+1

            # Thumb
            # if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
            #     left_hand_fingers = left_hand_fingers+1
            # # Index finger
            # if handLabel == "Left" and handLandmarks[8][0] < handLandmarks[6][0]:
            #     left_hand_fingers = left_hand_fingers+1
            # # Middle finger
            # if handLabel == "Left" and handLandmarks[12][0] < handLandmarks[10][0]:
            #     left_hand_fingers = left_hand_fingers+1
            # # Ring finger
            # if handLabel == "Left" and handLandmarks[16][0] < handLandmarks[14][0]:
            #     left_hand_fingers = left_hand_fingers+1
            # # Pinky
            # if handLabel == "Left" and handLandmarks[20][0] < handLandmarks[18][0]:
            #     left_hand_fingers = left_hand_fingers+1

            # if left_hand_fingers == 5:
            #     stage = 'right_head_tilt'
            #     press_key(Key.left)
            #     print('backward')

            # if handLabel == "Right" and handLandmarks[4][1] > handLandmarks[3][1]:
            #     right_hand_fingers = right_hand_fingers+1
            # # Index finger
            # if handLabel == "Right" and handLandmarks[8][1] < handLandmarks[6][1]:
            #     right_hand_fingers = right_hand_fingers+1
            # # Middle finger
            # if handLabel == "Right" and handLandmarks[12][1] < handLandmarks[10][1]:
            #     right_hand_fingers = right_hand_fingers+1
            # # Ring finger
            # if handLabel == "Right" and handLandmarks[16][1] < handLandmarks[14][1]:
            #     right_hand_fingers = right_hand_fingers+1
            # # Pinky
            # if handLabel == "Right" and handLandmarks[20][1] < handLandmarks[18][1]:
            #     right_hand_fingers = right_hand_fingers+1

            # if right_hand_fingers == 5:
            #     stage = 'left_head_tilt'
            #     press_key(Key.right)
            #     print('forward')

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


with col2:
    webrtc_streamer(
        key="streamer",
        video_frame_callback=transform,
        sendback_audio=False
    )
# <iframe
#     id="tekkenWindow"
# 			src=
# 			width="600"
# 			height="450"
# 			frameborder="no"
# 			allowfullscreen="true"
# 			webkitallowfullscreen="true"
# 			mozallowfullscreen="true"
# 			scrolling="no"
# 		></iframe>
