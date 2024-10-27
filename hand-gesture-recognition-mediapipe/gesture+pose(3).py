import cv2 as cv
import mediapipe as mp
import numpy as np
import collections
import time
from minecraft import *  # Ensure this module is correctly installed and accessible
from app import *
# Import necessary modules from MediaPipe Tasks
from mediapipe.tasks.python import vision
from pynput.mouse import Controller


# Import the protobuf definitions for landmarks
from mediapipe.framework.formats import landmark_pb2

# Frame buffer size for movement analysis
FRAME_BUFFER_SIZE = 16
mouse_controller = Controller()


# Initialize a deque (double-ended queue) to store keypoints
keypoints_buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)

# Path to the pre-trained gesture recognition model
model_path = 'gesture_recognizer.task'  # Update this path if necessary

# Initialize last_escape for cooldowns (used in gesture actions)
last_escape = time.time()
action = ""
# Set up MediaPipe Gesture Recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

# Create GestureRecognizer with default options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO  # Set to VIDEO mode for continuous recognition
)

# Load OpenPose model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")  # Ensure 'graph_opt.pb' is in the correct path

# Define BODY_PARTS and POSE_PAIRS for OpenPose
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Threshold for confidence in OpenPose
thr = 0.2

# Input dimensions for OpenPose
inWidth = 368
inHeight = 368

def get_hand_coordinates(frame):
    """
    Use OpenPose to detect hand coordinates in the frame.
    Returns a list of (hand_label, (x, y)) tuples.
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # Prepare the frame for OpenPose
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]  # Get only the first 19 elements (BODY_PARTS)

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Find the global maximum of the heatmap.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if its confidence is higher than threshold.
        if conf > thr:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw the skeleton on the frame
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
            cv.circle(frame, points[idFrom], 3, (0, 0, 255), cv.FILLED)
            cv.circle(frame, points[idTo], 3, (0, 0, 255), cv.FILLED)

    # Get the hand coordinates from the points
    hands_coords = []
    # Right hand
    if points[BODY_PARTS["RWrist"]]:
        hands_coords.append(('Right', points[BODY_PARTS["RWrist"]]))
    # Left hand
    if points[BODY_PARTS["LWrist"]]:
        hands_coords.append(('Left', points[BODY_PARTS["LWrist"]]))
    # If no wrists detected, return empty list
    return hands_coords

def crop_hand_region(frame, hand_coords, box_size=224):
    """
    Crop the frame to include the hand region centered at hand_coords.

    :param frame: The original frame (numpy array).
    :param hand_coords: Tuple (x, y) of hand coordinates in pixels.
    :param box_size: The size of the square box to crop.
    :return: Cropped image of size (box_size, box_size), bounding box coordinates (x1, y1, x2, y2).
    """
    x, y = hand_coords
    h, w, _ = frame.shape
    half_size = box_size // 3

    # Calculate top-left and bottom-right coordinates
    x1 = int(max(0, x - half_size))
    y1 = int(max(0, y - half_size))
    x2 = int(min(w, x + half_size))
    y2 = int(min(h, y + half_size))

    # Adjust if the box is smaller than box_size (e.g., near edges)
    if x2 - x1 < box_size:
        if x1 == 0:
            x2 = int(min(w, x1 + box_size))
        elif x2 == w:
            x1 = int(max(0, x2 - box_size))
    if y2 - y1 < box_size:
        if y1 == 0:
            y2 = int(min(h, y1 + box_size))
        elif y2 == h:
            y1 = int(max(0, y2 - box_size))

    # Crop the image
    cropped = frame[y1:y2, x1:x2]

    # Resize cropped image to (box_size, box_size) if necessary
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        cropped = cv.resize(cropped, (box_size, box_size))
    else:
        # If cropping failed due to small size, use a black image
        cropped = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    return cropped, (x1, y1, x2, y2)

def calculate_movement(buffer):
    movement = []
    for i in range(1, len(buffer)):
        frame_movement = []
        for p1, p2 in zip(buffer[i - 1], buffer[i]):
            if p1 and p2:
                x_dist = p2[0] - p1[0]
                y_dist = p2[1] - p1[1]
                frame_movement.append((x_dist, y_dist))
            else:
                frame_movement.append((0, 0))
        movement.append(frame_movement)
    return movement

def identify(movement, keypoints_buffer, baseline_distance=None):
    # Define body parts indices
    neck_keypoint = BODY_PARTS["Neck"]
    right_knee = BODY_PARTS["RKnee"]
    left_knee = BODY_PARTS["LKnee"]
    right_hip = BODY_PARTS["RHip"]
    left_hip = BODY_PARTS["LHip"]
    right_shoulder = BODY_PARTS["RShoulder"]
    left_shoulder = BODY_PARTS["LShoulder"]

    num_frames = min(len(movement), len(keypoints_buffer))

    # Initialize baseline distances if not provided
    if baseline_distance is None:
        baseline_distance = {}
        # Establish baselines for shoulder-hip distance
        for idx in range(num_frames):
            frame_points = keypoints_buffer[idx]
            lhip = frame_points[left_hip]
            rhip = frame_points[right_hip]
            lshoulder = frame_points[left_shoulder]
            rshoulder = frame_points[right_shoulder]
            if lhip and rhip and lshoulder and rshoulder:
                avg_hip_y = (lhip[1] + rhip[1]) / 2
                avg_shoulder_y = (lshoulder[1] + rshoulder[1]) / 2
                # Baseline for standing height (shoulder to hip)
                baseline_distance['shoulder_hip'] = abs(avg_shoulder_y - avg_hip_y)
                break  # Baseline established
        if not baseline_distance:
            # Could not establish baseline, return default action
            return "Standing", None

    # Initialize movement lists
    left_leg_vertical_movement = []
    right_leg_vertical_movement = []
    leg_vertical_movement_difference = []
    vertical_movements = {
        'shoulder_hip_distance': [],
    }

    # Process each frame
    for idx in range(1, num_frames):
        prev_frame_points = keypoints_buffer[idx - 1]
        curr_frame_points = keypoints_buffer[idx]

        # Get keypoint positions
        # Left Leg
        prev_left_knee = prev_frame_points[left_knee]
        curr_left_knee = curr_frame_points[left_knee]

        # Right Leg
        prev_right_knee = prev_frame_points[right_knee]
        curr_right_knee = curr_frame_points[right_knee]

        # Calculate left leg vertical movement
        if prev_left_knee and curr_left_knee:
            left_leg_dy = prev_left_knee[1] - curr_left_knee[1]  # Positive if moving up
            left_leg_vertical_movement.append(left_leg_dy)
        else:
            left_leg_vertical_movement.append(0)

        # Calculate right leg vertical movement
        if prev_right_knee and curr_right_knee:
            right_leg_dy = prev_right_knee[1] - curr_right_knee[1]  # Positive if moving up
            right_leg_vertical_movement.append(right_leg_dy)
        else:
            right_leg_vertical_movement.append(0)

        # Calculate movement difference
        dy_difference = abs(left_leg_vertical_movement[-1] - right_leg_vertical_movement[-1])
        leg_vertical_movement_difference.append(dy_difference)

        # Vertical movements for crouching detection
        lshoulder = curr_frame_points[left_shoulder]
        rshoulder = curr_frame_points[right_shoulder]
        lhip = curr_frame_points[left_hip]
        rhip = curr_frame_points[right_hip]
        if lshoulder and rshoulder and lhip and rhip:
            avg_shoulder_y = (lshoulder[1] + rshoulder[1]) / 2
            avg_hip_y = (lhip[1] + rhip[1]) / 2
            shoulder_hip_distance = abs(avg_shoulder_y - avg_hip_y)
            vertical_movements['shoulder_hip_distance'].append(shoulder_hip_distance)
        else:
            vertical_movements['shoulder_hip_distance'].append(None)

    # --- Decision Logic ---
    action = "Standing"

    # Crouching Detection
    crouch_threshold = 10  # Percentage decrease threshold
    crouching_frames = 0
    for dist in vertical_movements['shoulder_hip_distance']:
        if dist is not None:
            percent_decrease = ((baseline_distance['shoulder_hip'] - dist) / baseline_distance['shoulder_hip']) * 100
            if percent_decrease > crouch_threshold:
                crouching_frames += 1
    if crouching_frames > num_frames / 2:
        action = "Crouching"

    # Walking Detection
    if action == "Standing":
        walk_vertical_threshold = 2  # Minimum vertical movement to consider
        alternating_movement_frames = 0
        for idx in range(len(left_leg_vertical_movement)):
            left_dy = left_leg_vertical_movement[idx]
            right_dy = right_leg_vertical_movement[idx]

            # Check for significant vertical movement in one leg
            if (left_dy > walk_vertical_threshold and right_dy < walk_vertical_threshold) or \
               (right_dy > walk_vertical_threshold and left_dy < walk_vertical_threshold):
                alternating_movement_frames += 1

        if alternating_movement_frames > num_frames / 4:
            action = "Walking"

    # Jumping Detection
    if action == "Standing":
        jump_vertical_threshold = 3  # Threshold for jumping
        jumping_frames = 0
        for left_dy, right_dy in zip(left_leg_vertical_movement, right_leg_vertical_movement):
            if left_dy > jump_vertical_threshold and right_dy > jump_vertical_threshold:
                jumping_frames += 1
        if jumping_frames > num_frames / 4:
            action = "Jumping"

    return action, baseline_distance

def get_pose_keypoints(frame):
    """
    Use OpenPose to detect keypoints in the frame.
    Returns a list of keypoint coordinates.
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # Prepare the frame for OpenPose
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]  # Get only the first 19 elements (BODY_PARTS)

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Find the global maximum of the heatmap.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if its confidence is higher than threshold.
        if conf > thr:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return points

with GestureRecognizer.create_from_options(options) as recognizer:
    # Start video capture
     # Argument parsing #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    cursor_smoothing_history = deque(maxlen=5)
    screen_width, screen_height = pyautogui.size()
    mode = 0
    frame_counter = 0

    cap = cv.VideoCapture(0)
    while True:
        fps = cvFpsCalc.get()
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # # Hand sign classification
                # hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                # if hand_sign_id == 2:  # Point gesture
                #     point_history.append(landmark_list[8])
                # else:
                #     point_history.append([0, 0])

                # # Finger gesture classification
                # finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list)
                # if point_history_len == (history_length * 2):
                #     finger_gesture_id = point_history_classifier(
                #         pre_processed_point_history_list)
                #

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Assuming '2' corresponds to the pointing gesture
                    point_history.append(landmark_list[8])  # Index finger tip
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Update gesture history
                finger_gesture_history.append(finger_gesture_id)

                # **Call the function to get pointer finger info**
                pointer_coords, classification = get_pointer_finger_info(
                    landmark_list, hand_sign_id, finger_gesture_history, point_history_classifier_labels
                )

                # **Use the pointer_coords and classification as needed**
                if pointer_coords is not None:
                    print(f"Pointer Finger Coordinates: {pointer_coords}")
                    print(f"Gesture Classification: {classification}")
                    cv.putText(debug_image,
                               f"Coordinates: {pointer_coords}, Classification {classification}",
                               (pointer_coords[0] + 10, pointer_coords[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                               cv.LINE_AA)

                    # Map the camera coordinates to screen coordinates
                    screen_x, screen_y = map_coordinates(
                        pointer_coords[0],
                        pointer_coords[1],
                        cap_width,
                        cap_height,
                        screen_width,
                        screen_height
                    )
                    if classification == "Clockwise":
                        if time.time() - last_escape > 2:
                            pyautogui.scroll(1)
                            last_escape = time.time()
                    elif classification == "Counter Clockwise":
                        if time.time() - last_escape > 2:
                            pyautogui.scroll(-1)
                            last_escape = time.time()
                    elif classification == "Stop":
                        if time.time() - last_escape > 2:
                            pyautogui.click(button='left')
                            last_escape = time.time()

                    cursor_smoothing_history.append((screen_x, screen_y))

                    # Calculate the average position for smoothing
                    if classification == "Move":
                        avg_x = int(np.mean([pos[0] for pos in cursor_smoothing_history]))
                        avg_y = int(np.mean([pos[1] for pos in cursor_smoothing_history]))
                        mouse_controller.position = (avg_x, avg_y)
                        

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv.flip(frame, 1)

        # Get pose keypoints from OpenPose
        keypoints = get_pose_keypoints(frame)
        keypoints_buffer.append(keypoints)

        # Calculate movement and identify action if buffer is full
        if len(keypoints_buffer) == FRAME_BUFFER_SIZE:
            movement = calculate_movement(keypoints_buffer)
            action, baseline_distance = identify(movement, keypoints_buffer)
            # Print the action classification
            print(f"Action: {action}")

        # Get hand coordinates from OpenPose
        hands_coords = get_hand_coordinates(frame)

        # Initialize a list to store windows to display
        windows_to_display = []

        # If no hands detected, use the whole frame
        if not hands_coords:
            # Set the bounding box to the whole frame
            h, w, _ = frame.shape
            x1, y1, x2, y2 = 0, 0, w, h
            hand_label = 'No Hand Detected'

            # Use the whole frame
            cropped_frame = frame.copy()

            # Convert the BGR frame to RGB
            rgb_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)

            # Prepare the input image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate timestamp in microseconds
            timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC) * 1000)

            # Run gesture recognition synchronously for video
            result = recognizer.recognize_for_video(mp_image, timestamp)

            # Process the results and draw on the cropped frame
            # Convert the mp.Image to a numpy array for OpenCV
            output_frame = mp_image.numpy_view()
            output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

            # Draw hand landmarks and recognized gestures
            if result.hand_landmarks:
                for i in range(len(result.hand_landmarks)):
                    hand_landmarks = result.hand_landmarks[i]
                    # Convert landmarks to MediaPipe format
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    for lm in hand_landmarks:
                        landmark = landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        hand_landmarks_proto.landmark.append(landmark)

                    # Draw the hand landmarks on the cropped frame
                    mp.solutions.drawing_utils.draw_landmarks(
                        output_frame,
                        hand_landmarks_proto,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

                    # Get the recognized gesture
                    if result.gestures and result.gestures[i]:
                        gesture = result.gestures[i][0]
                        gesture_name = gesture.category_name
                        score = gesture.score

                        # Display the gesture on the cropped frame
                        cv.putText(output_frame, f"{gesture_name} ({score:.2f})",
                                   (10, 30 + i * 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Print the current gesture and hand label along with action
                        if gesture_name == "Open_Palm" and score > 0.6:
                            place_block()
                        if gesture_name == "Closed_Fist" and score > 0.6:
                            attack()
                        if gesture_name == "Thumb_Up" and score > 0.6:
                            if time.time() - last_escape > 2:
                                go_inventory()
                                last_escape = time.time()
                        if gesture_name == "Thumb_Down" and score > 0.6:
                            if time.time() - last_escape > 2:
                                leave_inventory()
                                last_escape = time.time()
                        print(f"Hand: {hand_label}, Gesture: {gesture_name}, Score: {score:.2f}, Action: {action}")

            else:
                # If no hand landmarks detected, display a message
                cv.putText(output_frame, "No Hand Detected",
                           (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Print no hand detected with action
                if result.gestures and result.gestures[i]:
                    gesture = result.gestures[i][0]
                    gesture_name = gesture.category_name
                    score = gesture.score

                    # Display the gesture on the cropped frame
                    cv.putText(output_frame, f"{gesture_name} ({score:.2f})",
                                (10, 30 + i * 30),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gesture_name == "Open_Palm" and score > 0.6:
                        place_block()
                    if gesture_name == "Closed_Fist" and score > 0.6:
                        attack()
                    if gesture_name == "Thumb_Up" and score > 0.6:
                        if time.time() - last_escape > 2:
                            go_inventory()
                            last_escape = time.time()
                    if gesture_name == "Thumb_Down" and score > 0.6:
                        if time.time() - last_escape > 2:
                            leave_inventory()
                            last_escape = time.time()

                print(f"Hand: {hand_label}, Gesture: None, Action: {action}")

            # Draw the bounding box on the original frame
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add windows to display list
            windows_to_display.append(("Cropped Hand - Live", output_frame))

        else:
            # Process each hand separately
            for hand_label, (x, y) in hands_coords:
                # Crop the hand region
                cropped_frame, (x1, y1, x2, y2) = crop_hand_region(frame, (x, y))

                # Convert the BGR frame to RGB
                rgb_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)

                # Prepare the input image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Calculate timestamp in microseconds
                timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC) * 1000)

                # Run gesture recognition synchronously for video
                result = recognizer.recognize_for_video(mp_image, timestamp)

                # Process the results and draw on the cropped frame
                # Convert the mp.Image to a numpy array for OpenCV
                output_frame = mp_image.numpy_view()
                output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)

                # Draw hand landmarks and recognized gestures
                if result.hand_landmarks:
                    for i in range(len(result.hand_landmarks)):
                        hand_landmarks = result.hand_landmarks[i]
                        # Convert landmarks to MediaPipe format
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        for lm in hand_landmarks:
                            landmark = landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                            hand_landmarks_proto.landmark.append(landmark)

                        # Draw the hand landmarks on the cropped frame
                        mp.solutions.drawing_utils.draw_landmarks(
                            output_frame,
                            hand_landmarks_proto,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style())

                        # Get the recognized gesture
                        if result.gestures and result.gestures[i]:
                            gesture = result.gestures[i][0]
                            gesture_name = gesture.category_name
                            score = gesture.score
                            # Display the gesture on the cropped frame
                            cv.putText(output_frame, f"{gesture_name} ({score:.2f})",
                                       (10, 30 + i * 30),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Print the current gesture, hand label, and action
                            print(f"Hand: {hand_label}, Gesture: {gesture_name}, Score: {score:.2f}, Action: {action}")

                            # Example Integration with Minecraft (Uncomment and define functions as needed)
                            if gesture_name == "Open_Palm" and score > 0.6:
                                place_block()
                            if gesture_name == "Closed_Fist" and score > 0.6:
                                attack()
                            if gesture_name == "Thumb_Up" and score > 0.6:
                                if time.time() - last_escape > 2:
                                    go_inventory()
                                    last_escape = time.time()
                            if gesture_name == "Thumb_Down" and score > 0.6:
                                if time.time() - last_escape > 2:
                                    leave_inventory()
                                    last_escape = time.time()

                else:
                    # If no hand landmarks detected, display a message
                    cv.putText(output_frame, "Hand Not Detected",
                               (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Print hand not detected with action
                    print(f"Hand: {hand_label}, Gesture: None, Action: {action}")
                # Get the recognized gesture
                    if result.gestures and result.gestures[i]:
                        gesture = result.gestures[i][0]
                        gesture_name = gesture.category_name
                        score = gesture.score
                        # Display the gesture on the cropped frame
                        cv.putText(output_frame, f"{gesture_name} ({score:.2f})",
                                    (10, 30 + i * 30),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Print the current gesture, hand label, and action
                        print(f"Hand: {hand_label}, Gesture: {gesture_name}, Score: {score:.2f}, Action: {action}")

                        # Example Integration with Minecraft (Uncomment and define functions as needed)
                        if gesture_name == "Open_Palm" and score > 0.6:
                            place_block()
                        if gesture_name == "Closed_Fist" and score > 0.6:
                            attack()
                        if gesture_name == "Thumb_Up" and score > 0.6:
                            if time.time() - last_escape > 2:
                                go_inventory()
                                last_escape = time.time()
                        if gesture_name == "Thumb_Down" and score > 0.6:
                            if time.time() - last_escape > 2:
                                leave_inventory()
                                last_escape = time.time()

                # Draw the bounding box on the original frame
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Label the hand (Left or Right)
                cv.putText(frame, f"{hand_label} Hand",
                           (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Add the output frame to windows to display
                windows_to_display.append((f"Cropped {hand_label} Hand - Live", output_frame))

                # Get the recognized gesture
                if result.gestures and result.gestures[i]:
                    gesture = result.gestures[i][0]
                    gesture_name = gesture.category_name
                    score = gesture.score
                    # Display the gesture on the cropped frame
                    cv.putText(output_frame, f"{gesture_name} ({score:.2f})",
                                (10, 30 + i * 30),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Print the current gesture, hand label, and action
                    print(f"Hand: {hand_label}, Gesture: {gesture_name}, Score: {score:.2f}, Action: {action}")

                    # Example Integration with Minecraft (Uncomment and define functions as needed)
                    if gesture_name == "Open_Palm" and score > 0.6:
                        place_block()
                    if gesture_name == "Closed_Fist" and score > 0.6:
                        attack()
                    if gesture_name == "Thumb_Up" and score > 0.6:
                        if time.time() - last_escape > 2:
                            go_inventory()
                            last_escape = time.time()
                    if gesture_name == "Thumb_Down" and score > 0.6:
                        if time.time() - last_escape > 2:
                            leave_inventory()
                            last_escape = time.time()

        # Display the original frame with bounding boxes and skeleton
        cv.imshow("Hand Gesture Recognition", frame)

        # Display all cropped frames with annotations (live feedback)
        for window_name, output_frame in windows_to_display:
            cv.imshow(window_name, output_frame)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()