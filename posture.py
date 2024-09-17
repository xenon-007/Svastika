import cv2
import mediapipe as mp
import time

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
mp_pose = mp.solutions.pose

# Function to check posture
def check_posture(landmarks):
    # Define indices for relevant landmarks
    left_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
    right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

    left_shoulder = landmarks[left_shoulder_idx]
    right_shoulder = landmarks[right_shoulder_idx]
    left_hip = landmarks[left_hip_idx]
    right_hip = landmarks[right_hip_idx]

    # Calculate the vertical alignment of shoulders and hips
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    hip_diff = abs(left_hip.y - right_hip.y)

    # Check if shoulders and hips are roughly aligned
    if shoulder_diff < 0.05 and hip_diff < 0.05:
        return "Good posture"
    else:
        return "Bad posture: Misaligned shoulders or hips"

# Initialize variables to track productivity time
good_posture_start_time = None
good_posture_duration = 0
last_posture = None

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Convert the image to RGB for processing
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect objects and pose landmarks
        objectron_results = objectron.process(image)
        pose_results = pose.process(image)

        # Draw object landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if objectron_results.detected_objects:
            for detected_object in objectron_results.detected_objects:
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
        
        # Draw pose landmarks on the image and check posture
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            posture_status = check_posture(pose_results.pose_landmarks.landmark)
            
            # Update productivity time based on posture status
            if posture_status == "Good posture":
                if last_posture != "Good posture":
                    good_posture_start_time = time.time()  # Start tracking good posture
                else:
                    good_posture_duration += time.time() - good_posture_start_time  # Update duration
                    good_posture_start_time = time.time()  # Reset start time
            last_posture = posture_status
            
        # Flip the image horizontally for a selfie-view display
        flipped_image = cv2.flip(image, 1)

        # Display posture status and productivity time on the flipped image
        cv2.putText(flipped_image, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        productivity_time = f"Productive time: {int(good_posture_duration)} sec"
        cv2.putText(flipped_image, productivity_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the flipped image
        cv2.imshow('Body tracker', flipped_image)
        
        # Add a wait key to process the display window events
        if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'Q' to exit
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
