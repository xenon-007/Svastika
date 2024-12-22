from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    hor_dist = np.linalg.norm(np.array(p1) - np.array(p4))
    ver_dist_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    ver_dist_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    ear = (ver_dist_1 + ver_dist_2) / (2.0 * hor_dist)
    return ear

# Function to calculate head tilt (vertical and horizontal)
def calculate_head_tilt(landmarks):
    left_ear = np.array(landmarks[234])  # Approx. left side of face
    right_ear = np.array(landmarks[454])  # Approx. right side of face
    nose_tip = np.array(landmarks[1])

    # Calculate the horizontal tilt using the nose and ear alignment
    horizontal_tilt = np.abs(left_ear[1] - right_ear[1])  # Compare y-coordinates

    # Calculate the lateral tilt (left/right) using x-coordinates
    lateral_tilt = np.abs(left_ear[0] - right_ear[0])

    return horizontal_tilt, lateral_tilt

# Function to calculate shoulder alignment (proxy using face landmarks)
def calculate_shoulder_alignment(landmarks):
    chin = np.array(landmarks[152])  # Chin point
    nose_tip = np.array(landmarks[1])  # Nose tip

    # Compare z-coordinates to determine leaning forward or backward
    forward_lean = chin[2] - nose_tip[2]
    return forward_lean

# Define indices for EAR calculation
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.3  # Eye closed if EAR < 0.3
TILT_THRESHOLD = 0.3  # Head tilt threshold (vertical)
LATERAL_TILT_THRESHOLD = 0.3  # Head tilt threshold (left/right)
LEAN_THRESHOLD = 0.15  # Forward/backward lean threshold

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    frame_file = request.files.get('frame')
    if not frame_file:
        return jsonify({"error": "No frame received"}), 400

    # Decode image to analyze
    img_array = np.frombuffer(frame_file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))

        # Eye aspect ratio calculation
        left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        # Head tilt calculation
        vertical_tilt, lateral_tilt = calculate_head_tilt(landmarks)

        # Shoulder alignment calculation (using chin and nose alignment as proxy)
        shoulder_alignment = calculate_shoulder_alignment(landmarks)

        # Posture evaluation
        if avg_ear < EAR_THRESHOLD:
            posture_status = "Bad Posture (Eyes Closed)"
        elif vertical_tilt > TILT_THRESHOLD:
            posture_status = "Bad Posture (Head Tilted Vertically)"
        elif lateral_tilt > LATERAL_TILT_THRESHOLD:
            posture_status = "Bad Posture (Head Tilted Laterally)"
        elif abs(shoulder_alignment) > LEAN_THRESHOLD:
            posture_status = "Bad Posture (Leaning)"
        else:
            posture_status = "Good Posture"

        return jsonify({"posture_status": posture_status})

    # No face detected
    return jsonify({"posture_status": "Bad Posture (No Face Detected)"})

