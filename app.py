from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Define basic ASL gestures based on finger states
def get_finger_states(hand_landmarks):
    # Get y coordinates of finger tips and middle joints
    thumb_tip = hand_landmarks.landmark[4].y
    thumb_ip = hand_landmarks.landmark[3].y
    index_tip = hand_landmarks.landmark[8].y
    index_pip = hand_landmarks.landmark[6].y
    middle_tip = hand_landmarks.landmark[12].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_tip = hand_landmarks.landmark[16].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_pip = hand_landmarks.landmark[18].y

    # Check if each finger is extended (tip is higher than pip)
    thumb_up = thumb_tip < thumb_ip
    index_up = index_tip < index_pip
    middle_up = middle_tip < middle_pip
    ring_up = ring_tip < ring_pip
    pinky_up = pinky_tip < pinky_pip

    finger_states = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    logger.info(f"Finger states: {finger_states}")
    return finger_states

def detect_asl_sign(finger_states):
    # Basic ASL letter detection based on finger states
    if finger_states == [False, True, False, False, False]:
        return "D"
    elif finger_states == [False, True, True, False, False]:
        return "V"
    elif finger_states == [False, True, True, True, False]:
        return "W"
    elif finger_states == [True, True, False, False, False]:
        return "L"
    elif finger_states == [False, True, True, True, True]:
        return "B"
    elif finger_states == [True, False, False, False, True]:
        return "Y"
    elif all(finger_states):
        return "5"
    elif not any(finger_states):
        return "A"
    elif finger_states == [False, True, False, False, True]:
        return "I"
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'image' not in request.files:
        logger.error("No image provided in request")
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read the image from the request
        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            logger.info("Hand detected in frame")
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            finger_states = get_finger_states(hand_landmarks)
            detected_sign = detect_asl_sign(finger_states)
            
            if detected_sign:
                logger.info(f"Detected sign: {detected_sign}")
                return jsonify({'sign': detected_sign})
            else:
                logger.info("No matching sign detected")
        else:
            logger.info("No hand detected in frame")
        
        return jsonify({'sign': '-'})

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 