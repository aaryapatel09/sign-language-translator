from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load the model
try:
    model = load_model('model/asl_model.h5')
except:
    print("Warning: Model file not found. Please train the model first.")
    model = None

# ASL alphabet mapping
ASL_ALPHABET = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def preprocess_landmarks(hand_landmarks):
    """Convert hand landmarks to a format suitable for the model."""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image from the request
    file = request.files['image']
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks and model is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocess landmarks and make prediction
            landmarks = preprocess_landmarks(hand_landmarks)
            prediction = model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Get the predicted letter
            predicted_letter = ASL_ALPHABET.get(predicted_class, 'Unknown')
            return jsonify({'sign': predicted_letter})
    
    return jsonify({'sign': '-'})

if __name__ == '__main__':
    app.run(debug=True) 