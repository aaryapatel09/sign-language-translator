from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import logging
import traceback
import os

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the trained model and class names
model_path = 'model/asl_model.h5'
class_names_path = 'model/class_names.npy'

if not os.path.exists(model_path) or not os.path.exists(class_names_path):
    logger.warning("Model files not found. Running in test mode.")
    TEST_MODE = True
else:
    TEST_MODE = False
    model = tf.keras.models.load_model(model_path)
    class_names = np.load(class_names_path)
    logger.info(f"Loaded model with {len(class_names)} classes")

def preprocess_image(image):
    """Preprocess the image for model input."""
    # Resize to 64x64
    image = cv2.resize(image, (64, 64))
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # Normalize
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def get_test_sign():
    """Return a random sign for testing."""
    import random
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return random.choice(signs)

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
        logger.info(f"Received image of size: {len(image_bytes)} bytes")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400

        logger.info(f"Decoded image shape: {frame.shape}")
        
        if TEST_MODE:
            # For testing, return a random sign
            detected_sign = get_test_sign()
            logger.info(f"Test mode: Detected sign: {detected_sign}")
            return jsonify({'sign': detected_sign})
        else:
            # Preprocess the image
            processed_image = preprocess_image(frame)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            if confidence > 0.7:  # Confidence threshold
                detected_sign = class_names[predicted_class]
                logger.info(f"Detected sign: {detected_sign} with confidence: {confidence:.2f}")
                return jsonify({'sign': detected_sign})
            else:
                logger.info(f"No confident sign detected. Best match: {class_names[predicted_class]} with confidence: {confidence:.2f}")
                return jsonify({'sign': '-'})

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 