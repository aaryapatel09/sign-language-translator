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

# Load the model and class names
model_path = os.path.join(os.path.dirname(__file__), 'model', 'asl_model.h5')
class_names_path = os.path.join(os.path.dirname(__file__), 'model', 'class_names.npy')

# Initialize variables
model = None
class_names = None

def create_model():
    """Create a new model with the same architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    return model

try:
    # Load class names
    if os.path.exists(class_names_path):
        class_names = np.load(class_names_path)
        logger.info(f"Successfully loaded {len(class_names)} class names")
    else:
        logger.error(f"Class names file not found at {class_names_path}")
        
    # Load the model if it exists
    if os.path.exists(model_path):
        try:
            # Create a new model with the same architecture
            model = create_model()
            # Load weights from the saved model
            model.load_weights(model_path)
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model created and weights loaded successfully")
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            model = None
    else:
        logger.error(f"Model file not found at {model_path}")
        
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())

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
        
        if model is None or class_names is None:
            logger.warning("Model not loaded. Running in test mode.")
            TEST_MODE = True
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