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

logger.info(f"Looking for model at: {model_path}")
logger.info(f"Looking for class names at: {class_names_path}")

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
        logger.info(f"Successfully loaded {len(class_names)} class names: {class_names}")
    else:
        logger.error(f"Class names file not found at {class_names_path}")
        
    # Load the model if it exists
    if os.path.exists(model_path):
        try:
            # Create a new model with the same architecture
            model = create_model()
            logger.info("Created new model with architecture")
            
            # Load weights from the saved model
            model.load_weights(model_path)
            logger.info("Loaded model weights successfully")
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model compiled successfully")
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            logger.error(traceback.format_exc())
            model = None
    else:
        logger.error(f"Model file not found at {model_path}")
        
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    try:
        logger.info(f"Input image shape: {image.shape}")
        
        # Resize image to 64x64
        image = cv2.resize(image, (64, 64))
        logger.info(f"Resized image shape: {image.shape}")
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            logger.info("Converted grayscale to RGB")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            logger.info("Converted RGBA to RGB")
            
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        logger.info(f"Normalized image. Value range: [{np.min(image)}, {np.max(image)}]")
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        logger.info(f"Final preprocessed image shape: {image.shape}")
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

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
    if model is None or class_names is None:
        logger.error("Model or class names not loaded")
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        # Get the image from the request
        file = request.files['image']
        logger.info(f"Received image file: {file.filename}")
        
        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 500
            
        logger.info(f"Decoded image shape: {image.shape}")
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # Make prediction
        predictions = model.predict(processed_image)
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Raw predictions: {predictions}")
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        logger.info(f"Predicted class index: {predicted_class_idx}")
        logger.info(f"Confidence: {confidence:.2f}")
        
        # Only return prediction if confidence is above threshold
        if confidence > 0.5:  # 50% confidence threshold
            predicted_sign = class_names[predicted_class_idx]
            logger.info(f"Detected sign: {predicted_sign} with confidence: {confidence:.2f}")
            return jsonify({
                'sign': predicted_sign,
                'confidence': confidence
            })
        else:
            logger.info(f"Low confidence prediction: {confidence:.2f}")
            return jsonify({'sign': '-'})
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 