from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
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

# For testing, return a random sign
def get_test_sign():
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
        
        # For testing, return a random sign
        detected_sign = get_test_sign()
        logger.info(f"Test mode: Detected sign: {detected_sign}")
        return jsonify({'sign': detected_sign})

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 