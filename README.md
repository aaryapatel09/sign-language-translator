# Real-Time Sign Language Translator

This application translates American Sign Language (ASL) gestures into text and speech in real-time using your computer's webcam.

## Features

- Real-time ASL gesture recognition using webcam
- Text output of recognized signs
- Text-to-speech conversion of recognized signs
- Support for basic ASL alphabet and common words

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python main.py
```

- Press 'q' to quit the application
- Make sure your hand is clearly visible in the camera frame
- Perform ASL signs in front of the camera

## Note

This is a basic implementation that recognizes ASL alphabet signs. The accuracy depends on lighting conditions and camera quality. 