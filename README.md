# Sign Language Translator

A real-time American Sign Language (ASL) translator that uses computer vision to detect and translate hand gestures into text.

## Features

- Real-time hand gesture detection using MediaPipe
- Support for basic ASL letters (A, B, D, I, L, V, W, Y, 5)
- Live camera feed with visual feedback
- User-friendly web interface
- Detailed error handling and logging

## Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- Flask
- Modern web browser with camera access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaryapatel09/sign-language-translator.git
cd sign-language-translator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open `index.html` in your web browser

3. Click "Start Camera" and allow camera access when prompted

4. Position your hand in the guide box and make ASL signs

5. The detected sign will appear on screen

## Supported Signs

Currently supports the following ASL letters:
- A (closed fist)
- B (all fingers up)
- D (index finger up)
- I (index and pinky up)
- L (thumb and index finger up)
- V (index and middle fingers up)
- W (index, middle, and ring fingers up)
- Y (thumb and pinky up)
- 5 (all fingers up)

## Troubleshooting

If you encounter any issues:
1. Make sure the Flask server is running
2. Check that your camera is working and accessible
3. Ensure you're using a modern browser
4. Look for error messages in the browser console (F12)
5. Check the Flask server logs for detailed error information

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 