# American Sign Language (ASL) Translator

A real-time ASL translator that uses computer vision and deep learning to detect and translate American Sign Language signs into text.

## Features

- Real-time sign language detection using webcam
- Support for ASL alphabet (A-Z)
- High accuracy with confidence threshold
- User-friendly web interface
- Cross-platform compatibility

## Prerequisites

- Python 3.9 or higher
- Webcam
- Modern web browser

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

2. Open your web browser and navigate to:
```
http://localhost:5001
```

3. Allow camera access when prompted
4. Show ASL signs to your webcam
5. The detected sign will be displayed on the screen

## Project Structure

```
sign-language-translator/
├── app.py              # Flask application
├── model/             # Model directory
│   ├── asl_model.h5   # Trained model
│   └── class_names.npy # Class labels
├── static/            # Static files
│   ├── css/          # Stylesheets
│   └── js/           # JavaScript files
├── templates/         # HTML templates
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Technical Details

- Built with Flask for the backend
- Uses TensorFlow for sign detection
- OpenCV for image processing
- Real-time webcam feed processing
- Responsive web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ASL Alphabet dataset from Kaggle
- TensorFlow and OpenCV communities
- Flask framework 