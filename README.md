# ASL Translator — browser-only

Real-time American Sign Language letter detection that runs **entirely in your browser**. No Python server, no install, no data leaves the page.

- Hand detection: [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) `hand_landmarker` (float16)
- Letter classification: rule-based on the 21 hand landmarks — covers a useful subset of the ASL alphabet (see below)
- Speech output: browser [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis)

## Run

Because the page uses the webcam, it needs to be served over `http://localhost` or HTTPS. Anything that serves static files works:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Click **Start camera**, allow webcam access, and sign. The detected letter auto-appends to the buffer when held for one second.

## Supported letters

**A · B · C · D · E · F · I · L · O · U · V · W · Y** — 13 static letters — plus an open-palm **Hello** sign.

Letters excluded by design:
- **J**, **Z** — require motion, not a single-frame pose
- **M**, **N**, **S**, **T** — differ only by which fingers tuck behind the thumb; landmark positions are nearly identical in 2D
- **G**, **H**, **K**, **P**, **Q**, **R**, **X** — fine orientation differences the rule set doesn't separate cleanly

For those, the UI shows `?` instead of guessing. A neural classifier trained on landmark sequences would be the right replacement.

## Why no server

The previous version had a Flask backend that loaded a TensorFlow model. The model architecture in `train_model.py` (29-class CNN on 64×64 RGB images) did not match the inference path in `main.py` (26-class landmark model) or in `app.py` (26-class CNN on raw frames) — so no trained artifact was compatible with any of the entry points. Moving to a pure-browser pipeline makes the app runnable in one click on any device with a camera.

## Privacy

The MediaPipe model assets are fetched from Google's CDN on first load. After that, video frames never leave your device.
