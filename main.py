import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from tensorflow.keras.models import load_model

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

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

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Load the pre-trained model (you'll need to train this or use a pre-trained model)
    try:
        model = load_model('model/asl_model.h5')
    except:
        print("Warning: Model file not found. Please train the model first.")
        return

    last_prediction = None
    last_speech_time = 0
    speech_cooldown = 2.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Preprocess landmarks and make prediction
                landmarks = preprocess_landmarks(hand_landmarks)
                prediction = model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction[0])
                
                # Get the predicted letter
                predicted_letter = ASL_ALPHABET.get(predicted_class, 'Unknown')
                
                # Display the prediction
                cv2.putText(frame, f"Sign: {predicted_letter}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Text-to-speech with cooldown
                current_time = time.time()
                if predicted_letter != last_prediction and (current_time - last_speech_time) > speech_cooldown:
                    engine.say(predicted_letter)
                    engine.runAndWait()
                    last_speech_time = current_time
                    last_prediction = predicted_letter

        # Display the frame
        cv2.imshow('ASL Translator', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 