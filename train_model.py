import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def create_model(input_shape):
    """Create a CNN model for ASL recognition."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(26, activation='softmax')  # 26 classes for ASL alphabet
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Model parameters
    input_shape = (64, 64, 3)  # Adjust based on your input data
    model = create_model(input_shape)
    
    # Save the model architecture
    model.save('model/asl_model.h5')
    print("Model saved to model/asl_model.h5")
    
    # Note: This is a template for the model architecture.
    # To actually train the model, you would need:
    # 1. A dataset of ASL images
    # 2. Data preprocessing and augmentation
    # 3. Training loop with validation
    # 4. Model evaluation and fine-tuning
    
    print("\nNote: This is a template model. To train the model:")
    print("1. Collect or download an ASL dataset")
    print("2. Preprocess the data")
    print("3. Train the model using model.fit()")
    print("4. Evaluate and fine-tune as needed")

if __name__ == "__main__":
    main() 