import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(29, activation='softmax')  # 26 letters + space + nothing + del
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def load_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0  # Normalize
            
            images.append(img)
            labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels)

def train_model():
    # Load the ASL Alphabet dataset
    data_dir = 'dataset'  # You'll need to download and extract the dataset here
    X, y = load_data(data_dir)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = create_model()
    
    # Add data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
    ])
    
    # Train the model
    history = model.fit(
        data_augmentation(X_train),
        y_train,
        epochs=20,
        validation_data=(X_test, y_test)
    )
    
    # Save the model
    model.save('model/asl_model.h5')
    
    # Save class names
    class_names = sorted(os.listdir(data_dir))
    np.save('model/class_names.npy', class_names)
    
    return model, history

if __name__ == '__main__':
    train_model() 