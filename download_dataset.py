import os
import requests
import zipfile
import io
import shutil

def download_dataset():
    print("Downloading ASL Alphabet dataset...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    # Create sample directories for testing
    for i in range(10):
        os.makedirs(f'dataset/{i}', exist_ok=True)
        # Create a sample image (blank for now)
        with open(f'dataset/{i}/sample.jpg', 'wb') as f:
            f.write(b'')  # Empty file for testing
    
    print("Created dataset structure for testing.")
    print("\nFor a complete dataset, please:")
    print("1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("2. Download the dataset")
    print("3. Extract the zip file")
    print("4. Move the contents of 'asl_alphabet_train' to your project's 'dataset' folder")
    print("\nThe dataset should be organized like this:")
    print("dataset/")
    print("  A/")
    print("    image1.jpg")
    print("    image2.jpg")
    print("    ...")
    print("  B/")
    print("    image1.jpg")
    print("    ...")

if __name__ == "__main__":
    download_dataset() 