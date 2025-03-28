import os
import requests
import zipfile
import io

def download_dataset():
    # URL of the ASL Alphabet dataset
    url = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/download"
    
    print("Please download the ASL Alphabet dataset manually from:")
    print("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("\nAfter downloading:")
    print("1. Extract the zip file")
    print("2. Move the 'asl_alphabet_train' folder contents to the 'dataset' directory")
    print("3. Rename the folders to match the letters (A, B, C, etc.)")
    print("\nThe dataset should be organized like this:")
    print("dataset/")
    print("  A/")
    print("    image1.jpg")
    print("    image2.jpg")
    print("    ...")
    print("  B/")
    print("    image1.jpg")
    print("    ...")
    print("\nOnce you've done this, run 'python train_model.py' to train the model.")

if __name__ == "__main__":
    download_dataset() 