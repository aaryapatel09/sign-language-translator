import os
import requests
import zipfile
import io
import shutil
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    print("Downloading ASL Alphabet dataset...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        print("Downloading from Kaggle...")
        api.dataset_download_files(
            'grassknoted/asl-alphabet',
            path='dataset',
            unzip=True
        )
        
        # Move files from asl_alphabet_train to dataset
        if os.path.exists('dataset/asl_alphabet_train'):
            for item in os.listdir('dataset/asl_alphabet_train'):
                src = os.path.join('dataset/asl_alphabet_train', item)
                dst = os.path.join('dataset', item)
                if os.path.isdir(src):
                    shutil.move(src, dst)
            shutil.rmtree('dataset/asl_alphabet_train')
        
        print("Dataset downloaded and organized successfully!")
        print("\nNow you can run 'python train_model.py' to train the model.")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nAlternative method:")
        print("1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("2. Click 'Download'")
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