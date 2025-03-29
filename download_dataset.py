import os
import requests
import zipfile
import io
import shutil

def download_dataset():
    print("Downloading ASL Alphabet dataset...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    # URLs for the ASL dataset (split into multiple parts for reliability)
    urls = [
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/0/0_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/1/1_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/2/2_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/3/3_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/4/4_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/5/5_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/6/6_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/7/7_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/8/8_1.jpg",
        "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Dataset/9/9_1.jpg"
    ]
    
    try:
        # Create directories for each digit
        for i in range(10):
            os.makedirs(f'dataset/{i}', exist_ok=True)
        
        # Download images
        for i, url in enumerate(urls):
            print(f"Downloading sample for digit {i}...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the image
            with open(f'dataset/{i}/sample.jpg', 'wb') as f:
                f.write(response.content)
        
        print("Dataset downloaded successfully!")
        print("\nNow you can run 'python train_model.py' to train the model.")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nAlternative method:")
        print("1. Visit: https://github.com/ardamavi/Sign-Language-Digits-Dataset")
        print("2. Click 'Code' and 'Download ZIP'")
        print("3. Extract the downloaded zip file")
        print("4. Move the contents of the 'Dataset' folder to your project's 'dataset' folder")
        print("\nThe dataset should be organized like this:")
        print("dataset/")
        print("  0/")
        print("    sample.jpg")
        print("  1/")
        print("    sample.jpg")
        print("  ...")

if __name__ == "__main__":
    download_dataset() 