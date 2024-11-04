import os
import numpy as np
import cv2

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0,1]
    return img

def process_dataset(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        processed_img = load_and_preprocess_image(img_path)
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, processed_img)
