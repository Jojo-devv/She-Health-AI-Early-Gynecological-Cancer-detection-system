import os
import cv2
import numpy as np

raw_data_dir = 'data/raw'
processed_data_dir = 'data/processed'

# Create processed directory if it doesn't exist
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

for filename in os.listdir(raw_data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(raw_data_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(processed_data_dir, filename), image)
