## Overview
**She Health AI** is an AI-powered tool designed to support the early detection of gynecological cancers among African women. The system leverages deep learning to analyze ultrasound images and symptoms, identifying potential cancer indicators for timely intervention and healthcare access. By focusing on underserved regions, this project aims to contribute to gender equality in healthcare, supporting UN Sustainable Development Goal 5.

## Features
- **Early Detection**: Uses AI to screen for potential gynecological cancer lesions.
- **User-Friendly Interface**: Upload ultrasound images through a web app to receive AI analysis.
- **Support for African Women Healthcare**: Targets healthcare inequalities by offering accessible detection tools.

## Installation and Setup

### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/yourusername/SheHealthAI.git
cd SheHealthAI
```

### Step 2: Create a Virtual Environment
Create and activate a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies
Install all necessary packages:
```bash
pip install -r requirements.txt
```

### Step 4: Configure the Application
The `config/config.yaml` file contains configurations for the model and dataset. Review it to customize model parameters if needed.

### Step 5: Prepare Data
Place ultrasound images in the `data/raw` directory. These images will be used for preprocessing and model training.

### Step 6: Preprocess the Data
Run the following script to resize and normalize the images:
```bash
python src/data_preprocessing.py
```

### Step 7: Train the Model
Execute the model training script to build the cancer detection model:
```bash
python src/model_training.py
```

### Step 8: Run the Flask Application
Start the Flask application:
```bash
python app/app.py
```

### Step 9: Access the Application
Open your web browser and go to:
```
http://127.0.0.1:5000
```
Upload an ultrasound image, and the system will display a prediction (e.g., "Cancer" or "No Cancer").

## Testing the Application
To test the model output:
1. Add images to the `data/raw` directory.
2. Run the preprocessing and model training steps as shown above.
3. Use the Flask app to upload images and verify the model predictions.

## Troubleshooting
- **Flask App Not Running**: Ensure all dependencies are installed, and your virtual environment is activated.
- **Data Errors**: Verify that the images are in the correct directory and match the
