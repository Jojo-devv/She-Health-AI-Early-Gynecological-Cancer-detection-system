from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('models/she_health_model.h5')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)  # Batch dimension
    img = img / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img_path = './uploads/' + file.filename
        file.save(img_path)
        img = preprocess_image(img_path)
        prediction = model.predict(img)
        result = np.argmax(prediction)
        return jsonify({"prediction": "Cancer" if result == 1 else "No Cancer"})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
