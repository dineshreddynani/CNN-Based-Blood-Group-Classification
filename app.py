from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('model/blood_group_model.h5', compile=False)

# Class labels for blood group prediction
class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Mapping corrections for both trained and real-time images
correction_mapping = {
    'AB+': 'B+',
    'AB-': 'B-',
    'B+': 'AB+',
    'B-': 'AB-'
}

def preprocess_image(image, input_type):
    if input_type == 'trained':
        img = image.convert('RGB')
        img = img.resize((64, 64))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    elif input_type == 'real_time':
        img = image.convert('L')  # Convert to grayscale
        img = img.resize((64, 64), Image.BILINEAR)  # Resize
        img_array = img_to_array(img)

        # Histogram equalization
        img_array = cv2.equalizeHist(img_array.astype(np.uint8))

        # Convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    input_type = request.form['input_type']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image, input_type)

        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Apply correction for both types
        if predicted_class in correction_mapping:
            predicted_class = correction_mapping[predicted_class]

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run( )
