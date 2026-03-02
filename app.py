from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import os

app = Flask(__name__)

# -------------------------------
# Recreate Model Architecture
# -------------------------------

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model


# Initialize model
model = create_model()

# Load only weights (bypasses serialization issues)
model.load_weights('model/blood_group_model.h5')

print("Model loaded successfully!")

# -------------------------------
# Class Labels
# -------------------------------

class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

correction_mapping = {
    'AB+': 'B+',
    'AB-': 'B-',
    'B+': 'AB+',
    'B-': 'AB-'
}

# -------------------------------
# Image Preprocessing
# -------------------------------

def preprocess_image(image, input_type):

    if input_type == 'trained':
        img = image.convert('RGB')
        img = img.resize((64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    elif input_type == 'real_time':
        img = image.convert('L')
        img = img.resize((64, 64), Image.BILINEAR)
        img_array = img_to_array(img)

        img_array = cv2.equalizeHist(img_array.astype(np.uint8))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    return None


# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    input_type = request.form.get('input_type')

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image, input_type)

        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Apply correction mapping
        if predicted_class in correction_mapping:
            predicted_class = correction_mapping[predicted_class]

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
