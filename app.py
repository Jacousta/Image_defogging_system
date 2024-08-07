
import os
from flask import Flask, request, render_template, send_from_directory,redirect
import numpy as np
from PIL import Image

# Load your deep learning model here
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assuming you're using TensorFlow
model = tf.keras.models.load_model('Double-U-Net.h5')

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def process_image(filepath):
    # Preprocess the image
    input_image = preprocess_image(filepath)
    
    # Make predictions
    processed_image = model.predict(input_image)
    
    # Post-process the output if necessary (e.g., convert to image format)
    processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert back to [0, 255]
    processed_image = Image.fromarray(processed_image)

    # Save the processed image
    enhanced_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
    processed_image.save(enhanced_image_path)
    return enhanced_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        processed_image_path = process_image(filepath)
        return send_from_directory(directory=PROCESSED_FOLDER, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
