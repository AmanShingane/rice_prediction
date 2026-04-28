from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__, template_folder='templates')

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# -----------------------------
# IMAGE SIZE (same as training)
# -----------------------------
IMAGE_SIZE = 256

# -----------------------------
# LOAD MODEL (SavedModel format)
# -----------------------------
MODEL_PATH = "model/1"

model = tf.saved_model.load(MODEL_PATH)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

infer = model.signatures["serving_default"]

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = infer(tf.constant(img_array))
    predictions = list(predictions.values())[0].numpy()

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(predictions[0])) * 100, 2)

    return predicted_class, confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_file = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file uploaded"
            return render_template('index.html', error=error)

        file = request.files['file']

        if file.filename == '':
            error = "Please select an image"
            return render_template('index.html', error=error)

        if file:
            filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                prediction, confidence = predict_image(file_path)
                image_file = file_path
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_file=image_file,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
