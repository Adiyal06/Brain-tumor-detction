from flask import Flask, request, render_template
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'adiyal2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Mapping of class indices to class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file
        file_path = os.path.join('static', f.filename)
        f.save(file_path)

        # Make prediction
        img = image.load_img(file_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(preds)
        # Get the corresponding class label
        predicted_class = class_labels[predicted_class_index]
        # Get the probability of the predicted class
        # probability = round(preds[0][predicted_class_index] * 100, 2)

        return render_template('result.html', predicted_class=predicted_class, file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)