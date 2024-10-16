from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('models/discriminator_model.h5')  # Adjust the path if necessary

def load_and_preprocess_image(img_file):
    img = image.load_img(img_file, target_size=(256, 256))  # Resize to match model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_anomaly(img_array):
    score = model.predict(img_array)
    is_anomaly = score < 0.5  # Adjust threshold as needed
    return "Pneumonia" if is_anomaly else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename != '':
            # Load the image directly from the uploaded file
            img_array = load_and_preprocess_image(BytesIO(uploaded_file.read()))
            prediction = predict_anomaly(img_array)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
