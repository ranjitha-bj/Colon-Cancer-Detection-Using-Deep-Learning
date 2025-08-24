
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model("model/colon_model.h5")

# Prediction function
def predict_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]

    label = "Cancerous" if prediction < 0.5 else "Non-Cancerous"
    confidence = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)

    return label, confidence

# Routes
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route('/logindash', methods=['GET', 'POST'])
def logindash():
    if request.method == 'POST':
        name = request.form.get('name')
        password = request.form.get('password')

        cname='admin'
        cpass='admin'
        # ✅ For now, no database check — just redirect
        if name == cname and cpass==password:  
            return render_template('index.html')

    return render_template('login.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        f = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)
        label, confidence = predict_image(filepath)
        return render_template("predict.html", file=f.filename, label=label, confidence=confidence)
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
