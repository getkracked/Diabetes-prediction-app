from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('diabetes_model.h5')
scaler = joblib.load('scaler.save')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Prepare the input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        result = "Positive for Diabetes" if prediction[0][0] >= 0.5 else "Negative for Diabetes"

        return render_template('form.html', result=result)
    except Exception as e:
        return render_template('form.html', result="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
