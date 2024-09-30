# Diabetes Prediction Flask App

This project is a Flask web application that predicts whether a person is likely to have diabetes based on certain health metrics. The model is built using Keras and trained on a dataset containing the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The prediction is based on a pre-trained machine learning model (`diabetes_model.h5`), and the form input data is scaled using a `StandardScaler` fitted during model training.

## Project Structure
```bash
├── app.py                  # Main Flask application
├── diabetes_model.h5        # Trained Keras model for diabetes prediction
├── scaler.pkl               # StandardScaler fitted on training data
├── templates/
│   ├── index.html           # Form to input health metrics
│   ├── result.html          # Displays the prediction result
└── README.md                # Project README
