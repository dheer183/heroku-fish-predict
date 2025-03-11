from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['Weight']),
        float(request.form['Length1']),
        float(request.form['Length2']),
        float(request.form['Length3']),
        float(request.form['Height']),
        float(request.form['Width'])
    ]
    prediction = model.predict([features])
    species = le.inverse_transform(prediction)[0]
    return render_template('index.html', prediction=species)

if __name__ == '__main__':
    app.run()
