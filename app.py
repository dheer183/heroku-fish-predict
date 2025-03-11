from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and encoder
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
