from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = np.array([list(map(float, data.values()))])
    
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text=f"Predicted Price: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)