from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

app = Flask(__name__)

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler with absolute paths
model = joblib.load(os.path.join(current_dir, 'best_model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get only the 7 features from form
            features = [float(request.form[f'f{i}']) for i in range(7)]
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            prediction = 'Benign' if pred == 1 else 'Malignant'
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction, feature_names=feature_names[:7])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)