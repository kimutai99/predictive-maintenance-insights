
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('C:/Users/HP/MachineFailurePredictor/src/models/machine_failure_model.pkl')
le = joblib.load('C:/Users/HP/MachineFailurePredictor/src/models/label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(df)
    component = le.inverse_transform(prediction)
    
    return jsonify({'predicted_component': component[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
