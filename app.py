from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the "Perfect" versions we just created
model = joblib.load('blower_model_v3.pkl')
scaler = joblib.load('scaler_v3.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get raw values from the web form
    temp = float(request.form['temp'])
    vib = float(request.form['vib'])
    volt = float(request.form['volt'])
    curr = float(request.form['curr'])
    
    # 2. Put them in a DataFrame (Matches the model's training)
    features = pd.DataFrame([[temp, vib, volt, curr]], 
                           columns=['Temperature', 'Vibration', 'Voltage', 'Current'])
    
    # 3. Scale the data exactly like we did in Colab
    features_scaled = scaler.transform(features)
    
    # 4. Get the prediction (0 or 1)
    prediction = model.predict(features_scaled)
    
    result = "⚠️ FAULT DETECTED" if prediction[0] == 1 else "✅ SYSTEM HEALTHY"
    color = "danger" if prediction[0] == 1 else "success"
    
    return render_template('index.html', prediction_text=result, alert_type=color)

if __name__ == "__main__":
    app.run(debug=True)
