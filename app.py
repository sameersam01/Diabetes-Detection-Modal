from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load and prepare the diabetes dataset
def prepare_model():
    # Sample diabetes dataset (you can replace this with your own dataset)
    data = pd.read_csv('diabetes.csv')
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_values = []
        
        for feature in features:
            value = float(request.form[feature])
            input_values.append(value)
        
        # Load the model and scaler
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the input values
        input_scaled = scaler.transform([input_values])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': round(probability * 100, 2)
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    prepare_model()  # Initialize the model before starting the app
    app.run(debug=True)
