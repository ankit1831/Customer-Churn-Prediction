from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

'''# Load model and full preprocessing pipeline
model = pickle.load(open('churn.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get input from form
        data = {
            'CreditScore': int(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Tenure': int(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': int(request.form['NumOfProducts']),
            'HasCrCard': int(request.form['HasCrCard']),
            'IsActiveMember': int(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }
        
        # Create DataFrame
        pred_df=pd.DataFrame([data])
        print(pred_df)

        # Predict
        predict_pipeline=PredictPipeline()
        result = predict_pipeline.predict(pred_df)[0]
        prediction = 'Customer Will Exit 😢' if result == 1 else 'Customer Will Stay 😊'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080) 
