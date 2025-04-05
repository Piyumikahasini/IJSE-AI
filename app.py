# pip install Flask

# python --version

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl") 

training_features = [
    'age', 'job',  
    'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month',
    'duration', 'campaign', 'pdays', 'previous', 'poutcome', 
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        age = data.get('age')
        job = data.get('job')
        marital = data.get('marital')
        education = data.get('education')
        default = data.get('default')
        housing = data.get('housing')
        loan = data.get('loan')

        duration = 563.6641774157049  # apply mean duration
        campaign = 1  # Number of contacts for the client during this campaign
        pdays = 999  # No previous contact
        previous = 0  # First contact
        poutcome = 'nonexistent'  # No previous outcome
        emp_var_rate = 1.4  # Default placeholder for economic data
        cons_price_idx = 93.918  # Default placeholder for economic data
        cons_conf_idx = -42.7  # Default placeholder for economic data
        euribor3m = 4.961  # Default placeholder for economic data
        nr_employed = 5228.1  # Default placeholder for economic data

        input_data = {
            'age': [int(age)] if age is not None else [0],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed],
            'contact': [0], 
            'month': [6], 
            'day_of_week': [1]
        }


        # Create DataFrame for the input sample
        df_input = pd.DataFrame(input_data)

        # List of columns to encode, ensure to exclude the columns that the user doesn't provide
        columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']

        # Apply One-Hot Encoding only to the available columns
        df_input_encoded = pd.get_dummies(df_input, columns=columns_to_encode, prefix='', prefix_sep='')

        # Remove duplicate columns (if any)
        df_input_encoded = df_input_encoded.loc[:, ~df_input_encoded.columns.duplicated()]

        # Ensure all training features exist in input data (set missing ones to 0)
        df_input_encoded = df_input_encoded.reindex(columns=training_features, fill_value=0)

        # Ensure the column order matches the scaler's features exactly
        df_input_encoded = df_input_encoded[scaler.feature_names_in_]

        # Convert the data type to float64 to match the scaler's expectations
        df_input_encoded = df_input_encoded.astype(np.float64)

        # Apply the same scaling as the training data
        df_input_scaled = scaler.transform(df_input_encoded)

        # Make prediction using the saved SVM model
        prediction = svm_model.predict(df_input_scaled)

        # Convert prediction to 'yes' or 'no'
        predicted_subscription = 'yes' if prediction[0] == 1 else 'no'

        # Return the result as JSON
        return jsonify({'result': predicted_subscription})

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(host="0.0.0.0", port=80) # to change the port .


# make docker file
# docker build -t class3 .
# docker run -p 5000:5000 class3

