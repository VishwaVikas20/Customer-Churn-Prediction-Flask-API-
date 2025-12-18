from flask import Flask, render_template, request
import joblib
import pandas as pd
model= joblib.load('churn_pred.pkl')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        gender = request.form['gender']
        senior_citizen = int(request.form['seniorCitizen'])
        partner = request.form['partner']
        dependents = request.form['Dependents']
        tenure = int(request.form['tenure'])
        phone_service = request.form['PhoneService']
        multiple_lines = request.form['MultipleLines']
        internet_service = request.form['InternetService']
        online_security = request.form['OnlineSecurity']
        online_backup = request.form['OnlineBackup']
        device_protection = request.form['DeviceProtection']
        tech_support = request.form['TechSupport']
        streaming_tv = request.form['StreamingTV']
        streaming_movies = request.form['StreamingMovies']
        contract = request.form['Contract']
        paperless_billing = request.form['PaperlessBilling']
        payment_method = request.form['PaymentMethod']
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        
        input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]],
                                  columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])

        prediction = model.predict(input_data)[0]
        if prediction == 1:
            prediction = "The customer is likely to churn."
        else:
            prediction = "The customer is not likely to churn."
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)