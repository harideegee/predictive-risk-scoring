from flask import Flask, request, render_template, jsonify
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Friendly form field labels
labels = {
    'age': 'Age (years)',
    'weight': 'Weight (in lbs)',
    'sex': 'Biological Sex',
    'height': 'Height (in inches)',
    'sys_bp': 'Systolic Blood Pressure (mmHg)',
    'smoker': 'Current Smoker Status',
    'nic_other': 'Use of Nicotine Products Other Than Smoking',
    'num_meds': 'Number of Medications Currently Taken',
    'occup_danger': 'Occupational Danger Level',
    'ls_danger': 'Lifestyle Danger Level',
    'cannabis': 'Cannabis Use',
    'opioids': 'Opioid Use',
    'other_drugs': 'Use of Other Drugs',
    'drinks_aweek': 'Alcoholic Drinks Per Week',
    'addiction': 'Addiction History',
    'major_surgery_num': 'Number of Major Surgeries',
    'diabetes': 'Diabetes Diagnosis',
    'hds': 'Health Disease Status',
    'cholesterol': 'Bad Cholesterol Level (mg/dL)',
    'asthma': 'Asthma Diagnosis',
    'immune_defic': 'Immune Deficiency',
    'family_cancer': 'Family History of Cancer',
    'family_heart_disease': 'Family History of Heart Disease',
    'family_cholesterol': 'Family History of High Cholesterol',
}

# Model Training
features = list(labels.keys())
cat_cols = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
            'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
            'family_cholesterol']

label_encoders = {}

binary_cols = ['smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
               'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
               'family_cholesterol']

for col in binary_cols:
    le = LabelEncoder()
    le.classes_ = np.array(['n', 'y'])
    label_encoders[col] = le

le_sex = LabelEncoder()
le_sex.classes_ = np.array(['f', 'm'])
label_encoders['sex'] = le_sex

df = pd.read_json('data.json')

X = df[features].copy()
for col in cat_cols:
    X[col] = label_encoders[col].transform(X[col].astype(str))
y = X.pop('hds')


model = LGBMClassifier(random_state=42)
model.fit(X, y)

# Flask App
@app.route('/')
def home():
    return render_template('index.html', features=features, labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for feature in features:
        val = request.form.get(feature)
        if val is None or val == '':
            return f'Missing input for {feature}', 400
        input_data[feature] = val

    input_df = pd.DataFrame([input_data])


    for col in cat_cols:
        if col in input_df:
            val = input_df.loc[0, col]
            if val in label_encoders[col].classes_:
                input_df.loc[0, col] = label_encoders[col].transform([val])[0]
            else:
                input_df.loc[0, col] = -1

    for col in cat_cols:
        if col in input_df:
            input_df[col] = int(input_df[col])

    numeric_cols = [col for col in features if col not in cat_cols]
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col])


    if 'hds' in input_df.columns:
        input_df = input_df.drop(columns=['hds'])



    prob = model.predict_proba(input_df)[:, 1][0]
    risk_score = round(prob * 100, 2)

    return render_template('result.html', risk_score=risk_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
