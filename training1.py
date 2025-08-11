# Student Grant Recommendation Prediction Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv('student_records.csv')
print("First 5 rows:")
print(data.head())

# 2. Basic data info
print("\nDataset Info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())

# 3. Encode categorical variables
# Encode 'Obedient' (Yes/No), 'Recommend' (Yes/No), and drop 'Name'
label_enc = LabelEncoder()
data['Obedient'] = label_enc.fit_transform(data['Obedient'])
data['Recommend'] = label_enc.fit_transform(data['Recommend'])

# Drop 'Name' as it's not useful for prediction
data = data.drop(columns=['Name'])

# 4. Split features and target
X = data.drop(columns=['Recommend'])
y = data['Recommend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Predict for a new student
new_student = pd.DataFrame({
    'OverallGrade': [85],
    'Obedient': [1],
    'ResearchScore': [90],
    'ProjectScore': [88]
})
new_student_scaled = scaler.transform(new_student)
recommendation = model.predict(new_student_scaled)
print("\nGrant Recommendation (1=Yes, 0=No):", recommendation[0])
print(data['Recommend'].value_counts())