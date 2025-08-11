import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier

# Load dataset
data = pd.read_csv("student_records.csv")

# One-hot encode categorical
onehot_cols = ["OverallGrade", "Obedient"]
onehot = OneHotEncoder(drop="first", sparse_output=False)
onehot_df = pd.DataFrame(
    onehot.fit_transform(data[onehot_cols]),
    columns=onehot.get_feature_names_out(onehot_cols),
    index=data.index
)

# Drop original categorical & name column
data = data.drop(columns=onehot_cols + ["Name"])

# Add one-hot features
data = pd.concat([data, onehot_df], axis=1)

# Encode target
label = LabelEncoder()
data['Recommend'] = label.fit_transform(data['Recommend'])

# Split into features/target
y = data['Recommend']
X = data.drop(columns='Recommend')

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train using SGDClassifier (logistic regression style)
model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Save model & encoders
with open("onehot.pkl", "wb") as f:
    pickle.dump(onehot, f)
with open("nb_model.pkl", "wb") as f:  # Keeping same name for app.py compatibility
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label.pkl", "wb") as f:
    pickle.dump(label, f)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClass Distribution after SMOTE:\n", pd.Series(y).value_counts())
