import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Student Recommendation Prediction - Fairer Model")

# Load saved model & encoders
try:
    with open("nb_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)
    with open("label.pkl", "rb") as f:
        loaded_label = pickle.load(f)
    with open("onehot.pkl", "rb") as f:
        loaded_onehot = pickle.load(f)
except FileNotFoundError:
    st.error("Trained model or encoders not found. Please train the model first.")
    st.stop()

# Manual input section
st.subheader("Enter Student Details")
overall_grade = st.selectbox("Overall Grade", loaded_onehot.categories_[0])
obedient = st.selectbox("Obedient", loaded_onehot.categories_[1])
research_score = st.number_input("Research Score", min_value=0.0)
project_score = st.number_input("Project Score", min_value=0.0)

if st.button("Predict"):
    # Create DataFrame with raw input
    input_df = pd.DataFrame(
        [[overall_grade, obedient, research_score, project_score]],
        columns=["OverallGrade", "Obedient", "ResearchScore", "ProjectScore"]
    )

    # One-hot encode categorical
    cat_transformed = loaded_onehot.transform(input_df[["OverallGrade", "Obedient"]])
    cat_df = pd.DataFrame(
        cat_transformed,
        columns=loaded_onehot.get_feature_names_out(["OverallGrade", "Obedient"])
    )

    # Merge numerical + categorical
    final_df = pd.concat(
        [input_df[["ResearchScore", "ProjectScore"]].reset_index(drop=True),
         cat_df.reset_index(drop=True)], axis=1
    )

    # Scale
    final_scaled = loaded_scaler.transform(final_df)

    # Predict
    pred = loaded_model.predict(final_scaled)[0]
    pred_label = loaded_label.inverse_transform([pred])[0]
    st.success(f"Prediction: {pred_label}")

    # Show feature importance if model supports it
    if hasattr(loaded_model, "coef_"):
        importance = loaded_model.coef_[0]
        importance_df = pd.DataFrame({
            "Feature": final_df.columns,
            "Importance": importance
        }).sort_values(by="Importance", key=abs, ascending=False)

        st.subheader("Feature Influence")
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm'))
