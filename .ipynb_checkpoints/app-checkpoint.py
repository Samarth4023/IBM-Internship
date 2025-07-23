import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set wide layout and title
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Prediction using Machine Learning")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** using ML models.")

# Load trained model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("Model/Salary_prediction_model.pkl")
    return model

model = load_model()

# Sidebar - Collect user input
st.sidebar.header("📋 Input Employee Information")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Display user input
st.subheader("🔍 Input Data Preview")
st.dataframe(input_df)

# --- Predict and Display Results ---
if st.button("🎯 Predict Salary Class"):
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("✅ Prediction Result")
        st.success(f"Predicted Salary Class: **{prediction}**")

        # Probability pie chart
        fig1, ax1 = plt.subplots()
        ax1.pie(proba, labels=model.classes_, autopct='%1.1f%%', startangle=90, colors=['#66bb6a', '#ef5350'])
        ax1.axis('equal')
        st.pyplot(fig1)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Batch Prediction ---
st.markdown("---")
st.subheader("📂 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV file with employee data", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("🔍 Preview of Uploaded Data")
        st.dataframe(batch_data.head())

        batch_preds = model.predict(batch_data)
        batch_proba = model.predict_proba(batch_data)

        batch_data["Predicted Salary Class"] = batch_preds
        st.success("✅ Batch Predictions Completed")
        st.dataframe(batch_data.head())

        # Visualization: Class Distribution
        st.subheader("📊 Predicted Salary Class Distribution")
        class_counts = batch_data["Predicted Salary Class"].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2', ax=ax2)
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Predicted Salary Class")
        st.pyplot(fig2)

        # Download predicted results
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
