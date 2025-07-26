import streamlit as st
import pandas as pd
import requests
import joblib
import io
import plotly.express as px
import os

MODEL_URL = "https://github.com/Samarth4023/IBM-Internship/releases/download/model/Final_Ensemble_Model.pkl"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or <=50K based on personal and professional details.")

# Sidebar - branding and batch prediction
with st.sidebar:
    st.image("Images/Employee.png", width=300)
    st.markdown("---")
    st.header("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file as per Input Summary and Scroll Down", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview:", batch_df.head())
        batch_df.replace('?', 'Others', inplace=True)
        
        preds = model.predict(batch_df)
        pred_labels = ['>50K' if p == 1 else '<=50K' for p in preds]
        batch_df['Predicted Salary Class'] = pred_labels
        
        st.success("✅ Batch prediction completed")
        st.dataframe(batch_df, use_container_width=True)
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, file_name="batch_predictions.csv")

# Main area - user input form
st.markdown("### 🧑‍💼 Enter Employee Details")
with st.form("user_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 30)
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Others'])
        education_num = st.slider("Education Number(Years)", 1, 20, 10)
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                         'Separated', 'Widowed', 'Married-spouse-absent', 'Others'])

    with col2:
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                                 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                                 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                                 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                                 'Armed-Forces', 'Others'])
        relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                                     'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        sex = st.radio("Gender", ['Male', 'Female'])

    with col3:
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany',
                                                         'Canada', 'India', 'England', 'China', 'Others'])

    submit_btn = st.form_submit_button("Predict Salary Class")


    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [0],  # Placeholder
        'educational-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    st.markdown("---")
    st.subheader("📥 Input Summary")
    st.markdown("NOTE: Below placeholders will only change to user values when the predict button will be clicked")
    st.write(input_df)


if submit_btn:
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pred_class = '>50K' if prediction == 1 else '<=50K'
    st.success(f"✅ Predicted Salary Class: {pred_class}")

    # Probability pie chart
    fig_pie = px.pie(values=proba, names=['<=50K', '>50K'], title="Prediction Confidence", hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Feature distributions
    st.markdown("---")
    st.subheader("📊 Feature Distributions for Predicted Class")

    df = pd.read_csv("Dataset/adult.csv")
    df.replace('?', 'Others', inplace=True)
    df = df.rename(columns={'educational-num': 'education_num', 'marital-status': 'marital_status',
                            'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss',
                            'hours-per-week': 'hours_per_week', 'native-country': 'native_country',
                            'gender': 'sex'})

    target_col = 'income'  # Define the target column explicitly
    df[target_col] = df[target_col].map({'>50K': 1, '<=50K': 0})
    df = df[df[target_col] == prediction]  # Filter only predicted class

    numeric_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col} for class {pred_class}")
        st.plotly_chart(fig, use_container_width=True)

    for col in categorical_cols:
        vc_df = df[col].value_counts().reset_index()
        vc_df.columns = [col, 'count']  # Rename properly

        fig = px.bar(vc_df, x=col, y='count',
                    title=f"Distribution of {col} for class {pred_class}",
                    labels={col: col, 'count': 'Count'})
        st.plotly_chart(fig, use_container_width=True)