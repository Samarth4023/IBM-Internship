# 🧠 Employee Salary Prediction – IBM Internship Project

This repository contains the implementation of an **Employee Salary Prediction System**, developed during my IBM Internship under the Artificial Intelligence role. The project predicts whether an individual earns **>50K** or **<=50K** annually based on demographic and professional attributes.

---

## 🚀 Project Overview

During the internship, I built an end-to-end ML system that:
- Uses data preprocessing and feature engineering to handle real-world employee data.
- Trains multiple machine learning models and combines them using **ensemble learning** for improved performance.
- Deploys a **Streamlit-based web application** that provides both single and batch predictions with user-friendly visualizations.

---

## 📂 Repository Structure

```bash
.
├── Dataset/
│   └── adult.csv                         # Adult Income Dataset
├── Images/
│   └── Employee.png                      # Visual asset for the app
├── Model/
│   └── Model Information.txt             # Model summary and notes
├── Skillsbuild Certificates/
│   ├── Artificial Intelligence - credly.pdf
│   ├── Edunet-Learning Plan Completion.pdf
│   └── Edunet-Orientation Certificate.pdf
├── sample_batch.csv                      # Example CSV for batch predictions
├── Employee_Salary_Prediction.ipynb      # Full Jupyter notebook (EDA to model)
├── app.py                                # Streamlit application
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation (this file)
```

---

## 🔍 Features

* ✅ Preprocessing: Handles missing values, encodes categoricals, and normalizes inputs.
* 📊 Visualizations: ROC, Precision-Recall Curves, Pie charts, Class distributions.
* 📈 Models: Logistic Regression, Random Forest, SVM + Voting Ensemble.
* 📁 Batch prediction: Upload CSV for multi-row prediction.
* 🧾 Real-time prediction summary and model confidence.

---

## 🏁 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Samarth4023/IBM-Internship.git
cd IBM-Internship
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Model Performance

### ✅ Final Ensemble Results:

| Metric             | Training Set | Test Set |
|--------------------|--------------|----------|
| Accuracy           | 93.89%       | 85.68%   |
| F1 Score (>50K)    | 0.87         | 0.68     |
| Precision (>50K)   | 0.94         | 0.76     |
| Recall (>50K)      | 0.80         | 0.62     |

- 📌 High precision ensures **fewer false positives** in predicting high earners (>50K).
- 📊 Training accuracy shows strong learning capability without overfitting.
- 🧠 Macro F1 Score (Test): **0.79**, indicating fair balance across classes.

> Ensemble model outperforms individual base models with better generalization and robust prediction confidence.

---

## 📚 Learning Outcomes

* Hands-on experience with data pipelines, model training, and app deployment
* Understanding of **bias handling**, **model evaluation**, and **user experience design**
* Deployment-ready ML app using **Streamlit**, **Pandas**, **Scikit-learn**, and **Joblib**

---

## 🧑‍🏫 Mentor Experience

Working under the guidance of industry professionals at IBM helped me sharpen both my **technical and communication skills**, with regular reviews and constructive feedback loops.

---

## 🔮 Future Scope

* Integrate deep learning models for better performance
* Add explainability using **SHAP** or **LIME**
* Enable API-based access for external systems
* Extend model to predict salary ranges, not just binary outcomes

---

## 🙌 Acknowledgements

* \[IBM Internship Program – Skillsbuild Platform and Edunet Foundation]
* [Scikit-Learn](https://scikit-learn.org/stable/user_guide.html)
* [Pandas](https://pandas.pydata.org/docs)
* [Streamlit.io](https://streamlit.io)
* [HuggingFace Spaces](https://huggingface.co/spaces)

## 👤 Author

Samarth Pujari

AI Intern @ IBM

Connect with me on [LinkedIn](www.linkedin.com/in/samarth-pujari-328a1326a) | [Kaggle](https://www.kaggle.com/samarthpujari)
