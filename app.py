import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BITS ML Evaluator App", layout="wide")

st.title("🔬 Chronic Disease & Diabetes Risk Evaluation Dashboard")
st.subheader("BITS Pilani WILP - M.Tech (AIML) Assignment 2")

st.sidebar.header("📁 Step 1: Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("🤖 Step 2: Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Classifier for Evaluation",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    if 'target' not in test_data.columns:
        st.error("Error: The dataset must contain a column named 'target'.")
    else:
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        st.success("Test dataset uploaded successfully!")

        with st.expander("👀 View Test Data Preview"):
            st.dataframe(test_data.head())
            st.write(f"**Shape:** {test_data.shape} rows, {test_data.shape} features")

        model_filename = f"model/{model_choice.lower().replace(' ', '_')}.pkl"

        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.markdown(f"### 📊 Performance Metrics for **{model_choice}**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("#### 📝 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())

            with right_col:
                st.markdown("#### 🧮 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
        else:
            st.error(f"Model artifact file not found at: `{model_filename}`")
else:
    st.info("💡 Please upload a `test_data.csv` file from the sidebar to begin model evaluation.")
