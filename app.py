import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Fraud Detection System using Machine Learning")

# Upload file
uploaded_file = st.file_uploader("Upload your creditcard.csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    # Preprocess
    df.drop(['Time'], axis=1, inplace=True)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Show data overview
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())
    st.bar_chart(df['Class'].value_counts())

    # Split data
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("✅ Model Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Predict on user input (Optional)
    st.subheader("🔍 Predict on a Single Transaction (Advanced)")
    user_input = st.text_input("Enter 30 comma-separated V1-V28, Amount values (optional):")
    if user_input:
        try:
            vals = np.array([float(x) for x in user_input.strip().split(',')]).reshape(1, -1)
            prediction = model.predict(vals)
            result = "⚠️ Fraudulent" if prediction[0] == 1 else "✅ Not Fraudulent"
            st.success(f"Prediction: {result}")
        except:
            st.error("Invalid input! Please enter exactly 30 numeric values.")
