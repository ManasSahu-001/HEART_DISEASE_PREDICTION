import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the probability of heart disease using Machine Learning.
*Data Source: UCI Heart Disease Repository*
""")

# --- STEP 1: LOAD AND PREPARE DATA ---
@st.cache_data # Caches data so it doesn't reload on every interaction
def load_data():
    # Ensure heart.csv is in the same directory
    df = pd.read_csv("heart.csv")
    return df

try:
    data = load_data()
    X = data.iloc[:, 0:13].values
    y = data.iloc[:, 13].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # --- STEP 2: TRAIN MODELS ---
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier()
    }

    accuracies = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies[name] = round(acc * 100, 2)
        trained_models[name] = model

    # --- STEP 3: SIDEBAR INPUTS ---
    st.sidebar.header("User Input Features")
    
    def user_input_features():
        age = st.sidebar.number_input("Age", 1, 100, 50)
        sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
        cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
        restecg = st.sidebar.selectbox("Resting ECG results (0-2)", [0, 1, 2])
        thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
        oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
        slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.sidebar.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversable)", [1, 2, 3])
        
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        return features

    input_df = user_input_features()

    # --- STEP 4: VISUALIZATION & COMPARISON ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Accuracy Comparison")
        fig, ax = plt.subplots()
        colors = ['red', 'orange', 'blue', 'green']
        ax.bar(accuracies.keys(), accuracies.values(), color=colors)
        ax.set_ylabel("Accuracy (%)")
        st.pyplot(fig)

    with col2:
        st.subheader("Model Performance Metrics")
        st.table(pd.DataFrame(accuracies.items(), columns=["Model", "Accuracy (%)"]))

    # --- STEP 5: PREDICTION ---
    st.divider()
    st.subheader("Make a Prediction")
    
    # Let user choose which model to use for the prediction
    selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
    
    if st.button("Predict"):
        prediction = trained_models[selected_model_name].predict(input_df)
        
        if prediction[0] == 1:
            st.error(f"**Result:** The {selected_model_name} model predicts you **MAY** have heart disease.")
        else:
            st.success(f"**Result:** The {selected_model_name} model predicts you **DO NOT** have heart disease.")

except FileNotFoundError:
    st.error("Error: 'heart.csv' not found. Please ensure the dataset is in the folder.")