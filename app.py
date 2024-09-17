import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('Heart_Disease_Prediction.csv')
    return data

# Preprocess the data
def preprocess_data(data):
    le = LabelEncoder()
    categorical_columns = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
    
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    
    return data

# Train the model
def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Save the model
def save_model(model, scaler):
    joblib.dump(model, 'heart_disease_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

# Load the model
def load_model():
    model = joblib.load('heart_disease_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Main Streamlit app
def main():
    st.title('Heart Disease Prediction App')

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Model Training", "Prediction"])

    # Load and preprocess data
    data = load_data()
    processed_data = preprocess_data(data)

    if page == "Data Exploration":
        st.header("Data Exploration")

        # Display raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw data")
            st.write(data)

        # Display processed data
        if st.checkbox("Show processed data"):
            st.subheader("Processed data")
            st.write(processed_data)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(processed_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Distribution of Heart Disease
        st.subheader("Distribution of Heart Disease")
        fig, ax = plt.subplots()
        sns.countplot(x='Heart Disease', data=data, ax=ax)
        st.pyplot(fig)

        # Feature distributions
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select a feature to visualize", data.columns)
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=feature, hue='Heart Disease', kde=True, ax=ax)
        st.pyplot(fig)

    elif page == "Model Training":
        st.header("Model Training")

        # Split the data
        X = processed_data.drop('Heart Disease', axis=1)
        y = processed_data['Heart Disease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                model = train_model(X_train_scaled, y_train)
                save_model(model, scaler)
                st.success("Model trained and saved successfully!")

            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            st.bar_chart(feature_importance.set_index('feature')['importance'])

    elif page == "Prediction":
        st.header("Heart Disease Prediction")

        # Load the trained model
        try:
            model, scaler = load_model()
        except FileNotFoundError:
            st.error("Trained model not found. Please train the model first.")
            return

        # User input
        st.subheader("Enter Patient Information")
        age = st.slider('Age', 20, 80, 50)
        sex = st.selectbox('Sex', ('Male', 'Female'))
        cp = st.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
        bp = st.slider('Resting Blood Pressure', 90, 200, 120)
        chol = st.slider('Cholesterol', 100, 600, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('Yes', 'No'))
        ekg = st.selectbox('EKG Results', ('Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'))
        max_hr = st.slider('Maximum Heart Rate', 60, 220, 150)
        angina = st.selectbox('Exercise Induced Angina', ('Yes', 'No'))
        st_dep = st.slider('ST Depression', 0.0, 6.2, 0.0, 0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', ('Upsloping', 'Flat', 'Downsloping'))
        vessels = st.selectbox('Number of Vessels Colored by Fluoroscopy', ('0', '1', '2', '3'))
        thal = st.selectbox('Thallium Stress Test Result', ('Normal', 'Fixed Defect', 'Reversible Defect'))

        # Prepare user input for prediction
        sex = 0 if sex == 'Male' else 1
        cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
        fbs = 1 if fbs == 'Yes' else 0
        ekg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(ekg)
        angina = 1 if angina == 'Yes' else 0
        slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
        vessels = int(vessels)
        thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

        user_input = np.array([age, sex, cp, bp, chol, fbs, ekg, max_hr, angina, st_dep, slope, vessels, thal]).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        if st.button('Predict'):
            prediction = model.predict(user_input_scaled)
            probability = model.predict_proba(user_input_scaled)

            st.subheader('Prediction Result')
            if prediction[0] == 1:
                st.warning('The model predicts a high likelihood of heart disease.')
            else:
                st.success('The model predicts a low likelihood of heart disease.')

            st.write(f'Probability of heart disease: {probability[0][1]:.2f}')

if __name__ == '__main__':
    main()