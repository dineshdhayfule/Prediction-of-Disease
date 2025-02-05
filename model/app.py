import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import os
# Function to load models
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load trained models
diabetes_model = load_model('model/diabetes_model.pkl')
heart_disease_model = load_model('model/heart_disease_model.pkl')
parkinsons_model = load_model('model/parkinsons_disease_model.pkl')

# Prediction functions
def predict_diabetes(inputs):
    return diabetes_model.predict(np.array([inputs]))

def predict_heart_disease(inputs):
    return heart_disease_model.predict(np.array([inputs]))

def predict_parkinsons(inputs):
    return parkinsons_model.predict(np.array([inputs]))

# Streamlit setup
st.set_page_config(page_title="Disease Prediction", layout="wide",page_icon="👨‍⚕️")

#center the header
# st.markdown("<h1 style='text-align: center; color: white;'>Disease Prediction App</h1>", unsafe_allow_html=True)
# Tabs for predictions and chat
# tab1, tab2, tab3 = st.tabs([
#     "Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction"])

with st.sidebar:
    selected_tab = option_menu("Disease Prediction App", ["Diabetes Prediction","Heart Disease Prediction","Parkinson's Prediction"]
            ,menu_icon='hospital-fill',icons=['activity','heart','person'],default_index=0)
                        

# Diabetes Prediction 
if selected_tab == "Diabetes Prediction":
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.text_input('Number of pregnancies')
    with col2:
        glucose = st.text_input('Glucose level')
    with col3:
        blood_pressure = st.text_input('Blood Pressure value')
    with col1:
        skin_thickness = st.text_input('Skin Thickness value')
    with col2:
        insulin = st.text_input('Insulin level')
    with col3:
        bmi = st.text_input('BMI value')
    with col1:
        diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function')
    with col2:
        age = st.text_input('Age')

    diabetes_inputs = {
        "Pregnancies": int(pregnancies) if pregnancies else 0,
        "Glucose": int(glucose) if glucose else 0,
        "Blood Pressure": int(blood_pressure) if blood_pressure else 0,
        "Skin Thickness": int(skin_thickness) if skin_thickness else 0,
        "Insulin": int(insulin) if insulin else 0,
        "BMI": float(bmi) if bmi else 0.0,
        "Diabetes Pedigree Function": float(diabetes_pedigree_function) if diabetes_pedigree_function else 0.0,
        "Age": int(age) if age else 0
    }
    if st.button("Predict Diabetes", key="predict_diabetes"):
        prediction = predict_diabetes(list(diabetes_inputs.values()))
        if prediction[0] == 1:
            stm ="Person is Diabetic"
            st.error(stm)
        else:
            # comment: 
            stm ="Person is Non-Diabetic"
            st.success(stm)
        # st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
        

# Heart Disease Prediction 
if selected_tab == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", key="heart_sex")
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Cholesterol')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('Rest ECG')
    with col2:
        thalach = st.text_input('Maximum Heart Rate')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('Oldpeak')
    with col2:
        slope = st.text_input('Slope')
    with col3:
        ca = st.text_input('Number of Major Vessels')
    with col1:
        thal = st.text_input('Thal')

    heart_inputs = {
        "Age": int(age) if age else 0,
        "Sex": int(sex),
        "Chest Pain Type": int(cp) if cp else 0,
        "Resting Blood Pressure": int(trestbps) if trestbps else 0,
        "Cholesterol": int(chol) if chol else 0,
        "Fasting Blood Sugar": int(fbs) if fbs else 0,
        "Rest ECG": int(restecg) if restecg else 0,
        "Maximum Heart Rate": int(thalach) if thalach else 0,
        "Exercise Induced Angina": int(exang) if exang else 0,
        "Oldpeak": float(oldpeak) if oldpeak else 0.0,
        "Slope": int(slope) if slope else 0,
        "Number of Major Vessels": int(ca) if ca else 0,
        "Thal": int(thal) if thal else 0
    }
    
    # Prediction button
    if st.button("Predict Heart Disease", key="predict_heart"):
        prediction = predict_heart_disease(list(heart_inputs.values()))
        if prediction[0] == 1:
            stm = "Heart Disease Detected"
            st.error(stm)
        else:
            stm = "No Heart Disease"
            st.success(stm)

# Parkinson's Disease Prediction 
if selected_tab == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    col1, col2, col3 = st.columns(3)
    parkinsons_inputs = []
    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    
    for i, feature in enumerate(parkinsons_features):
        if i % 3 == 0:
            with col1:
                parkinsons_inputs.append(st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.01,  # Explicitly set step size for floating-point precision
                    key=f"parkinsons_{feature}"
                ))
        elif i % 3 == 1:
            with col2:
                parkinsons_inputs.append(st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.01,  # Explicitly set step size for floating-point precision
                    key=f"parkinsons_{feature}"
                ))
        else:
            with col3:
                parkinsons_inputs.append(st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.01,  # Explicitly set step size for floating-point precision
                    key=f"parkinsons_{feature}"
                ))
    
    if st.button("Predict Parkinson's Disease", key="predict_parkinsons"):
        prediction = predict_parkinsons(parkinsons_inputs)
        if prediction[0] == 1:
            stm = "Parkinson's Detected"
            st.error(stm)
        else:
            stm = "No Parkinson's Detected"
            st.success(stm)
