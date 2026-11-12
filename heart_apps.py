import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient information to predict heart disease.")

# Load fitted pipeline
@st.cache_resource
def load_model():
    return joblib.load("heart_disease.joblib")

model = load_model()

# ==========================================
# Input form
# ==========================================

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 50)

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        dataset = st.selectbox(
            "Dataset",
            ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            0, 300, 120
        )

        chol = st.number_input(
            "Cholesterol",
            0, 700, 200
        )

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [True, False],
            format_func=lambda x: "Yes" if x else "No"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [
                "normal",
                "ST-T wave abnormality",
                "left ventricular hypertrophy"
            ]
        )

        thalch = st.number_input(
            "Maximum Heart Rate",
            0, 250, 150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [True, False],
            format_func=lambda x: "Yes" if x else "No"
        )

        oldpeak = st.number_input(
            "Oldpeak",
            0.0, 10.0, 1.0, step=0.1
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# ==========================================
# Prediction
# ==========================================

if submitted:

    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'dataset': dataset,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("No Heart Disease Predicted")
    else:
        st.warning("Heart Disease Predicted")

    st.metric(
        "Probability of Heart Disease",
        f"{probability[list(model.classes_).index(1)]:.2%}"
    )

    probability_df = pd.DataFrame({
        "Class": ["No Heart Disease", "Heart Disease 1", "Heart Disease 2"],
        "Probability": probability
    })

    st.write("Prediction Probability")
    st.bar_chart(
        probability_df.set_index("Class")
    )
