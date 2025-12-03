import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ========================================
# LOAD MODEL
# ========================================
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
model_name = data["model_name"]

st.title("🩺 Diabetes Risk Prediction App")
st.write(f"Model yang digunakan: **{model_name}**")

# ========================================
# FITUR YANG DIGUNAKAN (Disederhanakan)
# ========================================
FEATURE_COLS = [
    "_RFHYPE6",   # Hypertension
    "_RFCHOL3",   # High Cholesterol
    "_BMI5",      # BMI × 10
    "GENHLTH",    # General Health
    "_AGEG5YR",   # Age group
    "MENTHLTH",   # Mental unhealthy days
    "PHYSHLTH"    # Physical unhealthy days
]

NUM_COLS = ["_BMI5", "MENTHLTH", "PHYSHLTH"]
CAT_COLS = ["_RFHYPE6", "_RFCHOL3", "GENHLTH", "_AGEG5YR"]

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
scaler = StandardScaler()


def preprocess(df):
    df[NUM_COLS] = num_imputer.fit_transform(df[NUM_COLS])
    df[CAT_COLS] = cat_imputer.fit_transform(df[CAT_COLS])
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    return df


# ========================================
# INPUT FORM
# ========================================
st.subheader("Masukkan Data Anda")

col1, col2 = st.columns(2)

with col1:
    hypertension = st.selectbox(
        "Pernah didiagnosis hipertensi?", [0, 1],
        help="1 = Ya, 0 = Tidak"
    )
    cholesterol = st.selectbox(
        "Kolesterol Tinggi?", [0, 1],
        help="1 = Ya, 0 = Tidak"
    )
    bmi = st.number_input(
        "BMI Anda",
        min_value=10.0, max_value=60.0, step=0.1,
        help="Masukkan BMI normal (18–40). Sistem akan mengubah ke format _BMI5"
    )

with col2:
    genhlth = st.selectbox(
        "Seberapa sehat kondisi umum Anda?",
        [1, 2, 3, 4, 5],
        help="1=Excellent, 5=Poor"
    )
    age_group = st.selectbox(
        "Kelompok Usia",
        options=list(range(1, 14)),
        help="1 = 18–24, 2 = 25–29, ..., 13 = 80+"
    )
    menthlth = st.slider(
        "Hari kesehatan mental buruk (0–30)", 0, 30, 0
    )
    physhlth = st.slider(
        "Hari kesehatan fisik buruk (0–30)", 0, 30, 0
    )

# Convert BMI → dataset format (×10)
bmi_scaled = bmi * 10

# Create dataframe
input_df = pd.DataFrame([{
    "_RFHYPE6": hypertension,
    "_RFCHOL3": cholesterol,
    "_BMI5": bmi_scaled,
    "GENHLTH": genhlth,
    "_AGEG5YR": age_group,
    "MENTHLTH": menthlth,
    "PHYSHLTH": physhlth
}])

# ========================================
# PREDICTION
# ========================================
if st.button("Prediksi Risiko Diabetes"):
    processed = preprocess(input_df.copy())
    prob = model.predict_proba(processed)[0][1]
    pred = model.predict(processed)[0]

    st.markdown("---")
    st.write("### 🔍 Hasil Prediksi")

    if pred == 1:
        st.error("**Hasil: Risiko Diabetes Tinggi**")
    else:
        st.success("**Hasil: Risiko Diabetes Rendah**")

    st.write(f"**Probabilitas: {prob:.3f}**")
