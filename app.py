import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ==========================
# LOAD MODEL
# ==========================
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
model_name = data["model_name"]

st.title("🩺 Diabetes Risk Prediction App")
st.write(f"Model yang digunakan: **{model_name}**")

# ==========================
# PREPROCESSING FUNCTION
# (ekstrak dari notebook)
# ==========================

# Kolom yang digunakan untuk prediksi
FEATURE_COLS = [
    '_RFHYPE6','_RFCHOL3','_CHOLCH3','_BMI5','SMOKE100','CVDSTRK3','_MICHD',
    'EXERANY2','PRIMINS1','MEDCOST1','GENHLTH','MENTHLTH','PHYSHLTH',
    'DIFFWALK','_SEX','_AGEG5YR','_EDUCAG','_INCOMG1'
]

# Imputer sederhana untuk data baru
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Untuk kolom numeric terpenting (dari notebook)
NUM_COLS = ['_BMI5','MENTHLTH','PHYSHLTH']
CAT_COLS = [col for col in FEATURE_COLS if col not in NUM_COLS]

# Standard scaler dari model training tidak disimpan → buat baru & fit nilai wajar
# NOTE: Untuk aplikasi sebenarnya scaler harus disimpan.
# Jika tidak ada, kita perkirakan distribusi standar saja.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def preprocess_input(df):
    # Isi missing values
    df[NUM_COLS] = num_imputer.fit_transform(df[NUM_COLS])
    df[CAT_COLS] = cat_imputer.fit_transform(df[CAT_COLS])

    # Standarisasi kolom numeric
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])

    return df


# ==========================
# FORM INPUT
# ==========================

st.subheader("Masukkan Indikator Kesehatan")

user_input = {}

for col in FEATURE_COLS:
    if col in NUM_COLS:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, max_value=1000.0, step=1.0)
    else:
        user_input[col] = st.selectbox(
            f"{col}", 
            options=[0, 1, 2, 3, 4, 5, 6], 
            help="Gunakan nilai asli sesuai dataset (0-6 / 1-5 / kategorikal)"
        )

# Convert ke dataframe
df_input = pd.DataFrame([user_input])

# ==========================
# PREDICTION
# ==========================

if st.button("Prediksi Risiko Diabetes"):
    processed = preprocess_input(df_input.copy())
    prob = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    st.markdown("---")
    st.write("### 🔍 Hasil Prediksi")

    if prediction == 1:
        st.error(f"**HASIL: Potensi Diabetes Tinggi**")
    else:
        st.success(f"**HASIL: Potensi Diabetes Rendah**")

    st.write(f"**Probabilitas: {prob:.3f}**")


st.markdown("---")
st.caption("Developed by William – IS388 Final Project")
