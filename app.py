import csv
from io import StringIO

import pandas as pd
import streamlit as st

COLUMNS = "age sex dataset cp trestbps chol fbs restecg thalch exang oldpeak".split()
NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak"]
DEMO = """age,sex,dataset,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak
48,Female,A,asymptomatic,125,210,False,normal,155,False,0.4
61,Male,B,typical angina,140,245,True,normal,130,True,1.8
52,Male,A,non-anginal,132,?,False,normal,147,False,0.7
57,Female,B,atypical angina,128,224,False,normal,151,False,0.2
"""

from pathlib import Path
from hashlib import sha256
import joblib
import numpy as np

st.set_page_config(page_title="Studio Heart Disease", page_icon="🫀", layout="wide")
st.title("Studio Heart Disease")
page = st.sidebar.radio("Ruang kerja", ["Simulasi model", "Eksplorasi CSV"])

@st.cache_resource
def load_model():
    path = Path(__file__).with_name("heart_disease.joblib")
    if sha256(path.read_bytes()).hexdigest() != "52aad381256e22674613c51e72e45bc27087c0e8cb262732a13d790eea8fb981":
        raise ValueError("Gunakan model asli yang disertakan dalam paket.")
    return joblib.load(path)

if page == "Simulasi model":
    st.caption("RANDOM FOREST  /  SIMULASI SATU PROFIL")
    st.write("Masukkan hasil pemeriksaan, lalu lihat bagaimana model membagi skor di antara tiga kelas.")
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Model gagal dibuka. Periksa file model dan versi dependensi. Detail: {exc}")
        st.stop()
    prep = model.named_steps["preprocessing"]
    labels = {"age":"Usia (tahun)", "sex":"Jenis kelamin", "cp":"Tipe nyeri dada",
              "trestbps":"Tekanan darah saat istirahat (mmHg)", "chol":"Kolesterol (mg/dL)",
              "fbs":"Gula darah puasa > 120 mg/dL", "restecg":"Hasil EKG saat istirahat",
              "thalch":"Denyut jantung maksimum (denyut/menit)", "exang":"Angina saat aktivitas",
              "oldpeak":"Depresi segmen ST (oldpeak)"}
    translations = {"Male":"Laki-laki","Female":"Perempuan","True":"Ya","False":"Tidak",
                    "asymptomatic":"Tanpa gejala","atypical angina":"Angina atipikal",
                    "typical angina":"Angina tipikal","non-anginal":"Nyeri non-angina",
                    "normal":"Normal","lv hypertrophy":"Hipertrofi ventrikel kiri",
                    "st-t abnormality":"Kelainan ST-T"}
    categories = {}
    for group in ["nominal","ordinal"]:
        pipe = prep.named_transformers_[group]
        categories.update(zip(pipe.feature_names_in_, pipe.named_steps["encoder"].categories_))
    with st.form("profile", border=True):
        profile = {"dataset":np.nan}
        for col, title, fields in zip(st.columns(2), ["Profil dan gejala","Hasil pemeriksaan"],
                [["age","sex","cp","exang"],["trestbps","chol","fbs","restecg","thalch","oldpeak"]]):
            with col:
                st.subheader(title)
                for field in fields:
                    if field in categories:
                        profile[field] = st.selectbox(labels[field], categories[field], index=None,
                            placeholder="Pilih jawaban", format_func=lambda x: translations.get(str(x),str(x)))
                    else:
                        profile[field] = st.number_input(labels[field], value=None,
                            min_value=None if field == "oldpeak" else 0.0,
                            step=0.1 if field == "oldpeak" else 1.0)
        submit = st.form_submit_button("Hitung prediksi", type="primary")
    if submit:
        if any(value is None for value in profile.values()):
            st.info("Lengkapi semua isian untuk menghitung prediksi profil ini.")
        else:
            row = pd.DataFrame([profile], columns=model.feature_names_in_)
            scores = model.predict_proba(row)[0]
            st.subheader(f"Hasil simulasi · Kelas {model.classes_[scores.argmax()]}")
            for col, name, score in zip(st.columns(len(scores)), model.classes_, scores):
                col.metric(f"Skor kelas {name}", f"{score:.1%}")
                col.progress(float(np.clip(score,0,1)))
    st.info("Kelas 0, 1, dan 2 mengikuti label model. Definisi klinis kelas dan hasil validasi belum tersedia. Skor menunjukkan kecenderungan model untuk profil ini.")
    with st.expander("Bagaimana model menggunakan data?"):
        st.write("Pipeline menerima 11 kolom. Sepuluh variabel digunakan untuk prediksi. Kolom dataset diabaikan. Pilihan kategori pada formulir diambil dari encoder model.")
    st.stop()

st.subheader("Eksplorasi data Heart Disease")
file = st.sidebar.file_uploader("Unggah CSV", type="csv")
if file is None:
    st.info("Menampilkan empat baris data sintetis. Unggah CSV untuk menganalisis data Anda.")
try:
    df = pd.read_csv(file if file is not None else StringIO(DEMO), sep=None,
                     engine="python", dtype="string", na_values=["?"], encoding="utf-8-sig")
except (csv.Error, ValueError, UnicodeError, pd.errors.ParserError) as error:
    st.error(f"CSV gagal dibaca. Periksa header dan pemisah kolom. {error}")
    st.stop()

df.columns = df.columns.str.strip().str.lower()
missing = [key for key in COLUMNS if key not in df.columns]
if missing or df.columns.duplicated().any():
    st.error("Periksa nama kolom. Kolom wajib harus unik. Kolom yang belum ada: " + ", ".join(missing))
    st.stop()
df = df[COLUMNS].apply(lambda col: col.str.strip()).replace(["", "?"], pd.NA)
for key in NUMERIC:
    values = pd.to_numeric(df[key].str.replace(",", ".", regex=False), errors="coerce")
    values = values.replace([float("inf"), -float("inf")], float("nan"))
    invalid = (df[key].notna() & values.isna()).sum()
    if invalid:
        st.warning(f"{key} mengandung {invalid} nilai numerik tidak valid, ditampilkan sebagai kosong.")
    df[key] = values

for key in ["dataset", "sex", "cp"]:
    choices = st.sidebar.multiselect(f"Filter {key}", sorted(df[key].dropna().unique()))
    if choices:
        df = df[df[key].isin(choices)]
if df.empty:
    st.info("Tidak ada baris untuk ditampilkan. Periksa CSV atau sesuaikan filter.")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Jumlah baris", len(df))
middle.metric("Sumber dataset", df.dataset.nunique())
right.metric("Sel kosong", int(df.isna().sum().sum()))
plot, summary, data = st.tabs(["Distribusi", "Statistik", "Tabel data"])
with plot:
    field = st.selectbox("Variabel", COLUMNS)
    series = df[field].dropna()
    if series.empty:
        st.info("Seluruh nilai variabel ini kosong.")
    else:
        if field in NUMERIC and series.nunique() > 10:
            series = pd.cut(series.astype(float), bins=10)
        counts = series.value_counts(sort=False).sort_index()
        counts.index = counts.index.astype(str)
        st.bar_chart(counts.rename("Jumlah baris"), x_label=field, y_label="Jumlah baris")
with summary:
    st.subheader("Statistik variabel numerik")
    st.dataframe(df[NUMERIC].describe().T)
    st.subheader("Jumlah nilai kosong per kolom")
    st.dataframe(df.isna().sum().rename("Sel kosong"))
with data:
    st.dataframe(df, hide_index=True)
    st.download_button("Unduh data hasil filter", df.to_csv(index=False).encode("utf-8-sig"),
                       "heart_disease_filtered.csv", "text/csv")
