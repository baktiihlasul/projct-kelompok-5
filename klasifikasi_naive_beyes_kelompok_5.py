import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Konfigurasi tampilan Streamlit
st.set_page_config(page_title="Klasifikasi Naive Bayes", layout="wide")
st.title("📊 Klasifikasi Naive Bayes - Dataset Kejahatan")

# Upload file
uploaded_file = st.file_uploader("📁 Unggah file CSV Anda", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Pratinjau Dataset")
        st.dataframe(df.head())

        if st.checkbox("Tampilkan info dataset"):
            buffer = []
            df.info(buf=buffer)
            st.text('\n'.join(buffer))

        # Ambil kolom numerik untuk fitur dan target
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset harus memiliki minimal dua kolom numerik.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fitur = st.selectbox("🧮 Pilih kolom sebagai fitur (X)", numeric_cols)
            with col2:
                target_options = [col for col in numeric_cols if col != fitur]
                target = st.selectbox("🎯 Pilih kolom sebagai target (y)", target_options)

            # Siapkan data
            clean_df = df[[fitur, target]].dropna()
if len(clean_df) < len(df):
    st.warning(f"{len(df) - len(clean_df)} baris dihapus karena memiliki nilai kosong (NaN).")

X = clean_df[[fitur]].values
y_raw = clean_df[target].values

            # Diskretisasi target jika kontinu
            y = pd.cut(y_raw, bins=4, labels=[0, 1, 2, 3])

            # Split & standardisasi
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi
            st.subheader("📈 Hasil Evaluasi Model")
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Akurasi", value=f"{acc*100:.2f}%")
            with col2:
                st.text("Confusion Matrix (Tabel):")
                st.write(cm)

            st.text("Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())

            result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
            st.subheader("📋 Perbandingan Hasil Prediksi")
            st.dataframe(result_df)

            # Visualisasi
            st.subheader("📊 Visualisasi Hasil Klasifikasi")

            # Confusion Matrix - Heatmap
            st.markdown("### 🔥 Heatmap Confusion Matrix")
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel("Prediksi")
            ax1.set_ylabel("Aktual")
            st.pyplot(fig1)

            # Diagram Batang y_test vs y_pred
            st.markdown("### 📉 Diagram Batang: y_test vs y_pred")
            fig2, ax2 = plt.subplots()
            result_df.value_counts().unstack().plot(kind='bar', ax=ax2)
            ax2.set_title("Distribusi y_test vs y_pred")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file CSV terlebih dahulu untuk memulai.")
