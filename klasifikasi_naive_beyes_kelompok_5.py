pip install seaborn matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Klasifikasi Naive Bayes - Dataset Kejahatan")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV Anda", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Pratinjau Dataset")
        st.write(df.head())

        if st.checkbox("Tampilkan info dataset"):
            buffer = []
            df.info(buf=buffer)
            st.text('\n'.join(buffer))

        # Pilih fitur dan target
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset harus memiliki minimal dua kolom numerik.")
        else:
            fitur = st.selectbox("Pilih kolom sebagai fitur (X)", numeric_cols)
            target = st.selectbox("Pilih kolom sebagai target (y)", [col for col in numeric_cols if col != fitur])

            X = df[[fitur]].values
            y = df[target].values

            # Diskretisasi target jika kontinu
            y = pd.cut(y, bins=4, labels=[0, 1, 2, 3])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standarisasi
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi
            st.subheader("Hasil Evaluasi Model")
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            acc = accuracy_score(y_test, y_pred)

            st.write("Confusion Matrix (dalam bentuk tabel):")
            st.write(cm)

            st.write("Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())

            st.success(f"Akurasi Model: {acc * 100:.2f}%")

            result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
            st.subheader("Perbandingan Hasil Prediksi")
            st.dataframe(result_df)

            # === Visualisasi ===
            st.subheader("Visualisasi Hasil Klasifikasi")

            # Confusion Matrix Heatmap
            st.markdown("### Heatmap Confusion Matrix")
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel("Predicted Label")
            ax1.set_ylabel("True Label")
            ax1.set_title("Confusion Matrix")
            st.pyplot(fig1)

            # Bar Plot y_test vs y_pred
            st.markdown("### Diagram Batang: y_test vs y_pred")
            fig2, ax2 = plt.subplots()
            result_df.value_counts().unstack().plot(kind='bar', ax=ax2)
            ax2.set_title("Distribusi y_test vs y_pred")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
