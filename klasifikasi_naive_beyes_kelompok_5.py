import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.title("Klasifikasi Naive Bayes - Dataset Kejahatan")

# Upload file dari user
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

        # Pilih kolom numerik sebagai fitur dan target
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset harus memiliki minimal dua kolom numerik.")
        else:
            fitur = st.selectbox("Pilih kolom sebagai fitur (X)", numeric_cols)
            target = st.selectbox("Pilih kolom sebagai target (y)", [col for col in numeric_cols if col != fitur])

            X = df[[fitur]].values
            y = df[target].values

            # Diskretisasi target jika berupa nilai kontinu
            y = pd.cut(y, bins=4, labels=[0, 1, 2, 3])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standarisasi fitur
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Model Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi model
            st.subheader("Hasil Evaluasi Model")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

            st.write("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Akurasi Model: {acc * 100:.2f}%")

            # Hasil prediksi
            st.subheader("Perbandingan Hasil Prediksi")
            result = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
            st.dataframe(result)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca atau memproses file: {e}")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
