import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.title("Klasifikasi Naive Bayes - Proporsi Korban Kejahatan")

uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.subheader("Pratinjau Dataset")
    st.write(dataset.head())

    if st.checkbox("Tampilkan informasi dataset"):
        buffer = []
        dataset.info(buf=buffer)
        s = '\n'.join(buffer)
        st.text(s)

    if not dataset.empty:
        # Label encoding pada kolom target
        le = LabelEncoder()
        dataset['Proporsi_Korban_Kejahatan_Persen'] = le.fit_transform(dataset['Proporsi_Korban_Kejahatan_Persen'])

        # Pemilihan fitur dan target
        x = dataset.iloc[:, -1:].values
        y = dataset.iloc[:, 1].values

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Standarisasi
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Klasifikasi Naive Bayes
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        # Evaluasi
        st.subheader("Hasil Evaluasi")
        cm = confusion_matrix(y_test, y_pred)
        st.text("Confusion Matrix:")
        st.write(cm)

        report = classification_report(y_test, y_pred, output_dict=True)
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        akurasi = accuracy_score(y_test, y_pred)
        st.success(f"Tingkat akurasi: {akurasi * 100:.2f}%")

        # Tampilkan data aktual vs prediksi
        ydata = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        st.subheader("Perbandingan y_test vs y_pred")
        st.dataframe(ydata)

        # Download file hasil prediksi
        hasil_excel = ydata.to_excel(index=False, engine='openpyxl')
        st.download_button(
            label="Unduh Hasil Prediksi (Excel)",
            data=hasil_excel,
            file_name='dataactualpred.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )