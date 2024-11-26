# 9 & 10 Machine Learning
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import pickle
# import os
# import streamlit as st

# # Memeriksa apakah file CSV ada di direktori yang benar
# if not os.path.exists('CarPrice_Assignment.csv'):
#     st.write("File 'CarPrice_Assignment.csv' tidak ditemukan!")
# else:
#     # Membaca data
#     df_mobil = pd.read_csv('CarPrice_Assignment.csv')

#     # Descriptive statistics
#     descriptive_stats = df_mobil.describe()
#     st.write("Descriptive Statistics:")
#     st.write(descriptive_stats)

#     # Menampilkan tipe data dari setiap kolom
#     st.write("Data Types:")
#     st.write(df_mobil.dtypes)

#     # Visualisasi distribusi harga mobil
#     st.write("Distribusi Harga Mobil:")
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.title('Car Distribution Plot')
#     sns.histplot(df_mobil['price'])
#     st.pyplot(plt)

#     # Distribusi jumlah mobil berdasarkan CarName
#     car_counts = df_mobil['CarName'].value_counts()
#     st.write("Distribusi Jumlah Mobil Berdasarkan CarName:")
#     plt.figure(figsize=(10, 6))
#     car_counts.plot(kind="bar")
#     plt.title("CarName Distribution")
#     plt.xlabel("CarName")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)
#     st.pyplot(plt)

#     # Menampilkan 10 nama mobil terbanyak
#     top_10_cars = df_mobil['CarName'].value_counts().head(10)
#     st.write("10 Nama Mobil Terbanyak:")
#     st.write(top_10_cars)

#     # Visualisasi 10 nama mobil terbanyak
#     st.write("Visualisasi 10 Nama Mobil Terbanyak:")
#     plt.figure(figsize=(10, 6))
#     car_counts.head(10).plot(kind="bar", color="blue")
#     plt.title("10 Nama Mobil Terbanyak pada Dataset", fontsize=16)
#     plt.xlabel("Nama Mobil", fontsize=12)
#     plt.ylabel("Jumlah", fontsize=12)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.tight_layout()
#     st.pyplot(plt)

#     # Membuat WordCloud dari nama mobil
#     car_names = " ".join(df_mobil['CarName'])
#     wordcloud = WordCloud(
#         width=800, height=400,
#         background_color='white',
#         colormap='viridis',
#         random_state=42
#     ).generate(car_names)

#     st.write("Word Cloud of Car Names:")
#     plt.figure(figsize=(12, 6))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title("Word Cloud of Car Names", fontsize=16)
#     st.pyplot(plt)

#     # Scatter plot harga mobil vs highwaympg
#     st.write("Scatter Plot Harga Mobil vs Highwaympg:")
#     plt.scatter(df_mobil['highwaympg'], df_mobil['price'])
#     plt.xlabel('highwaympg')
#     plt.ylabel('price')
#     st.pyplot(plt)

#     # Persiapan data untuk model regresi
#     x = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
#     y = df_mobil['price']

#     # Membagi data menjadi data latih dan data uji
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#     # Membuat dan melatih model regresi linear
#     model_regresi = LinearRegression()
#     model_regresi.fit(x_train, y_train)

#     # Prediksi harga mobil
#     model_regresi_pred = model_regresi.predict(x_test)

#     # Visualisasi harga mobil yang diprediksi dan harga sesungguhnya
#     st.write("Prediksi vs Harga Sebenarnya:")
#     plt.scatter(x_test.iloc[:, 0], y_test, label='Actual Price', color='blue')
#     plt.scatter(x_test.iloc[:, 0], model_regresi_pred, label='Predicted Prices', color='red')
#     plt.xlabel('highwaympg')
#     plt.ylabel('price')
#     plt.legend()
#     st.pyplot(plt)

#     # Prediksi harga untuk input baru
#     X = np.array([[32, 2338, 75]])
#     harga_X = model_regresi.predict(X)
#     harga_X_int = int(harga_X[0])
#     st.write(f'Harga Prediksi untuk Input: {harga_X_int}')

#     # Menghitung error
#     mae = mean_absolute_error(y_test, model_regresi_pred)
#     st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

#     mse = mean_squared_error(y_test, model_regresi_pred)
#     st.write(f'Mean Square Error (MSE): {mae:.2f}')

#     rmse = np.sqrt(mse)
#     st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')

#     # Menyimpan model regresi menggunakan pickle
#     filename = 'model_prediksi_harga_mobil.sav'
#     pickle.dump(model_regresi, open(filename, 'wb'))
#     st.write("Model berhasil disimpan!")

# 11 mempercantik tampilan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import streamlit as st

# Memeriksa apakah file CSV ada di direktori yang benar
if not os.path.exists('CarPrice_Assignment.csv'):
    st.write("File 'CarPrice_Assignment.csv' tidak ditemukan!")
else:
    # Membaca data
    df_mobil = pd.read_csv('CarPrice_Assignment.csv')

    # Menambahkan Header Aplikasi
    st.title("Prediksi Harga Mobil dengan Regresi Linear")
    st.markdown("""
    Aplikasi ini digunakan untuk menganalisis harga mobil berdasarkan fitur-fitur seperti `highwaympg`, `curbweight`, dan `horsepower`.
    Data yang digunakan adalah dataset mobil yang mencakup berbagai informasi mengenai harga mobil.
    """)

    # Kolom untuk statistik deskriptif
    st.header("Statistik Deskriptif Dataset")
    st.write(df_mobil.describe())

    # Kolom untuk tipe data setiap kolom
    st.header("Tipe Data Kolom")
    st.write(df_mobil.dtypes)

    # Membuat kolom untuk visualisasi dan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Harga Mobil")
        plt.figure(figsize=(10, 4))
        sns.histplot(df_mobil['price'])
        st.pyplot(plt)

    with col2:
        st.subheader("Distribusi Jumlah Mobil Berdasarkan Nama")
        car_counts = df_mobil['CarName'].value_counts()
        plt.figure(figsize=(10, 6))
        car_counts.plot(kind="bar", color="lightblue")
        plt.title("CarName Distribution")
        plt.xlabel("CarName")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Menampilkan 10 mobil terpopuler
    st.header("10 Nama Mobil Terpopuler")
    top_10_cars = df_mobil['CarName'].value_counts().head(10)
    st.write(top_10_cars)

    # Kolom untuk WordCloud
    st.subheader("Word Cloud Nama Mobil")
    car_names = " ".join(df_mobil['CarName'])
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        random_state=42
    ).generate(car_names)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Kolom Scatter Plot Harga Mobil vs Highwaympg
    st.subheader("Harga Mobil vs Highwaympg")
    plt.figure(figsize=(10, 6))  # Memperbesar ukuran grafik

    # Menambahkan grid dan transparansi pada titik
    plt.scatter(df_mobil['highwaympg'], df_mobil['price'], color='purple', alpha=0.5)

    # Menambahkan label dan judul yang lebih jelas
    plt.title('Harga Mobil vs Highway MPG', fontsize=16, weight='bold')
    plt.xlabel('Highway MPG', fontsize=12)
    plt.ylabel('Harga Mobil (USD)', fontsize=12)

    # Menambahkan grid untuk memudahkan pembacaan
    plt.grid(True, linestyle='--', alpha=0.7)

    # Menampilkan grafik dengan ukuran yang lebih besar
    st.pyplot(plt)

    # Model regresi linear
    st.header("Model Prediksi Harga Mobil")
    st.markdown("""
    Model ini menggunakan regresi linear untuk memprediksi harga mobil berdasarkan tiga fitur: 
    `highwaympg`, `curbweight`, dan `horsepower`.
    """)

    # Persiapan data untuk model regresi
    x = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
    y = df_mobil['price']

    # Membagi data menjadi data latih dan data uji
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Membuat dan melatih model regresi linear
    model_regresi = LinearRegression()
    model_regresi.fit(x_train, y_train)

    # Prediksi harga mobil
    model_regresi_pred = model_regresi.predict(x_test)

    # Visualisasi harga mobil yang diprediksi dan harga sesungguhnya
    st.subheader("Prediksi vs Harga Sebenarnya")
    plt.scatter(x_test.iloc[:, 0], y_test, label='Actual Price', color='blue')
    plt.scatter(x_test.iloc[:, 0], model_regresi_pred, label='Predicted Prices', color='red')
    plt.xlabel('Highway MPG')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Input prediksi harga
    st.subheader("Prediksi Harga Mobil Baru")
    st.markdown("""
    Masukkan nilai `highwaympg`, `curbweight`, dan `horsepower` untuk mendapatkan prediksi harga mobil.
    """)
    
    # Input dari pengguna
    highwaympg_input = st.slider("Highway MPG", 10, 50, 30)
    curbweight_input = st.slider("Curbweight", 1500, 5000, 3000)
    horsepower_input = st.slider("Horsepower", 50, 300, 100)

    # Prediksi harga berdasarkan input pengguna
    X = np.array([[highwaympg_input, curbweight_input, horsepower_input]])
    harga_X = model_regresi.predict(X)
    harga_X_int = int(harga_X[0])

    st.write(f'Harga Prediksi untuk Mobil dengan Fitur Tersebut: ${harga_X_int}')

    # Menghitung error
    st.subheader("Evaluasi Model")
    mae = mean_absolute_error(y_test, model_regresi_pred)
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

    mse = mean_squared_error(y_test, model_regresi_pred)
    st.write(f'Mean Square Error (MSE): {mse:.2f}')

    rmse = np.sqrt(mse)
    st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')

    # Menyimpan model regresi menggunakan pickle
    filename = 'model_prediksi_harga_mobil.sav'
    pickle.dump(model_regresi, open(filename, 'wb'))
    st.write("Model berhasil disimpan!")
