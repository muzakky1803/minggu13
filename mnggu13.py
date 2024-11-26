# 1
# import streamlit as st

# st.write("Hello world")

# 2
# import streamlit as st

# st.header('st.button')
# if st.button('Saya Novan'):
#     st.write('Kenapa saya?')
# else:
#     st.write('Dadaaa')

# 3
# import streamlit as st

# # Menampilkan judul aplikasi
# st.title("this is the app title")

# # Menampilkan teks markdown
# st.markdown("### this is the markdown")
# st.markdown("this is the header")
# st.markdown("## this is the subheader")
# st.caption("this is the caption")

# # Menampilkan kode dengan sorotan
# st.code("x = 2021", language="python")

# # 4
# import streamlit as st

# # Checkbox
# if st.checkbox("yes"):
#     st.write("You selected Yes!")

# # Button
# if st.button("Click"):
#     st.write("Button clicked!")

# # Radio button untuk memilih gender
# st.write("Pick your gender")
# gender = st.radio("", ("Male", "Female"))
# st.write(f"You selected: {gender}")

# # Dropdown untuk memilih gender
# st.write("Pick your gender")
# selected_gender = st.selectbox("Select your gender", ["Male", "Female"])
# st.write(f"You picked: {selected_gender}")

# # Dropdown untuk memilih planet
# st.write("Choose a planet")
# planet = st.selectbox("Choose an option", ["Mercury", "Venus", "Earth", "Mars"])
# st.write(f"You selected: {planet}")

# # Slider untuk menilai
# st.write("Pick a mark")
# mark = st.slider("Bad ↔ Excellent", min_value=0, max_value=10, value=5)
# st.write(f"Mark: {mark}")

# # Slider untuk memilih angka
# st.write("Pick a number")
# number = st.slider("", min_value=0, max_value=50, value=25)
# st.write(f"Number: {number}")

# 5
# import streamlit as st

# # Input angka
# st.number_input("Pick a number", min_value=0, max_value=100, step=1)

# # Input email
# st.text_input("Email address")

# # Input tanggal
# st.date_input("Travelling date")

# # Input waktu
# st.time_input("School time")

# # Input teks area untuk deskripsi
# st.text_area("Description")

# # Input file untuk mengunggah foto
# st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

# # Pilih warna favorit
# st.color_picker("Choose your favourite color", "#ff00ff")

# 6
# import numpy as np
# import altair as alt
# import pandas as pd
# import streamlit as st

# # Header
# st.header('st.write Example')

# # Menampilkan teks
# st.write('Hello, *World!* :sunglasses:')

# # Menampilkan angka
# st.write(1234)

# # DataFrame pertama
# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })
# st.write(df)

# # Menampilkan DataFrame dengan teks tambahan
# st.write('Below is a DataFrame:', df, 'Above is a DataFrame.')

# # DataFrame kedua
# df2 = pd.DataFrame(
#     np.random.randn(200, 3),
#     columns=['a', 'b', 'c']
# )

# # Membuat grafik menggunakan Altair
# c = alt.Chart(df2).mark_circle().encode(
#     x='a',
#     y='b',
#     size='c',
#     color='c',
#     tooltip=['a', 'b', 'c']
# )

# # Menampilkan grafik
# st.write(c)

# 7
# import streamlit as st
# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=['x', 'y']
# )
# st.line_chart(df)

# chart_data = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=["x", "y",]
# )
# st.bar_chart(chart_data)

# chart_data = pd.DataFrame(
#     np.random.randn(20, 2),
#     columns=["a", "b",]
#     )
# st.area_chart(chart_data)

# 8
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image

# # Set judul aplikasi
# st.title("Roda Berputar")
# st.sidebar.title("Pilih Menu")

# # Menu di sidebar dengan key unik
# menu = st.sidebar.selectbox(
#     "Pilih Menu",
#     options=["Home", "Dataset", "Grafik"],
#     key="menu_selectbox"  # Menambahkan key unik
# )

# # Membaca dataset dari file CSV
# try:
#     df = pd.read_csv("CarPrice_Assignment.csv")  # Pastikan file CarPrice_Assignment.csv tersedia
# except FileNotFoundError:
#     st.error("Dataset 'CarPrice_Assignment.csv' tidak ditemukan! Pastikan file tersebut ada di folder yang sama.")

# # Home Page
# if menu == "Home":
#     st.header("Home Page")
    
#     # Menampilkan gambar
#     try:
#         image = Image.open("Mobil.jpg")  # Pastikan file Montor.jpg tersedia
#         st.image(image, caption="Contoh Gambar", use_container_width=True)  # Gunakan use_container_width
#     except FileNotFoundError:
#         st.error("Gambar 'Mobil.jpg' tidak ditemukan! Pastikan file tersebut ada di folder yang sama.")
    
#     # Menampilkan dataset
#     st.write("Dataset:")
#     st.dataframe(df)

# # Dataset Page
# elif menu == "Dataset":
#     st.header("Dataset Page")
    
#     # Menampilkan dataset
#     st.write("Dataset Mobil dan Harga:")
#     st.dataframe(df)

# # Grafik Page
# elif menu == "Grafik":
#     st.header("Menampilkan Grafik")
    
#     # Pastikan dataset memiliki kolom yang diperlukan
#     if "CarName" in df.columns and "price" in df.columns:
#         # Membuat grafik batang untuk harga rata-rata berdasarkan nama mobil
#         avg_price_per_car = df.groupby("CarName")["price"].mean().reset_index()

#         # Membuat grafik batang untuk harga rata-rata per mobil
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.bar(avg_price_per_car["CarName"], avg_price_per_car["price"], color="skyblue")
#         ax.set_title("Harga Rata-rata Mobil per Nama", fontsize=16)
#         ax.set_xlabel("Nama Mobil", fontsize=12)
#         ax.set_ylabel("Harga (Rupiah)", fontsize=12)
#         plt.xticks(rotation=90)
#         st.pyplot(fig)

#         # Membuat grafik hubungan antara harga dan horsepower
#         if "horsepower" in df.columns:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.scatter(df["horsepower"], df["price"], color="lightcoral")
#             ax.set_title("Hubungan antara Horsepower dan Harga Mobil", fontsize=16)
#             ax.set_xlabel("Horsepower", fontsize=12)
#             ax.set_ylabel("Harga (Rupiah)", fontsize=12)
#             st.pyplot(fig)

#         # Menampilkan grafik menggunakan Streamlit bar_chart
#         st.write("Harga Rata-rata per Mobil:")
#         st.bar_chart(avg_price_per_car.set_index("CarName")["price"])
#     else:
#         st.error("Dataset tidak memiliki kolom 'CarName' atau 'price'.")
        
# # Footer aplikasi
# st.sidebar.info("Aplikasi sederhana dengan Streamlit")

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
