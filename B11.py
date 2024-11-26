import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df_mobil = pd.read_csv('CarPrice_Assignment.csv')

# Load the trained model
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Streamlit Title
st.title("Car Price Prediction and Visualization")

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Price Prediction", "Data Exploration", "Visualizations"]
)

# Home Page
if menu == 'Home':
    st.header("Welcome to the Car Price Prediction App!")
    st.write(
        """
        This app allows you to predict the price of a car based on features like:
        - Highway-MPG
        - Curbweight
        - Horsepower
        
        You can also explore data insights, visualizations, and make predictions!
        """
    )

# Price Prediction Page
elif menu == 'Price Prediction':
    st.header('Predict Car Price')

    # Input fields for prediction
    highwaympg = st.number_input('Highway-mpg', min_value=0, max_value=100, value=30)
    curbweight = st.number_input('Curbweight', min_value=500, max_value=5000, value=2000)
    horsepower = st.number_input('Horsepower', min_value=0, max_value=1000, value=150)

    if st.button('Predict'):
        # Make prediction using the input values
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

        # Convert prediction to string
        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])

        # Format and display the result
        harga_mobil_formatted = f"$ {harga_mobil_float:,.2f}"
        st.success(f"Predicted Car Price: {harga_mobil_formatted}")

# Data Exploration Page
elif menu == 'Data Exploration':
    st.header('Dataset Overview')
    # Load the dataset
    st.write(df_mobil)

    # Check for missing values
    missing_values = df_mobil.isnull().sum()
    st.subheader("Missing Values")
    st.write(missing_values)

    # Descriptive Statistics
    st.subheader('Descriptive Statistics')
    st.write(df_mobil.describe())

# Visualizations Page
elif menu == 'Visualizations':
    st.header('Visualizations')

    # Visualization 1: Car Price Distribution
    st.subheader("Car Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_mobil['price'], kde=True, color='blue', ax=ax)
    ax.set_xlabel('Price')
    st.pyplot(fig)

    # Visualization 2: Car Name Distribution
    st.subheader("Car Name Distribution")
    car_counts = df_mobil['CarName'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    car_counts.plot(kind="bar", color='green', ax=ax)
    ax.set_title("Car Name Distribution")
    ax.set_xlabel("Car Name")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 3: Top 10 Car Names
    st.subheader("Top 10 Car Names")
    top_10_cars = car_counts.head(10)
    st.write(top_10_cars)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_cars.plot(kind="bar", color='skyblue', ax=ax)
    ax.set_title("Top 10 Car Names Distribution")
    ax.set_xlabel("Car Name")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 4: Word Cloud of Car Names
    st.subheader("Word Cloud of Car Names")
    car_names = " ".join(df_mobil['CarName'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(car_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of Car Names", fontsize=16)
    st.pyplot(fig)

    # Visualization 5: Scatter Plot - Highway MPG vs Price
    st.subheader("Highway MPG vs Price")
    fig, ax = plt.subplots()
    ax.scatter(df_mobil['highwaympg'], df_mobil['price'], color='orange', alpha=0.7)
    ax.set_xlabel('Highway MPG')
    ax.set_ylabel('Price')
    ax.set_title('Highway MPG vs Price')
    st.pyplot(fig)

    # Train the Linear Regression model (for internal visualizations)
    X = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
    y = df_mobil['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_regresi = LinearRegression()
    model_regresi.fit(X_train, y_train)

    # Predict using the model
    model_regresi_pred = model_regresi.predict(X_test)

    # Visualization 6: Actual vs Predicted Prices
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test['highwaympg'], y_test, label='Actual Prices', color='blue', alpha=0.6)
    ax.scatter(X_test['highwaympg'], model_regresi_pred, label='Predicted Prices', color='red', alpha=0.6)
    ax.set_xlabel('Highway MPG')
    ax.set_ylabel('Price')
    ax.legend()
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)

    # Model Evaluation
    mae = mean_absolute_error(y_test, model_regresi_pred)
    mse = mean_squared_error(y_test, model_regresi_pred)
    rmse = np.sqrt(mse)

    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Save the model
    filename = 'model_prediksi_harga_mobil.sav'
    pickle.dump(model_regresi, open(filename, 'wb'))
    st.write(f"Model saved as {filename}")