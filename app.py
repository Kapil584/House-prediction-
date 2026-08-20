import streamlit as st
import pandas as pd
import joblib


# Load model
model = joblib.load("housing_model.pkl")

# Load column names
columns = joblib.load("columns.pkl")


st.title("🏠 Housing Price Prediction")

st.write("Enter house details to predict price")
area = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=20000
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10
)

stories = st.number_input(
    "Stories",
    min_value=1,
    max_value=10
)


mainroad = st.selectbox(
    "Main Road",
    ["yes","no"]
)

guestroom = st.selectbox(
    "Guest Room",
    ["yes","no"]
)

basement = st.selectbox(
    "Basement",
    ["yes","no"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["yes","no"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["yes","no"]
)

parking = st.number_input(
    "Parking",
    min_value=0,
    max_value=5
)

prefarea = st.selectbox(
    "Preferred Area",
    ["yes","no"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    [
        "furnished",
        "semi-furnished",
        "unfurnished"
    ]
)


# Convert input into dataframe

input_data = pd.DataFrame({

    "area":[area],
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "stories":[stories],

    "mainroad":[1 if mainroad=="yes" else 0],
    "guestroom":[1 if guestroom=="yes" else 0],
    "basement":[1 if basement=="yes" else 0],
    "hotwaterheating":[1 if hotwaterheating=="yes" else 0],
    "airconditioning":[1 if airconditioning=="yes" else 0],

    "parking":[parking],

    "prefarea":[1 if prefarea=="yes" else 0],

})


# Handle furnishing encoding

input_data["furnishingstatus_semi-furnished"] = (
    1 if furnishingstatus=="semi-furnished" else 0
)

input_data["furnishingstatus_unfurnished"] = (
    1 if furnishingstatus=="unfurnished" else 0
)


# Arrange columns same as training

input_data = input_data.reindex(
    columns=columns,
    fill_value=0
)


# Prediction

if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(
        f"Estimated House Price: {prediction[0]:,.0f}"
    )