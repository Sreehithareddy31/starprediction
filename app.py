import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model from the pickle file
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

star_colors = ['Red', 'Blue', 'Yellow', 'White', 'Orange']  
spectral_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']  
label_encoders = {
    'Star color': LabelEncoder().fit(star_colors),
    'Spectral Class': LabelEncoder().fit(spectral_classes)
}

st.markdown(
    """
    <style>
    .stApp {
        background-color: beige;  
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Star Type Prediction")
st.image("starimage.jpg", caption="Star Image", use_column_width=True)
temperature = st.number_input("Temperature (K)", value=5000)
luminosity = st.number_input("Luminosity (L/Lo)", value=1.0)
radius = st.number_input("Radius (R/Ro)", value=1.0)
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", value=4.8)
star_color = st.selectbox("Star Color", options=star_colors)
spectral_class = st.selectbox("Spectral Class", options=spectral_classes)

if st.button("Predict"):
    input_data = {
        'Temperature (K)': temperature,
        'Luminosity(L/Lo)': luminosity,
        'Radius(R/Ro)': radius,
        'Absolute magnitude(Mv)': absolute_magnitude,
        'Star color': star_color,
        'Spectral Class': spectral_class
    }
    input_df = pd.DataFrame([input_data])
    for column in ['Star color', 'Spectral Class']:
        input_df[column] = label_encoders[column].transform(input_df[column])
    prediction = model.predict(input_df)
    st.write(f"Predicted Star Type: {prediction[0]}")




