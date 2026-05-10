import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("### Machine Learning Web Application")

st.write(
    "This application predicts whether a person has Parkinson's Disease "
    "using Machine Learning techniques."
)

# Sample Dataset
data = {
    'MDVP:Fo(Hz)': [119.992, 122.400, 116.682, 116.676, 116.014],
    'MDVP:Fhi(Hz)': [157.302, 148.650, 131.111, 137.871, 141.781],
    'MDVP:Flo(Hz)': [74.997, 113.819, 111.555, 111.366, 110.655],
    'status': [1, 1, 0, 0, 1]
}

# Create DataFrame
parkinsons_data = pd.DataFrame(data)

# Split Features and Labels
X = parkinsons_data.drop('status', axis=1)
Y = parkinsons_data['status']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Sidebar
st.sidebar.header("About Project")
st.sidebar.info(
    "Developed using Python, Streamlit, and Machine Learning."
)

# Input Section
st.subheader("Enter Patient Voice Measurements")

fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
fhi = st.number_input("MDVP:Fhi(Hz)", value=150.0)
flo = st.number_input("MDVP:Flo(Hz)", value=75.0)

# Prediction Button
if st.button("Predict Result"):

    input_data = np.asarray([fo, fhi, flo]).reshape(1, -1)

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ The person has Parkinson's Disease")
    else:
        st.success("✅ The person does not have Parkinson's Disease")

# Footer
st.markdown("---")
st.markdown("### 👨‍💻 Developed by Siva Narasimha Varma Chinta")
