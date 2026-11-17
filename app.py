
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Application title
st.title("🚢 Titanic Survival Prediction")
st.write("Predict the survival probability of a Titanic passenger.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

# User inputs
st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Gender",
    ["female", "male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    1, 80, 30
)

fare = st.slider(
    "Ticket Fare",
    0, 500, 30
)

# Prediction button
if st.button("Predict Survival", type="primary"):

    try:
        model = load_model()

        # Prepare input data
        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=["pclass", "sex", "age", "fare"]
        )

        # Make prediction
        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]

        # Display result
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.1%}"
        )

        if prediction == 1:
            st.success("The model predicts that the passenger may have survived.")
        else:
            st.error("The model predicts that the passenger may not have survived.")

    except FileNotFoundError:
        st.error("Model file is missing. Please add model.joblib.")

    except (ValueError, KeyError) as e:
        st.error(f"Invalid input or model error: {e}")

# Project information
st.divider()

st.caption(
    "This educational model uses a historical Titanic passenger dataset. "
    "It was trained using a Random Forest Classifier. "
    "Predictions are estimates, not historical verification or advice."
)
