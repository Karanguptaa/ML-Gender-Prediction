import streamlit as st
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('gender_model.sav', 'rb'))

def check(input_data):
    array_input = np.array(input_data)
    reshaped_input = array_input.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)
    return prediction

def main():
    st.title("Gender Prediction")

    long_hair = st.number_input("Long Hair")
    forehead_width_cm = st.number_input("Forehead CM")
    forehead_height_cm = st.number_input("Forehead Height")
    nose_wide = st.number_input("Nose Width")
    nose_long = st.number_input("Nose Long")
    lips_thin = st.number_input("Lips Thin")
    distance_nose_to_lip_long = st.number_input("Distance Between Nose to Lip long")

    pred = ""
    if st.button("Click Here for Gender Prediction"):
        prediction = check([long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long])

        if prediction[0] == 0:
            pred = 'Female'
        else:
            pred = 'Male'

    st.success(f"To Identify Gender is {pred}")

if __name__ == '__main__':
    main()
