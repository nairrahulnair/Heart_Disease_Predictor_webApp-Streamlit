import streamlit as st
import pickle
import pandas as pd
import numpy as np

from PIL import Image

## Importing the Pickle file

pickle_in=open("model.pkl","rb")
model=pickle.load(pickle_in)

def predict_heart_disease(Sex, AgeGroup, ChestPainType, RestingBP, Cholesterol, FastingBS,
                          RestingECG, MaxHR, ExerciseAngina, OldPeak, ST_Slope):
    prediction=model.predict([[Sex, AgeGroup, ChestPainType, RestingBP, Cholesterol, FastingBS,
                               RestingECG, MaxHR, ExerciseAngina, OldPeak, ST_Slope]])
    if prediction == 0:
        print(prediction," patient doesn't have a heart disease")
    else:
        print(prediction,"patient has heart condition")
    return prediction

def main():
    st.title("Heart Disease Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Heart Disease Predictor App</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    Sex = st.text_input("Sex 0:F, 1:M")
    AgeGroup = st.text_input("AgeGroup 25-35:0, 36-45:1, 45-55:2, 55-65:3, 65-80:4")
    ChestPainType = st.text_input("Chest Pain ASY:0 ATA:1 NAP:2 TA:3")
    RestingBP = st.text_input("Resting BP")
    Cholesterol = st.text_input("Cholesterol")
    FastingBS = st.text_input("Fasting BS")
    RestingECG = st.text_input("RestingECG LVH:0 Normal:1 ST:2")
    MaxHR = st.text_input("MaxHR", "Type Here")
    ExerciseAngina = st.text_input("ExerciseAngina No:0 Yes:1")
    OldPeak = st.text_input("OldPeak", "type Here")
    ST_Slope = st.text_input("StSlope Down:0 Flat:1 UP:2")
    
    result=""
    
    if st.button("Predict"):
        result=predict_heart_disease(Sex, AgeGroup, ChestPainType, RestingBP, Cholesterol, FastingBS,
                                     RestingECG, MaxHR, ExerciseAngina, OldPeak, ST_Slope)
    
    
    st.success("The Output is {}".format(result))
        
    if st.button("About"):
        st.text("Happy to Announce First ML APP :)")
        st.text("Built with Streamlit")
  


if __name__=='__main__':
    main()
    