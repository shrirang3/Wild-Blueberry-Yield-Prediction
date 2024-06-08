import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import select_features_infogain_based, get_prediction

model=joblib.load('./Notebook/blueberry_yield_model.pkl')

st.set_page_config(page_title='Blueberry Yield Prediction', page_icon='🍇', layout='wide')

options_clonesize=['25', '12.5', '37.5', '20', '10']
options_bumbles=['0.38', '0.25', '0.117']
options_andrena=['0.5'  , '0.63' , '0.25' , '0.38' , '0.75' , '0.409']
options_osmia=['0.63' , '0.5'  , '0.75' , '0.25' , '0.38' , '0.058']
options_upperTRange=['71.9', '58.2', '79' , '64.7', '65.6']
options_lowerTRange=['50.8', '41.2', '55.9', '45.8', '45.3']
options_rain=['0.39', '0.56', '0.1' , '0.26', '0.06']


st.markdown("<h1 style='text-align: center; color: white;'>Blueberry Yield Prediction 🍇</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader('Enter the following features rounded to nearest of given options:')

        clonesize=st.selectbox('The average blueberry clone size in the field ', options_clonesize)
        bumbles=st.selectbox('Bumblebee density in the field (bees/m^2/min)', options_bumbles)
        andrena=st.selectbox('Andrena bee density in the field (bees/m^2/min)', options_andrena)
        osmia=st.selectbox('Osmia bee density in the field (bees/m^2/min)', options_osmia)
        upperTRange=st.selectbox('UpperTRange ', options_upperTRange)
        lowerTRange=st.selectbox('LowerTRange ', options_lowerTRange)
        rain=st.selectbox('Rain ', options_rain)

        submit=st.form_submit_button('Predict')

        if submit:
            clonesize=float(clonesize)
            bumbles=float(bumbles)
            andrena=float(andrena)
            osmia=float(osmia)
            upperTRange=float(upperTRange)
            lowerTRange=float(lowerTRange)
            rain=float(rain)

            data=pd.DataFrame({'clonesize': [clonesize], 'bumbles': [bumbles], 'andrena': [andrena], 'osmia': [osmia], 
                               'AverageOfUpperTRange': [upperTRange], 'AverageOfLowerTRange': [lowerTRange], 'AverageRainingDays': [rain]})
            
            pred = get_prediction(model=model, df=data)

            st.write(f"The predicted yield after regression analysis is:  {pred[0]}")

if __name__ == '__main__':
    main()
