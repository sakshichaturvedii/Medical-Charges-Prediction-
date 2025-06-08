import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    model = joblib.load('model/rf_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    feature_cols = joblib.load('model/feature_columns.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found.")
    st.stop() 
st.set_page_config(layout="centered", page_title="Medical Cost Prediction")

st.title("Medical Cost Prediction")
st.markdown("This app predicts medical insurance charges based on user inputs.")


st.header("Enter Your Details")

with st.form("medical_cost_form"):
    
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=1, step=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

    submitted = st.form_submit_button("Predict Medical Charges")


if submitted:
 
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    def feature_engineering(df):
        def bmi_group(b):
            if b < 18.5:
                return 'underweight'
            elif b < 25:
                return 'normal'
            elif b < 30:
                return 'overweight'
            else:
                return 'obese'

        df['bmi_group'] = df['bmi'].apply(bmi_group)

        df['age_group'] = pd.cut(df['age'],bins=[0, 18, 35, 50, 65, 100],labels=['child', 'young_adult', 'adult', 'senior', 'elder'],right=False) 

        df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region', 'age_group', 'bmi_group'], drop_first=True)

        processed_features = []
        for col in feature_cols:
            if col in df_encoded.columns:
                processed_features.append(df_encoded[col])
            else:
                processed_features.append(pd.Series([0] * len(df_encoded), name=col)) 

        df_final = pd.concat(processed_features, axis=1)
        df_final = df_final[feature_cols] 

        return df_final

    processed_df = feature_engineering(input_data)
    scaled_input = scaler.transform(processed_df)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Predicted Medical Charges")
    st.success(f"{prediction:.2f}")

    st.subheader("Your Inputs")
    st.write(input_data)

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    fig = px.bar(imp_df.sort_values('Importance', ascending=False), x='Importance', y='Feature', orientation='h',title="Impact of Features on Prediction")
    st.plotly_chart(fig)