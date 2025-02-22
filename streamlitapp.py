import pandas as pd
import streamlit as st 
import pickle, joblib

ridge_model = pickle.load(open("ridge_model.pkl","rb"))
lasso_model = pickle.load(open("lasso_model.pkl","rb"))

scale = joblib.load('standard_scaler')


def predict_diabetes(age, sex, bmi, bp, s1 , s2, s3, s4, s5, s6):
    data = pd.DataFrame([{'age': age, 'sex': sex, 'bmi': bmi, 'bp': bp, 's1': s1,'s2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6}])
    scaled_data = scale.transform(data)
    ridge_predict = ridge_model.predict(scaled_data)[0]
    lasso_predict = lasso_model.predict(scaled_data)[0]

    return ridge_predict, lasso_predict


def main():
    st.title("Diabeties prediction")

    age = st.number_input("age",min_value=0, max_value=120, step=1)
    sex = st.number_input("sex", min_value=0, max_value=1, step=1)
    bmi = st.number_input("bmi", min_value=0.0, max_value=100.0, step=0.1)
    bp = st.number_input("bp", min_value=0.0, max_value=200.0, step=0.1)
    s1 = st.number_input("s1",  min_value=0.0, step=0.1)
    s2 = st.number_input("s2", min_value=0.0, step=0.1)
    s3 = st.number_input("s3", min_value=0.0, step=0.1)
    s4 = st.number_input("s4",min_value=0.0, step=0.1)
    s5 = st.number_input("s5",min_value=0.0, step=0.1)
    s6 = st.number_input("s6", min_value=0.0, step=0.1)

    if st.button("Predict"):
        ridge_result, lasso_result = predict_diabetes(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
        st.success(f"Ridge Prediction: {ridge_result:.2f}")
        st.success(f"Lasso Prediction: {lasso_result:.2f}")


if __name__=="__main__":
    main()