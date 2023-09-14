#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg
from pycaret.classification import predict_model
from pycaret.regression import predict_model as predict_model_reg
import numpy as np

st.set_page_config(page_title="AutoNickML")

st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
st.sidebar.title("AutoNickML")
choice = st.sidebar.radio("Navigation", ["Upload", "Modelling", "Download"])
st.sidebar.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset (CSV file)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df)

if "df" not in locals():
    st.warning("Please upload a dataset first.")
else:
    if choice == "Modelling":
        st.title("Model Building")

        # Ask the user for the target column
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        # Determine task type (regression or classification) based on the target column's data type
        if np.issubdtype(df[chosen_target].dtype, np.number):
            task_type = "Regression"
        else:
            task_type = "Classification"

        st.write(f"Task Type: {task_type}")

        # Preprocessing - Handle missing values and categorical/continuous transformations
        for column in df.columns:
            if column != chosen_target:
                if np.issubdtype(df[column].dtype, np.number):
                    impute_method = st.selectbox(f"Select imputation method for {column}", ["mean", "median", "mode"])
                    if impute_method == "mean":
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif impute_method == "median":
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    impute_method = st.selectbox(f"Select imputation method for {column}", ["most frequent", "add additional class"])
                    if impute_method == "most frequent":
                        df[column].fillna(df[column].value_counts().idxmax(), inplace=True)
                    else:
                        df[column].fillna("Missing", inplace=True)

        # Ask the user to select columns to drop
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        df.drop(columns=columns_to_drop, inplace=True)

        # Use PyCaret for model selection and evaluation
        if task_type == "Classification":
            setup_df = setup(data=df, target=chosen_target, silent=True)
            best_model = compare_models()
            st.write(best_model)
        else:
            setup_df = setup_reg(data=df, target=chosen_target, silent=True)
            best_model_reg = compare_models_reg()
            st.write(best_model_reg)

    if choice == "Download":
        
        pass


# In[ ]:




