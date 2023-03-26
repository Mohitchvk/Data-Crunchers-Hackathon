import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def get_data():
    df=pd.read_csv("data.csv", index_col=False)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


# @st.cache_data
# def holdout_split():
#     X=get_data().drop(['AveBedrms','MedHouseVal','Longitude'],axis=1)
#     y=get_data()['MedHouseVal']
#     return train_test_split(X,y,test_size=0.3,random_state=0)

header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()

numerical_cols=['Population',
       'DrugNarcoticOffenses', 'DrugNarcoticViolations',
       'DrugEquipmentViolations', 'GamblingOffenses', 'BettingWagering',
       'OperatingPromotingAssistingGambling', 'GamblingEquipmentViolations',
       'SportsTampering', 'PornographyObsceneMaterial', 'ProstitutionOffenses',
       'Prostitution', 'AssistingorPromotingProstitution',
       'WeaponLawViolations']

with header:
    st.title('FBI crime report analysis!!...')
    st.write(" ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write(" ")

with dataset:
    st.header('FBI Crime Report Data')
    st.text('This dataset is available from Kaggle')
    st.write(get_data().head())
    st.write(" ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write(" ")


with eda:
    st.header('Here we will explore the Data from the Dataset and tune the data to make it Linear model friendly')
    st.write('Checking For null values')
    null_percentages = get_data().isnull().sum() / len(get_data()) * 100
    null_percentages=pd.DataFrame(null_percentages)
    null_percentages.columns=['percentages of null values']
    st.write(null_percentages.sort_values(by='percentages of null values'))

    sel_col, disp_col = st.columns(2)
    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.subheader('Check For the Distribution of a predictors')
    inp_feature = sel_col.selectbox('Select a predictor for Box plot', options = numerical_cols)
    fig, ax = plt.subplots()
    get_data().boxplot(column=[inp_feature])
    disp_col.pyplot(fig)

    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write(" ")
    sel_col.write(" ")
    sel_col.write("   ")
    # sel_col.subheader('Check For the Distribution of a predictors')
    inp_feature = sel_col.selectbox('Select a predictor for hist plot', options = numerical_cols)
    fig, ax = plt.subplots()
    get_data()[inp_feature].hist()
    disp_col.pyplot(fig)



