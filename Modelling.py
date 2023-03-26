import streamlit as st

@st.cache_data
def get_data():
    california_housing = fetch_california_housing(as_frame=True)
    return california_housing.frame


@st.cache_data
def holdout_split():
    X=get_data().drop(['AveBedrms','MedHouseVal','Longitude'],axis=1)
    y=get_data()['MedHouseVal']
    return train_test_split(X,y,test_size=0.3,random_state=0)





header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()