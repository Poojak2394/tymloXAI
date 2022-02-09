import streamlit as st
from build import build_model
from build import Scatterplot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston,load_digits


title_container = st.sidebar.container()
image =  'tymlo.png'   
with title_container:
    st.sidebar.title("Explainable AI Toolkit")
    st.sidebar.image(image)
    
#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file",accept_multiple_files=False, type=["csv"],key='1')
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.header('Select target')
        target = st.sidebar.selectbox('Columns:', df.columns)
        submit=st.sidebar.button("Select")


#with st.sidebar.header(' Select Model'):
   # select = st.sidebar.radio("Classification: ", ('RandomForestClassifier','DecisionTreeClassifier'))#, 'KNeighborsClassifier','GaussianNB','DecisionTreeClassifier','SVC'


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

pressed=0

if uploaded_file is not None:
    #df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)  
    #st.sidebar.header('3.Select target')
    #target = st.sidebar.selectbox('Columns:', df.columns)
    #submit=st.sidebar.button("Select")
    #if submit:
    #st.write(target)
    if submit and pressed==0:        
        build_model(df,target)
        #Scatterplot(df, target)
        pressed==1
    elif pressed==1:
        build_model(df,target)
        #Scatterplot(df, target)
       

else:
        #Scatterplot(df, target)
        # Boston housing dataset
    data = pd.read_csv('Aer_test.csv')
    
    X = data.drop(['card'], axis=1)
    Y =  data['card']
    #st.write(Y.head())
    #st.write(X.head())
    df = pd.concat( [X,Y], axis=1 )
    st.markdown('This dataset is used as the example.')
    st.write(df.head(5))

    build_model(df,'')
    #Scatterplot(df, Y.name)


# Sidebar - Specify parameter settings
