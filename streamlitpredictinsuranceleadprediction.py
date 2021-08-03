#!/usr/bin/env python
# coding: utf-8

# # Predict insurance charges based on a person's attributes

# In[1]:


import pickle
import numpy as np


# In[2]:


# load the model from disk
loaded_model = pickle.load(open('streamlit_insurance_leadprediction.pkl', 'rb'))


# In[3]:


import streamlit as st


# In[4]:


# Creating the Titles and Image
st.title("Insurance Lead prediction")
st.header("A model to predict whether the person will be interested in their proposed Health plan/policy given the information about: a)Demographics (city, age, region etc.  b)Information regarding holding policies of the customer c)Recommended Policy Information. ")




# In[5]:


import pandas as pd
def load_data():
    df = pd.DataFrame({'Accomodation': ['Owned','Rented'],
                       'IsMarried': ['Yes', 'No'],
                       'InsuranceType': ['Individual', 'Joint']}) 
    return df
df = load_data()


# In[6]:


import pandas as pd
def load_data():
    df1 = pd.DataFrame({'HealthIndicator' : ['X1' ,'X2' ,'X3' ,'X4','X5' ,'X6' ,'X7' ,'X8','X9']}) 
    return df1
df1 = load_data()


# In[7]:


# Take the users input
Upper_Age = st.slider("What is your Upper Age Limit?", 0, 100)
Lower_Age = st.slider("What is your Lower Age Limit?", 0, 100)
Holding_Policy_Duration=st.slider("Select your Holding Policy Duration?", 1, 15)
Holding_Policy_Type=st.slider("Select your Holding Policy Type?", 1, 4)
Reco_Policy_Premium=st.slider("Choose Recommended Policy Premium?", 0, 80000)
Reco_Policy_Cat=st.slider("Choose Recommended Policy Category?", 1, 25)

Reco_Insurance_Type=st.selectbox("Select Recommended Insurance Type", df['InsuranceType'].unique())
Accomodation_Type = st.selectbox("Select Accomodation Type", df['Accomodation'].unique())

Is_Spouse=st.selectbox("Are the customers are married to each other",df['IsMarried'].unique())
Health_Indicator=st.selectbox("Health Indicator",df1['HealthIndicator'].unique())


# converting text input to numeric to get back predictions from backend model.
if Reco_Insurance_Type == 'Individual':
    Reco_Insurance_Type_Individual = 1
    Reco_Insurance_Type_Joint=0   
else:
    Reco_Insurance_Type_Individual=0
    Reco_Insurance_Type_Joint = 1

# converting text input to numeric to get back predictions from backend model.
if Accomodation_Type == 'Owned':
    Accomodation_Type_Owned = 1
    Accomodation_Type_Rented=0
    
else:
    Accomodation_Type_Rented=1
    Accomodation_Type_Owned = 0

# converting text input to numeric to get back predictions from backend model.
if Is_Spouse == 'Yes':
    Is_Spouse_Yes = 1
    Is_Spouse_No=0
    
else:
    Is_Spouse_Yes=0
    Is_Spouse_No = 1
    
    

    
if Health_Indicator == 'X1':
    Health_Indicator_X1=1
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X2':
    Health_Indicator_X1=0
    Health_Indicator_X2=1
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X3':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=1
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X4':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=1
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X5':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=1
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X6':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=1
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X7':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=1
    Health_Indicator_X8=0
    Health_Indicator_X9=0
elif Health_Indicator == 'X8':
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=1
    Health_Indicator_X9=0
else:
    Health_Indicator_X1=0
    Health_Indicator_X2=0
    Health_Indicator_X3=0
    Health_Indicator_X4=0
    Health_Indicator_X5=0
    Health_Indicator_X6=0
    Health_Indicator_X7=0
    Health_Indicator_X8=0
    Health_Indicator_X9=1
    

# store the inputs
features = [Upper_Age, Lower_Age, Holding_Policy_Duration, Holding_Policy_Type, Reco_Policy_Cat, Reco_Policy_Premium,
           Accomodation_Type_Owned,Accomodation_Type_Rented,Reco_Insurance_Type_Individual,Reco_Insurance_Type_Joint,
           Is_Spouse_No,Is_Spouse_Yes,Health_Indicator_X1,Health_Indicator_X2, Health_Indicator_X3, Health_Indicator_X4,
           Health_Indicator_X5, Health_Indicator_X6, Health_Indicator_X7,Health_Indicator_X8, Health_Indicator_X9]
# convert user inputs into an array fr the model

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


# In[8]:


if st.button('Predict'):           # when the submit button is pressed
    prediction =  loaded_model.predict(final_features)
    #st.balloons()
    if (prediction[0]) == 1:
        st.success('The Customer can be a Lead.')
    else:
        st.success('The Customer can not be a Lead.')

    
    

