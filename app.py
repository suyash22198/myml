# -*- coding: utf-8 -*-

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('myml.pkl','rb'))

dataset= pd.read_csv('CLASSIFICATION DATASET.csv')

X = dataset.iloc[:, 0:14].values
# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value="Female", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 5:6]) 
#Replacing missing data with the calculated constant value  
X[:, 5:6]= imputer.transform(X[:,5:6])
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value="Spain", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 6:7]) 
#Replacing missing data with the calculated constant value  
X[:, 6:7]= imputer.transform(X[:,6:7])
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', 
                        fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,1:5]) 
#Replacing missing data with the calculated mean value  
X[:, 1:5]= imputer.transform(X[:, 1:5])
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,7:13]) 
#Replacing missing data with the calculated mean value  
X[:, 7:13]= imputer.transform(X[:, 7:13])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict(age,cp,trestbps,chol, 
      fbs, Gender, Geography, 
      restecg ,thalach,exang, oldpeak,
      slope, ca, thal):
  output= model.predict(sc.transform([[age,cp,trestbps,chol, 
      fbs, Gender, Geography, 
      restecg ,thalach,exang, oldpeak,
      slope, ca, thal]]))
  print("Heart Disease", output)

  if output==[0]:
    prediction="It is HeartDisease Category 0"
  elif output==[1]:
    prediction="It is HeartDisease Category 1"
  elif output==[2]:
    prediction="It is HeartDisease Category 2"
  elif output==[3]:
    prediction="It is HeartDisease Category 3"
  elif output==[4]:
    prediction="It is HeartDisease Category 4"
    
  
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Deep Learning  Lab Experiment Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart disease category prediction")
    
   
    age=st.number_input('Insert Age(29-77)',29,77)
    cp=st.number_input('Insert cp value(1-4)',1,4)
    trestbps=st.number_input('Insert trestbps(94-200)',94,200)
    chol=st.number_input('Insert Chol(126-564) ',126,564)
    fbs = st.number_input("Insert fps(0-1)",0,1)
    Gender = st.number_input("Insert Gender male 1 female 0",0,1)
    Geography= st.number_input("Insert Geography France 0 Germany 1",0,1)
    restecg= st.number_input("Insert restecg",0,200)
    thalach = st.number_input("Insert thalach(0-187)",0,187)
    exang = st.number_input("Insert exang(0-6.2)",0.0,6.2)
    oldpeak = st.number_input("Insert oldpeak(0.0-3.6)",0.0,3.6)
    slope = st.number_input("Insert slope(0-3)",0,3)
    ca = st.number_input("Insert ca(0-7)",0,7)
    thal = st.number_input("Insert thal(0-6)",0,)
    
    resul=""
    if st.button("Predict"):
      result=predict(age,cp,trestbps,chol, 
      fbs, Gender, Geography, 
      restecg ,thalach,exang, oldpeak,
      slope, ca, thal)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Suyash Sharma")
      st.subheader("Department of Computer Engineering")

if __name__=='__main__':
  main()
