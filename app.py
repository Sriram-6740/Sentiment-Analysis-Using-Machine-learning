import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os
import nltk
nltk.download('stopwords')
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#vs = SentimentIntensityAnalyzer()

#pick1 = {'vectorizer': vectorizer,
         #'model1': model
         #}
#pickle.dump(pick1, open('models'+".p", "wb"))
st.title('SENTIMENT ANALYSIS')
text = st.text_input("Enter your sentence")

if text is not None:
  st.text(text)
def predict_res(text):
  model = pickle.load(open('models.p','rb'))
  text_feature = model['vectorizer'].transform([text])
  model1 = model['model1'].predict(text_feature)
  return model1
  

if st.button('PREDICT'):
  st.write('Result....')
    
    #test_feature = vectorizer.transform(['awesome boy'])
  st.title(predict_res(text))

    

    
