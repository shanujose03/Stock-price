


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
with open('model.pkl','rb')as f:
  model=pickle.load(f)

import calendar
st.title('Google stock price prediction')
month=st.number_input("Month",min_value=1,max_value=12,value=1)
year=st.number_input("Year",min_value=2000,max_value=2100,value=2025)
if st.button('Predict'):
  days=calendar.monthrange(year,month)[1]
  predictions=[]
  dates=[]
  for day in range(1,days+1):
    input=pd.DataFrame([[day,month,year]],columns=['day','month','year'])
    price=model.predict(input)[0]
    predictions.append(price)
    dates.append(f"{day}-{month}-{year}")
  plt.figure(figsize=(10,10))
  plt.plot(dates,predictions,marker='o')
  plt.xlabel('Date')
  plt.ylabel('Predicted price')
  plt.xticks(rotation=45)
  st.pyplot(plt)
  plt.close()



