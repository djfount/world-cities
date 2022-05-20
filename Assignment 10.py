import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('seaborn')


st.title('housing data by danny fountain')
df_housing = pd.read_csv('housing.csv')

p_filter = st.slider('house values', 0,500001, 200000) 

location_filter = st.sidebar.multiselect(
    'location types',
    df_housing.ocean_proximity.unique(),
    df_housing.ocean_proximity.unique())

ih_filter=st.sidebar.radio(
    'choose income',
    ('L','M','H'))
df_housing = df_housing[df_housing.median_house_value <= price_filter]

df_housing = df_housing[df_housing.ocean_proximity.isin(location_filter)]

if ih_filter =='L':
    df_housing = df_housing[df_housing.median_income <=2.5]
elif ih_filter =='M':
    df_housing = df_housing[(df_housing.median_income>2.5)&(df.median_income<4.5)]
else: 
    df_housing = df_housing[df_housing.median_income >=4.5]
    
st.map(df_housing)

st.subheader('Cxx:')
st.map(df_housing)

st.subheader('Total Population by Country')
fig, ax = plt.subplots()
df_housing.median_house_value.hist(bins=30, ax=ax)
st.pyplot(fig)

