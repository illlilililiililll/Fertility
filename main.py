import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prophet
import matplotlib.font_manager as fm

path = 'fonts/font.ttf'
fm.fontManager.addfont(path)
plt.rc('font', family=fm.FontProperties(fname=path).get_name())

df = pd.read_csv('new.csv')

age_list = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
select = st.sidebar.select_slider("Age", age_list)
age = str(age_list.index(select) + 1)

st.sidebar.title("")

reg = st.sidebar.selectbox("Region", ['전국', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', '세종특별자치시', '경기도', 
                                '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도', '제주특별자치도'])

reg = "전국" if reg is None else reg

col = ['행정구역별']
col += [f'{year}.{age}' for year in range(2000, 2023)]
df = df[col]

df.columns = ['Region', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', 
                         '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', 
                         '2017', '2018', '2019', '2020', '2021', '2022']

df = df.drop(0)

df.set_index('Region', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

st.title("Total Fertility Rate Graph")
plt.figure(figsize=(15, 10))
for region in df.index:
    if region in reg:
        plt.plot(df.columns, df.loc[region], label=region)
    else:
        plt.plot(df.columns, df.loc[region], label=region, alpha=0.3)

plt.title(f'Total Fertility Rate for Age Group {select}')
plt.xlabel('Year')
plt.ylabel('Fertility Rate (per 1000)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

region_df = df.loc[reg].reset_index()
region_df.columns = ['ds', 'y']
region_df['ds'] = pd.to_datetime(region_df['ds'], format='%Y')

m = prophet.Prophet()
m.fit(region_df)

future = m.make_future_dataframe(periods=10, freq='Y')
forecast = m.predict(future)

# Convert 'ds' in forecast to year string format for proper plotting
forecast['ds'] = forecast['ds'].dt.year.astype(str)

plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color="red")

plt.legend()
st.pyplot(plt)
