import streamlit as st
import pandas as pd

st.set_page_config(page_title='Home', layout='wide')

st.markdown(
    """
    <style>
        .block-container.css-18e3th9.egzxvld2 {
        padding-top: 0;
    }
        header.css-vg37xl.e8zbici2 {
        background: none;
    }
        span.css-10trblm.e16nr0p30 {
        color: #2c39b1;
    }
        .css-1dp5vir.e8zbici1 {
        background-image: linear-gradient(
        90deg, rgb(130 166 192), rgb(74 189 130)
        );
    }
        .css-tw2vp1.e1tzin5v0 {
        gap: 10px;
    }
        h1#liverpool-chapter {
        padding: 0;
    }
        h1#liverpool-chapter span.css-10trblm.e16nr0p30 {
        border-bottom: none;
        font-variant: inherit;
    }
        label.css-cgyhhy.effi0qh3, span.css-10trblm.e16nr0p30 {
        font-weight: bold;
        font-variant-caps: small-caps;
        border-bottom: 3px solid #4abd82;
    }
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns((1, 5))
with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Omdena-Logo.png?raw=true")
with col2:
    st.write('# Liverpool Chapter')

st.markdown('# Project Overview')
st.markdown('Over the last few years improvements to roads in the UK have been implemented across the country in order to create a safer roading system with some great effect.')
st.markdown('The number of RTCs or road traffic collisions are reported to be in decline.')
st.markdown('However there still seems to be a rise in severe and fatal collisions.')
st.markdown('Using datasets from the Department of Transport, we hope to be able to uncover the probability of the severity of a collision.')


st.markdown('## Organisation')
col1, col2 = st.columns((1, 1))
with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/pipeline.jpg?raw=true")
with col2:
    st.markdown('### Predict RTC Severity')
    st.markdown('Using the data,  build an accurate model that can predict the severity of a collision')
    st.markdown('Bonus: create artificial data with improvements in hotspot areas and demonstrate whether or not this reduces the prediction severity ')
    st.markdown('### Geospatial Heatmap')
    st.markdown('Create a robust UK heat map to illustrate the collision hotspots.')
    st.markdown('This maybe done by using a Spatial-Temporal Residual Networks.')
    st.markdown('You must be able to scroll through the years which adjusts the map')
    st.markdown('Bonus: create artifical data with improvements in hotspot areas and demonstrate whether or not this reduces the occurrence of accidents')
    st.markdown('### Time Series Analysis & Predictions')
    st.markdown('Plot the accident occurence over the UK, forecast the rate with the data as is.')
    st.markdown('Bonus: Focus on smaller areas of the UK and compare them with artificial improvements')
    st.markdown('### Vehicle Analysis & Prediction')
    st.markdown('In-depth analysis of vehicles involved in collisions and create a model to predict the likelihood of specific cohorts being involved in a collision')
    st.markdown('Bonus: predicting what class of vehicle is in specific accidents')
st.write('## Pipelines')
de_dict = {
    'TASK': 'Data Preprocessing',
    'Lead Engineers': 'Vivek Srinivasan',
    'Engineers': ['Jahanzeb Hussain', 'Owais Tahir']
}
org = pd.DataFrame(de_dict)
sp_dict = {
    'TASK': [
        'Exploratory Data Analysis', 'Feature Engineering',
        'Model Development & Eval.'
        ],
    'Lead Engineers': [
            'Ibrahim KAISER', 'Nisar Ahmad', 'Michael Welter'
        ],
    'Engineers': [
            'Kevin Medri', 'Joel Luis', 'Lucas Silva/Djazila Souhila Korti'
        ]
}
org1 = pd.DataFrame(sp_dict)
gad_dict = {
    'TASK': [
        'Exploratory Data Analysis', 'Feature Engineering',
        'Model Development & Eval.'
        ],
    'Lead Engineers': [
            'Usman Ayaz', 'Francisca Ngeno', 'Monisha Ayelligadala'
        ],
    'Engineers': [
            'Neethu Prabhakaran', 'Dilane FOGUE KAMGA', 'Puneet Surya'
        ]
}
org2 = pd.DataFrame(gad_dict)
tsa_dict = {
    'TASK': [
        'Exploratory Data Analysis', 'Feature Engineering',
        'Model Development & Eval.'
        ],
    'Lead Engineers': [
            'Cannel Maina', 'Aakanksha', 'Dhananjai Singh'
        ],
    'Engineers': [
            'Mahrukh waqar', 'Wajeeha Imtiaz', 'Syed Muhammad Mubashir Rizvi'
        ]
}
org3 = pd.DataFrame(tsa_dict)
vap_dict = {
    'TASK': [
        'Exploratory Data Analysis', 'Feature Engineering',
        'Model Development & Eval.'
        ],
    'Lead Engineers': [
            'Shrawan Baral/Faizan', 'Gordana Vujovic', 'Shrawan Baral'
        ],
    'Engineers': [
            'Hasaan Maqsood', 'Raza Ali', 'Suneeta Modekurty'
        ]
}
org4 = pd.DataFrame(vap_dict)
col1, col2 = st.columns((1, 1))
with col1:
    st.write('### PL 1 - Severity Prediction')
    st.table(org1)
    st.write('### PL 3 - Time Series Analysis & Prediction')
    st.table(org3)
    st.write('### Data Engineers')
    st.table(org)
with col2:
    st.write('### PL 2 - Geospatial Accident Data')
    st.table(org2)
    st.write('### PL 4 - Vehicle Analysis & Prediction')
    st.table(org4)

st.markdown('Using Data Science we will develop and deploy a machine learning model in an effort to predict RTC severity:')
st.markdown('- Preprocessing')
st.markdown('- Exploratory Data Analysis')
st.markdown('- Feature Engineering')
st.markdown('- Modeling')
st.markdown('- Machine Learning')
st.markdown('The project has six sections:')
st.markdown('1. Data Engineering')
st.markdown('2. Pipeline 1 Predicting RTC Severity')
st.markdown('3. Pipeline 2 Geospatial Heatmap')
st.markdown('4. pipeline 3 Time Series Analysis')
st.markdown('5. Pipeline 4 Vehicle Analysis and Predictions')
st.markdown('6. Solution Deployment')
st.markdown('**Data Engineering** prepares the datasets for pipelines 1 - 4')
st.markdown('**Pipelines 1 - 4** will run concurrently and have three tasks:')
st.markdown('- EDA')
st.markdown('- Feature Engineering')
st.markdown('- Model Development and Evaluation')
st.markdown('**Solution Deployment** will bring together the models and create the solution to be deployed.')
st.markdown('Each Pipeline will produce a Jupyter notebook, based on the findings of each of the team members notebooks, for their task.')
st.markdown('The task lead will then produce a combined notebook, being passed on to the next task until completion of all three tasks.')
st.markdown('The notebooks will be published on the Omdena Liverpool GitHub site.')