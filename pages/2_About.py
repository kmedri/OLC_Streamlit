import streamlit as st

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
    text-align: center;
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
    .css-50ug3q {
    font-size: 1.2em;
    font-weight: 600;
    color: #2c39b1;
    }
    .row-widget.stSelectbox {
    padding: 10px;
    background: #ffffff;
    border-radius: 7px;
    }
    .row-widget.stRadio {
    padding: 10px;
    background: #ffffff;
    border-radius: 7px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

APP_TITLE = 'Predicting RTC severity using Machine Learning - About the Project'

col1, col2 = st.columns((1, 3))
with col1:
    st.image("https://github.com/liskibruh/streamlit-dashboard-RTC/blob/main/assets/omdenaliverpoollogo.png?raw=true")
with col2:
    st.image("https://raw.githubusercontent.com/liskibruh/streamlit-dashboard-RTC/main/assets/accidents1104x271.png")

st.markdown('## Project Overview')
st.markdown('Over the last few years improvements to roads in the UK have been implemented across the country in order to create a safer roading system with some great effect.')
st.markdown('The number of RTCs or road traffic collisions are reported to be in decline.')
st.markdown('However there still seems to be a rise in severe and fatal collisions.')
st.markdown('Using datasets from the Department of Transport, we hope to be able to uncover the probability of the severity of a collision.')
st.markdown('Using Data Science we will develop and deploy a machine learning model in an effort to predict RTC severity:')
st.markdown('- Preprocessing')
st.markdown('- Exploratory Data Analysis')
st.markdown('- Feature Engineering')
st.markdown('- Modeling')
st.markdown('- Machine Learning')
st.markdown('The project has been broken down into six pipelines:')
st.markdown('1. Data Engineering')
st.markdown('2. Group 1 Predicting RTC Severity')
st.markdown('3. Group 2 Geospatial Heatmap')
st.markdown('4. Group 3 Time Series Analysis')
st.markdown('5. Group 4 Vehicle Analysis and Predictions')
st.markdown('6. Solution Deployment')
st.markdown('**Pipeline 1** prepares the datasets for groups 1 - 4')
st.markdown('**Pipelines 2 - 5** will run concurrently and have three tasks:')
st.markdown('- EDA')
st.markdown('- Feature Engineering')
st.markdown('- Model Development and Evaluation')
st.markdown('**Pipeline 6** will bring together the models and create the solution to be deployed.')
st.markdown('Each Pipeline will produce a Jupyter notebook, based on the findings of each of the team members notebooks, for their task.')
st.markdown('The task lead will then produce a combined notebook, being passed on to the next task until completion of all three tasks.')
st.markdown('The notebooks will be published on the Omdena Liverpool GitHub site.')