import streamlit as st

APP_TITLE = 'Predicting RTC severity using Machine Learning'

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

col1, col2 = st.columns((1, 3))
with col1:
    st.image("https://github.com/liskibruh/streamlit-dashboard-RTC/blob/main/assets/omdenaliverpoollogo.png?raw=true")
with col2:
    st.image("https://raw.githubusercontent.com/liskibruh/streamlit-dashboard-RTC/main/assets/accidents1104x271.png")



st.markdown('## Visualisations')