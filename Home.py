import folium as flm
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

APP_TITLE = 'Predicting RTC severity using Machine Learning'
st.set_page_config(page_title='Home', layout='wide')



# Load the DATA and cache.
@st.cache
def get_data(url):
    df = pd.read_parquet(url)
    return df


url = 'data/full_accident_data_time_series.parquet'
df = get_data(url)


def main():
    # Colors:
    # Blue = #182D40
    # Light Blue = #82a6c0
    # Green = #4abd82

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
        label.css-cgyhhy.effi0qh3, span.css-10trblm.e16nr0p30 {
        font-size: 1.1em;
        font-weight: bold;
        font-variant-caps: small-caps;
        border-bottom: 3px solid #4abd82;
        }
        .css-12w0qpk.e1tzin5v2 {
        background: #d2d2d2;
        border-radius: 8px;
        padding: 5px 10px;
        }
        label.css-18ewatb.e16fv1kl2 {
        font-variant: small-caps;
        font-size: 1em;
        }
        .css-1xarl3l.e16fv1kl1 {
        float: right;
        }
        div[data-testid="stSidebarNav"] li div a {
        margin-left: 1rem;
        padding: 1rem;
        width: 300px;
        border-radius: 0.5rem;
        }
        div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
        }
        svg.e1fb0mya1.css-fblp2m.ex0cdmw0 {
        width: 2rem;
        height: 2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    col1, col2 = st.columns((1, 3))
    with col1:
        st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/omdenaliverpoollogo.png?raw=true")
    with col2:
        st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/accident_image_fp.png?raw=true")
    st.title(APP_TITLE)
    st.write('**Under Construction** - Please be aware we are currently building this app, so it will change over the next few weeks. Thank you for your patience.')
    st.write('Over the last few years improvements to roads in the UK have been implemented across the country in order to create a safer roading system with some great effect.  \nThe number of **road traffic collisions** are reported to be in decline.  \nUsing datasets from the Department of Transport, we hope to be able to uncover the probability of the severity of a collision.')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Project Background')
        st.write('UK RTCs which have resulted in a persons death have been on a downward trend since the 1960s – however in 2020 1,516 people lost their lives on UK roads. The UK road systems, especially in Liverpool, are dated which means they have not been upgraded to reflect the increase of cars on the road. This means there are still preventative measures that could be implemented to prevent even more deaths on UK roads.')
        st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/map_fp.jpg?raw=true")
    with col2:
        st.subheader('The Problem')
        st.write('By harnessing the power of Machine Learning we intend to predict the severity of RTCs and RTC hotspots which would allow the local authority to implement further traffic safety measures.')
        st.subheader('Project Goals')
        st.write('1. Classifying RTC severity. 2. Identifying areas with the highest number of RTCs. 3. Identifying what type of vehicles are involved in most RTCs. 4. Monitoring the rate of RTCs over time.')
        st.subheader('Project Plan')
        st.markdown('- Week 1 - 1. Data preprocessing - Week 2 - 2. Exploratory Data Analysis to draw insights - Week 3 - 3. Feature Engineering – creating new features based on insights drawn from EDA. - Week 4 - 4. Model Development Week - 5 - 5. Model Evaluation and Deployment – perhaps on AWS or Google Cloud.')
        st.subheader('Learning Outcomes')
        st.write('Data Processing, Exploratory Data Analysis, Feature Engineering, Model Development, Model Evaluation and Model Deployment.')
if __name__ == "__main__":
    main()
