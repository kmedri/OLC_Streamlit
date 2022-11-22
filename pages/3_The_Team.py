import streamlit as st
import pandas as pd

st.set_page_config(page_title='The Team', layout='wide')

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
        h3#chapter-lead span.css-10trblm.e16nr0p30,
        h3#chief-engineer-and-educator span.css-10trblm.e16nr0p30,
        h3#team-member span.css-10trblm.e16nr0p30 {
        font-variant: none;
        border-bottom: 2px solid #82a6c0;
        font-size: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns((1, 5))
with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Omdena-Logo.png?raw=true")
with col2:
    st.write('# Liverpool Chapter')

col1, col2, col3, col4 = st.columns((1, 1, 1, 1))

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/rich_gregson.jpg?raw=true")
    st.header('Rich Gregson')
    st.subheader('Chapter Lead')
    st.write('I am an accomplished Data Scientist & Analyst with over 10 years of experience. I am well versed in Data Science and Machine Learning and can use advanced data methodologies to reach useful business insights. I am proficient in the all data science stages, including data preprocessing, statistical method application, predictive modeling, data visualisation and result communication.')
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Salman_Khaliq.jpg?raw=true")
    st.header('Salmon Khaliq')
    st.subheader('Chief Engineer and Educator')
    st.write('⚡ My Machine Learning journey 🚗 started in 2008 when I learned to code in MATLAB, and then started implementing the code for the numerical methods course the next year. Thats when I joined Dr. Usmanis Lab, where the research focus was on applying machine learning techniques for predictive modeling of time series 🕝 data.')
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Owais_Tahir.jpg?raw=true")
    st.header('Owais Tahir')
    st.subheader('Team Member')
    st.write('I am doing masters in data sciences, I have spent the past several years immersing myself in data analytics and machine learning. I have experience in data visualization and developing workflows to compare diverse data sources. I have strong communication skills and experience in explaining technical topics to a general audience.')
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/kevin_medri.jpg?raw=true")
    st.header('Kevin Medri')
    st.subheader('Team Member')
    st.write('I am a data specialist working with data from varied market areas. I have years of domain knowledge and am always committed to the projects at hand. I am able to retrieve great business insights, am well versed in stakeholder presentations and working with and managing teams. I have a sharp eye for detail and have great communication skills.')
