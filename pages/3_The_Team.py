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
        h3#chapter-lead span.css-10trblm.e16nr0p30 {
        font-variant: none;
        border-bottom: 4px solid #82a6c0;
        font-size: 1.5rem;
        }
        h3#cheif-engineer-and-educator span.css-10trblm.e16nr0p30 {
        font-variant: none;
        border-bottom: 4px solid #82a6c0;
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
    st.write('Some Text')
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Salman_Khaliq.jpg?raw=true")
    st.header('Salmon Khaliq')
    st.subheader('Cheif Engineer and educator')
    st.write('Some Text')
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/rich_gregson.jpg?raw=true")
    st.header('Rich Gregson')
    st.subheader('Chapter Lead')
    st.write('Some Text')
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/rich_gregson.jpg?raw=true")
    st.header('Rich Gregson')
    st.subheader('Chapter Lead')
    st.write('Some Text')