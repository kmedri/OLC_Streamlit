import streamlit as st
import streamlit.components.v1 as components
import time

st.set_page_config(page_title='Time Series', layout='wide')


def load_time_series(url):
    html_string = open(url)
    return html_string


st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
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
        #a-safe-space-to-practice span.css-10trblm.e16nr0p30{
        text-align: center;
        font-size: 2rem;
        color: #82a6c0;
        border-bottom: none;
        }
        .css-1al18u3.e1tzin5v0, iframe {
        height: 1150px;
        }
        </style>
    """, unsafe_allow_html=True
)


col1, col2 = st.columns((1, 5))
with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Omdena-Logo.png?raw=true")
with col2:
    st.write('# Liverpool Chapter')

st.markdown('# Time Series')

col1, col2 = st.columns((1, 1))
with col1:
    components.html(load_time_series('data/map_test.html').read())

with col2:
    st.title('Time Series Using Plotly')
    st.write('Using Plotly we created a time series of the severe and fatal accidents across the UK.')
    st.write('By using the latitude and longitude values we managed to plot each accident.')
    st.write('It then runs on a timer across each day of the year from 1st Jan 2015 to 31st Dec 2017')
    st.write('If you hover over the map you can see more details')
    st.write('The map can be paused, you can use the slider to move across dates.')
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/time_series.png?raw=true")
