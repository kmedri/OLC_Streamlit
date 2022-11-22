import streamlit as st
import pandas as pd

st.set_page_config(page_title='The Team', layout='wide')

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
        </style>
    """, unsafe_allow_html=True
)

fa_li = """
<i class="fab fa-linkedin"></i>
"""
fa_em = """
<i class="far fa-envelope"></i>
"""

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
    st.write('I am an accomplished Data Scientist & Analyst with 10+ years of experience. I am well versed in Data Science and Machine Learning using advanced data methodologies to reach useful business insights. I am proficient in data preprocessing, statistical method application, predictive modeling, data visualisation and result communication.')
    st.write(fa_li, '[linkedin.com/in/rich-gregson](https://www.linkedin.com/in/rich-gregson/)', unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Salman_Khaliq.jpg?raw=true")
    st.header('Salmon Khaliq')
    st.subheader('Chief Engineer and Educator')
    st.write('⚡ My Machine Learning journey 🚗 started in 2008 when I learned to code in MATLAB, and then started implementing the code for the numerical methods course the next year. Thats when I joined Dr. Usmanis Lab, where the research focus was on applying machine learning techniques for predictive modeling of time series 🕝 data.')
    st.write(fa_li, '[linkedin.com/in/salmankhaliq22](https://www.linkedin.com/in/salmankhaliq22/)', unsafe_allow_html=True)
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Owais_Tahir.jpg?raw=true")
    st.header('Owais Tahir')
    st.subheader('Team Member')
    st.write('I am doing masters in data sciences, I have spent the past several years immersing myself in data analytics and machine learning. I have experience in data visualization and developing workflows to compare diverse data sources. I have strong communication skills and experience in explaining technical topics to a general audience.')
    st.write(fa_li, '[linkedin.com/in/owais-tahir-a049b6233](https://www.linkedin.com/in/owais-tahir-a049b6233/)', unsafe_allow_html=True)
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/kevin_medri.jpg?raw=true")
    st.header('Kevin Medri')
    st.subheader('Team Member')
    st.write('I am a data specialist working with data from varied market areas. I have years of domain knowledge and am always committed to the projects at hand. I am able to retrieve great business insights, am well versed in stakeholder presentations and working with and managing teams. I have a sharp eye for detail and have great communication skills.')
    st.write(fa_li, '[www.linkedin.com/in/kevinmedri](https://www.linkedin.com/in/kevinmedri/)', unsafe_allow_html=True)

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Francisca_Ngeno.jpg?raw=true")
    st.header('Francisca Ngeno')
    st.subheader('Team Member')
    st.write('This was my first Omdena Project and I have had a great experience in the Liverpool Chapter. I had an amazing team and above all, I have learnt so much. My skills are better than they were two months ago. I acknowledge the efforts that Rich, Salman and everyone else put in while working on the project. Good luck in your future projects.')
    st.write(fa_li, '[www.linkedin.com/in/francisca-ngeno](https://www.linkedin.com/in/francisca-ngeno-0a0b49222/)', unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Dounia_Sadiky.jpg?raw=true")
    st.header('Dounia Sadiky')
    st.subheader('Team Member')
    st.write('I am an engineer passionate about mathematics, data science, finance, football...I try to find answers to questions related to my fields of interest using my scientific toolbox. I have joined two Omdena projects so far in order to enhance my data science skills and I had the chance to meet amazing people from whom I learned a lot.')
    st.write(fa_li, '[www.linkedin.com/in/dounia-sadiky-56a729204](https://www.linkedin.com/in/dounia-sadiky-56a729204/)', unsafe_allow_html=True)
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Saleh_Ahmed_Rony.gif?raw=true")
    st.header('Saleh Ahmed Rony')
    st.subheader('Team Member')
    st.write('I am a data specialist working with data from varied market areas. I have years of domain knowledge and am always committed to the projects at hand. I am able to retrieve great business insights, am well versed in stakeholder presentations and working with and managing teams. I have a sharp eye for detail and have great communication skills.')
    st.write(fa_li, '[www.linkedin.com/in/saleh-ahmed-rony](https://www.linkedin.com/in/saleh-ahmed-rony-135493156/)', unsafe_allow_html=True)
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Wajeeha_Imtiaz.gif?raw=true")
    st.header('Wajeeha Imtiazi')
    st.subheader('Team Member')
    st.write('Wajeeha Imtiaz- A Computer Science student from NUST.')
    st.write(' ')
    st.write('Deeply interested in the Artificial Intelligence and Machine Learning domain.')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(fa_li, '[www.linkedin.com/in/wajeeha-imtiaz](https://www.linkedin.com/in/wajeeha-imtiaz-37a7ab197/)', unsafe_allow_html=True)

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Enyinnaya_Benjamin_Uzuegbu.jpg?raw=true")
    st.header('Enyinnaya Uzuegbu')
    st.subheader('Team Member')
    st.write('Its Enyinnaya Benjamin uzuegbu from Omdena Liverpool chapter.')
    st.write(' ')
    st.write(' I have one year experience in data science. Having graduated from University of East London.')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(fa_li, '[www.linkedin.com/in/enyinnaya-uzuegbu](https://www.linkedin.com/in/enyinnaya-benjamin-uzuegbu/)', unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Jarukamol')
    st.subheader('Team Member')
    st.write('Jarukamol Dawkrajai is a PhD student.')
    st.write(' ')
    st.write('Working in chemical process fault detection using deep learning.')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(fa_em, '[www.linkedin.com/in/dounia-sadiky-56a729204](https://www.linkedin.com/in/dounia-sadiky-56a729204/)', unsafe_allow_html=True)
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Lavanya')
    st.subheader('Team Member')
    st.write('Lavanya Galgali is a BE graduate.')
    st.write(' ')
    st.write('With 6 yrs of HR domain experience and exploring data in Machine learning  & AI. ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(fa_em, '<a href="mailto:lavanyagalgali@gmail.com">Email Lavanya</a>', unsafe_allow_html=True)
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Raafe_Asad.jpg?raw=true")
    st.header('Raafe Asad')
    st.subheader('Team Member')
    st.write('Virtual Reality Software Engineer. Skilled in Unity, 3D Modelling in Blender, Engineering, VR Development and Game Development. Strong engineering professional with a Bachelor of Engineering - BE focused in Computer Software Engineering from NED University of Engineering and Technology.')
    st.write(fa_li, '[www.linkedin.com/in/raafe-asad](https://www.linkedin.com/in/raafe-asad-01b08114b/)', unsafe_allow_html=True)

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Shrawan_Baral.jpg?raw=true")
    st.header('Shrawan Baral')
    st.subheader('Team Member')
    st.write('Avid learner, love exploring and exploiting knowledge, who is always fascinated by the technological advancements and their use for uplifting peoples quality of life, nature and surrounding. A keen observer of advancements and their impacts.')
    st.write(' ')
    st.write(' ')
    st.write(fa_li, '[www.linkedin.com/in/shrawan-baral](https://www.linkedin.com/in/shrawan-baral/)', unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Mahrukh_Waqar.jpg?raw=true")
    st.header('Mahrukh Waqar')
    st.subheader('Team Member')
    st.write('Mahrukh Waqar is a data science enthusiast who worked on various projects. She is a keen observer and tends to understand problems to build their solutions.')
    st.write(fa_li, '[www.linkedin.com/in/mahrukhw](https://www.linkedin.com/in/mahrukhw/)', unsafe_allow_html=True)
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Lavanya')
    st.subheader('Team Member')
    st.write('Lavanya Galgali is a BE graduate.')
    st.write(' ')
    st.write('With 6 yrs of HR domain experience and exploring data in Machine learning  & AI. ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(fa_li, '<a href="mailto:lavanyagalgali@gmail.com">Email Lavanya</a>', unsafe_allow_html=True)
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Raafe Asad')
    st.subheader('Team Member')
    st.write('Virtual Reality Software Engineer. Skilled in Unity, 3D Modelling in Blender, Engineering, VR Development and Game Development. Strong engineering professional with a Bachelor of Engineering - BE focused in Computer Software Engineering from NED University of Engineering and Technology.')
    st.write(fa_li, '[www.linkedin.com/in/raafe-asad](https://www.linkedin.com/in/raafe-asad-01b08114b/)', unsafe_allow_html=True)
