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
        #a-safe-space-to-practice span.css-10trblm.e16nr0p30{
        text-align: center;
        font-size: 2rem;
        color: #82a6c0;
        border-bottom: none;
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

st.write('Omdena local chapters are made up of people from all over the world, from all areas of expertise, and levels of experience.')
st.write('From complete beginners looking to start on a direction in Data Science, to seasoned modeling and AI scientists, engineers and analysts.')
st.subheader('“A safe space to practice”')
st.write('Omdena Liverpool chapter was made up of a team from aver 10 different timezones. Despite the distances and time separations we manged to get together and successfully collaborate and produce some ecxcellent work.')
st.write('All involved played an important role to the outcomes of the project.')

col1, col2, col3, col4 = st.columns((1, 1, 1, 1))

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/rich_gregson.jpg?raw=true")
    st.header('Rich Gregson')
    st.subheader('Chapter Lead')
    st.write('I am an accomplished Data Scientist & Analyst with 10+ years of experience. I am well versed in Data Science and Machine Learning using advanced data methodologies to reach useful business insights. I am proficient in data preprocessing, statistical method application, predictive modeling, data visualisation and result communication.')
    st.write(
        fa_li,
        '[linkedin.com/in/rich-gregson](https://www.linkedin.com/in/rich-gregson/)',
        unsafe_allow_html=True
    )
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
    st.write(
        fa_li,
        '[linkedin.com/in/owais-tahir-a049b6233](https://www.linkedin.com/in/owais-tahir-a049b6233/)',
        unsafe_allow_html=True
    )
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/kevin_medri.jpg?raw=true")
    st.header('Kevin Medri')
    st.subheader('Team Member')
    st.write('I am a data specialist working with data from varied market areas. I have years of domain knowledge and am always committed to the projects at hand. I am able to retrieve great business insights, am well versed in stakeholder presentations and working with and managing teams. I have a sharp eye for detail and have great communication skills.')
    st.write(
        fa_li, '[www.linkedin.com/in/kevinmedri](https://www.linkedin.com/in/kevinmedri/)',
        unsafe_allow_html=True
    )

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Francisca_Ngeno.jpg?raw=true")
    st.header('Francisca Ngeno')
    st.subheader('Team Member')
    st.write('This was my first Omdena Project and I have had a great experience in the Liverpool Chapter. I had an amazing team and above all, I have learnt so much. My skills are better than they were two months ago. I acknowledge the efforts that Rich, Salman and everyone else put in while working on the project. Good luck in your future projects.')
    st.write(
        fa_li, '[www.linkedin.com/in/francisca-ngeno](https://www.linkedin.com/in/francisca-ngeno-0a0b49222/)',
        unsafe_allow_html=True
    )
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Dounia_Sadiky.jpg?raw=true")
    st.header('Dounia Sadiky')
    st.subheader('Team Member')
    st.write('I am an engineer passionate about mathematics, data science, finance, football...I try to find answers to questions related to my fields of interest using my scientific toolbox. I have joined two Omdena projects so far in order to enhance my data science skills and I had the chance to meet amazing people from whom I learned a lot.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/dounia-sadiky-56a729204](https://www.linkedin.com/in/dounia-sadiky-56a729204/)',
        unsafe_allow_html=True
    )
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Saleh_Ahmed_Rony.gif?raw=true")
    st.header('Saleh Ahmed Rony')
    st.subheader('Team Member')
    st.write('I am a data specialist working with data from varied market areas. I have years of domain knowledge and am always committed to the projects at hand. I am able to retrieve great business insights, am well versed in stakeholder presentations and working with and managing teams. I have a sharp eye for detail and have great communication skills.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/saleh-ahmed-rony](https://www.linkedin.com/in/saleh-ahmed-rony-135493156/)',
        unsafe_allow_html=True
    )
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Wajeeha_Imtiaz.gif?raw=true")
    st.header('Wajeeha Imtiazi')
    st.subheader('Team Member')
    st.write('Wajeeha Imtiaz- A Computer Science student from NUST. Student at National University of Sciences and Technology (NUST). NUST Cyber Security Club. Deeply interested in the Artificial Intelligence and Machine Learning domain. National University of Sciences & Technology (NUST), located in the heart of Islamabad.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/wajeeha-imtiaz](https://www.linkedin.com/in/wajeeha-imtiaz-37a7ab197/)',
        unsafe_allow_html=True
    )

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Enyinnaya_Benjamin_Uzuegbu.jpg?raw=true")
    st.header('Enyinnaya Uzuegbu')
    st.subheader('Team Member')
    st.write('Its Enyinnaya Benjamin uzuegbu from Omdena Liverpool chapter. I have one year experience in data science. Having graduated from University of East London. Over 2 years of experience in environmental industry, gaining valuable transferable skills in relationship building, strategic planning, data analysis, project management.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/enyinnaya-uzuegbu](https://www.linkedin.com/in/enyinnaya-benjamin-uzuegbu/)',
        unsafe_allow_html=True
    )
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Michael_Welter.jpg?raw=true")
    st.header('Michael Welter')
    st.subheader('Team Member')
    st.write('Data Center Operations Server Engineer at AWS | Certified Data Scientist. Philadelphia, USA Chapter – Data Science for Climate Change: Mitigate Greenhouse Gases Emissions by reducing energy consumption of buildings (part 1). Philadelphia, USA Chapter – Data Science for Climate Change: Reducing energy consumption of buildings (part 2)')
    st.write(
        fa_li,
        '[www.linkedin.com/in/michaeltwelter](https://www.linkedin.com/in/michaeltwelter/)',
        unsafe_allow_html=True
    )
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/lavanya-galgali.jpg?raw=true")
    st.header('Lavanya Galgali')
    st.subheader('Team Member')
    st.write('A BE graduate with 6 yrs of HR domain experience and exploring data in Machine learning  & AI. Skilled in handling end-to-end HR activities and maintaining cordial relationship with the employees. A dynamic HR Professional with around 6 years of rich experience in HR Generalist, HR Analyst, Proficient in MS-Office tools')
    st.write(
        fa_li,
        '[www.linkedin.com/in/lavanya-galgali](https://www.linkedin.com/in/lavanya-galgali-2a684b4/)',
        unsafe_allow_html=True
    )
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Raafe_Asad.jpg?raw=true")
    st.header('Raafe Asad')
    st.subheader('Team Member')
    st.write('Virtual Reality Software Engineer. Skilled in Unity, 3D Modelling in Blender, Engineering, VR Development and Game Development. Strong engineering professional with a Bachelor of Engineering - BE focused in Computer Software Engineering from NED University of Engineering and Technology.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/raafe-asad](https://www.linkedin.com/in/raafe-asad-01b08114b/)',
        unsafe_allow_html=True
    )

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Shrawan_Baral.jpg?raw=true")
    st.header('Shrawan Baral')
    st.subheader('Team Member')
    st.write('ML practitioner :: Cybersec :: Software Engineer. Avid learner, love exploring and exploiting knowledge, who is always fascinated by the technological advancements and their use for uplifting peoples quality of life, nature and surrounding. A keen observer of advancements and their impacts.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/shrawan-baral](https://www.linkedin.com/in/shrawan-baral/)',
        unsafe_allow_html=True
    )
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Mahrukh_Waqar.jpg?raw=true")
    st.header('Mahrukh Waqar')
    st.subheader('Team Member')
    st.write('Data Analyst | Data Scientist | ML Development. Mahrukh Waqar is a data science enthusiast who worked on various projects. She is a keen observer and tends to understand problems to build their solutions. Fluent in English and basic understanding of German language. Google Badges: https://g.dev/mahrukhw')
    st.write(
        fa_li,
        '[www.linkedin.com/in/mahrukhw](https://www.linkedin.com/in/mahrukhw/)',
        unsafe_allow_html=True
    )
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Gordana_Vujović.jpg?raw=true")
    st.header('Gordana Vujović')
    st.subheader('Team Member')
    st.write('Im a dedicated and passionate engineer working with Machine Learning / AI and have a great interest in algorithm development, technology, and programming. I like new challenges and stay up to date on whats happening in modern platforms and development frameworks. As a person Im flexible, positive, open, dare to say what I think, and like.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/gordana-vujovic-665a42b5](https://www.linkedin.com/in/gordana-vujovic-665a42b5/)',
        unsafe_allow_html=True
    )
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Ayoub_BERDEDDOUCH.jpg?raw=true")
    st.header('Ayoub B.')
    st.subheader('Team Member')
    st.write('Associate Member Omdena. Google Cloud Associate Engineering Mentor. Data Scientist | Lead ML Engineer @Omdena | 🐍 Django | ☁️ Engineer | Researcher Scientist | NaaS.ai Contributor. PhD Researcher Scientist. 👨‍💻 Data Scientist 👨‍💻 Ph.D. Candidate 👨‍💻 Python ~Django Dev👨‍💻 Cloud Engineer 📧 : Envoie moi un Message en privé.')
    st.write(
        fa_li,
        '[www.linkedin.com/in/ayoub-berdeddouch](https://www.linkedin.com/in/ayoub-berdeddouch/)',
        unsafe_allow_html=True
    )

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Pransenjit_Chowdhury.jpg?raw=true")
    st.header('Pransenjit C.')
    st.subheader('Team Member')
    st.write('I accept with pleasure the challenges and goals assigned to me. Service Management AnalystService Management Analyst Kantar IT Partnership. IT AnlaystIT Anlayst HSBC. Bachelors degree, Information technology. 1729 Original1729 Original Analytics Vidhya. AWS Academy Graduate - AWS Academy Cloud Architecting')
    st.write(
        fa_li,
        '[www.linkedin.com/in/prasenjit-chowdhury](https://www.linkedin.com/in/prasenjit-chowdhury-b3049050/)',
        unsafe_allow_html=True
    )
with col2:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Jawairia_Rasheed.jpg?raw=true")
    st.header('Jawairia Rasheed')
    st.subheader('Team Member')
    st.write('Data Scientist. I am data and machine learning practitioner using Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn. Lecturer University of Gujrat, Pakistan. Lab EngineerLab Engineer SEECS. College of Electrical & Mechanical Engineering NUST. Masters Degree, Computer Software Engineering')
    st.write(
        fa_li,
        '[www.linkedin.com/in/jawairia-rasheed](https://www.linkedin.com/in/jawairia-rasheed-159ab755/)',
        unsafe_allow_html=True
    )
with col3:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('JUN T.')
    st.subheader('Team Member')
    st.write('👨‍💻 Programmer')
    st.write(
        fa_li,
        '[www.linkedin.com/in/jun-t](https://www.linkedin.com/in/jun-t-82376b257)',
        unsafe_allow_html=True
    )
with col4:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Djazila KORTI')
    st.subheader('Team Member')
    st.write('Teaching practical work for bachelor degree students. (local networks) ')
    st.write('Teaching practical work for Master 2 students. (Mobile and Wireless Networks)')
    st.write(
        fa_li,
        '[www.linkedin.com/in/djazila-souhila-korti](https://www.linkedin.com/in/djazila-souhila-korti-1470521b9/)',
        unsafe_allow_html=True
    )

with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/head-placeholder.png?raw=true")
    st.header('Jarukamol')
    st.subheader('Team Member')
    st.write('Jarukamol Dawkrajai is a PhD student. Working in chemical process fault detection using deep learning.')
    st.write(
        fa_em, '<a href="mailto:ormjarukamol@gmail.com">Email Jarukamol</a>',
        unsafe_allow_html=True
    )
with col2:
    st.write(' ')
with col3:
    st.write(' ')
with col4:
    st.write(' ')
