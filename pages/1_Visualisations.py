import streamlit as st
from PIL import Image

image4 = Image.open('assets/accs_with_time4.png')
image5 = Image.open('assets/accs_with_time5.png')
image6 = Image.open('assets/accs_with_time6.png')
image7 = Image.open('assets/accs_with_time7.png')

        # images = [image4, image5, image6, image7]
        # titles = ['By Year', 'By Month', 'By Quarter', 'By Hours']

        # for title, image in zip(titles, images):
        #     st.subheader(title)
        #     st.image(image)
st.title('By Year')
st.markdown('We can see that from 2005 we have a healthy decline in **RTCs**')
st.markdown('However both **serious** and **Fatal** **RTCs** have either plateaued or are on the rise')
st.image(image4)
st.title('By Month')
st.markdown('Our monthly data differs from severity to severity.')
st.markdown('The **RTCs** for **Fatal** peak in August, October and November.')
st.markdown('The **RTCs** for **Serious** peak in July.')
st.markdown('The **RTCs** for **Slight** peak in November.')
st.image(image5)
st.title('By Quarter')
st.markdown('Both **Fatal** and **Slight** **RTCs** peak in Quarter 4.')
st.image(image6)
st.title('By Hours')
st.markdown('All of our **RTCs** show an increase at rush hours.')
st.image(image7)