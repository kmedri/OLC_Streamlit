import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import altair as alt
import io

st.set_page_config(page_title='EDA', layout='wide')

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
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns((1, 5))
with col1:
    st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Omdena-Logo.png?raw=true")
with col2:
    st.write('# Liverpool Chapter')

st.header('Omdena Liverpool Chapter - Predicting RTC Severity - EDA')

st.write('**This is a sample of some of the EDA carried out on the data. It is only a sample of the data so the numbers and results may differ from the main EDA published on GitHub.**')

st.markdown(
    """
### Table of Contents
- [Project Overview](#project-overview)
- [Predicting RTC Severity - EDA](##predicting-rtc-severity---eda)
- [Features](#features---accident_dataparquet)
  - [Junction Map](#map-showing-how-to-code-the-roads-at-a-roundabout-and-slip-roads)
- [Project Dependencies](#import-dependencies)
- [Importing and Loading the Dataset](#importing-and-loading-data-into-dataframe)
  - [Data Shape](#data-shape)
  - [Data Types](#data-types)
- [Data Cleaning](#data-cleaning)
  - [Missing Values](#missing-values)
  - [Dropping Columns](#dropping-columns)
  - [1st and 2nd Road Classes](#1st_road_class)
  - [Duplicates](#duplicates)
    - [Data Types and Formatting](#data-types-and-formatting)
- [Average Counts](#average-counts)
- [Collision Severity Spread](#collision-severity-spread)
    """)

st.markdown(
    """
## Project Overview
Over the last few years improvements to roads in the UK have been implemented across the country in order to create a safer roading system with some great effect.  
The number of RTC's (road traffic collisions) are reported to be in decline.  
  
However there still seems to be a rise in severe and fatal collisions.  
  
Using datasets from the Department of Transport, we hope to be able to uncover the probability of the severity of a collision.  
Using Data Science we will develop and deploy a machine learning model in an effort to predict RTC severity:
- Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Modeling
- Machine Learning
  
The project has been broken down into six pipelines:
1. Data Engineering
2. Group 1 Predicting RTC Severity
3. Group 2 Geospatial Heatmap
4. Group 3 Time Series Analysis
5. Group 4 Vehicle Analysis and Predictions
6. Solution Deployment
  
**Pipeline 1** prepares the datasets for groups 1 - 4  
  
**Pipelines 2 - 5** will run concurrently and have three tasks:
- EDA
- Feature Engineering
- Model Development and Evaluation
  
**Pipeline 6** will bring together the models and create the solution to be deployed.  
  
Each Pipeline will produce a Jupyter notebook, based on the findings of each of the team members notebooks, for their task.  
The task lead will then produce a combined notebook, being passed on to the next task until completion of all three tasks.  
The notebooks will be published on the Omdena Liverpool GitHub site.  
  
This is one of the **Pipeline 2, Group 1, Predicting RTC Severity EDA's** notebooks.
    """)

st.markdown(
    """
## Predicting RTC Severity - EDA

Group 1 is tasked with predicting the road traffic collision severity.  
We will be using a dataset from the Department of Transport, consisting of over a million rows and 34 columns.  
It has a date range from 2005 to 2010.  
    """)

st.markdown(
    """
## Features - accident_data.parquet
We have a reasonable set of features, lets take a look.

**Accident_Index**
- Unique number linking accident with vehicles and casualties.

**1st_Road_Class**
- Motorway
- A(M)
- A
- B
- C
- Unclassified

**1st_Road_Number**
- Number of road if applicable (not all roads have a number)

**2nd_Road_Class**
- None
- Motorway
- A(M)
- A
- B
- C
- Unclassified

**2nd_Road_Number**
- Number of road if applicable (not all roads have a number)

**Accident_Severity**
- Fatal
- Serious
- Slight

**Carriageway_Hazards**
- None
- Vehicle load on road
- Other object on road
- Previous accident
- Dog on road
- Other animal on road
- Pedestrian in carriageway - not injured
- Any animal in carriageway (except ridden horse)
- Data missing or out of range
- unknown (self reported)

**Date**
- Date of accident

**Day_of_Week**
- Day of accident - Monday to Sunday

**Did_Police_Officer_Attend_Scene_of_Accident**
- Yes
- No
- No - accident was reported using a self completion  form (self rep only)

**Junction_Control**
- Not at junction or within 20 metres
- Authorised person
- Auto traffic signal
- Stop sign
- Give way or uncontrolled
- Data missing or out of range
- unknown (self reported)

**Junction_Detail**
- Not at junction or within 20 metres
- Roundabout
- Mini-roundabout
- T or staggered junction
- Slip road
- Crossroads
- More than 4 arms (not roundabout)
- Private drive or entrance
- Other junction
- unknown (self reported)

**Latitude**
- Geographical information

**Light_Conditions**
- Daylight
- Darkness - lights lit
- Darkness - lights unlit
- Darkness - no lighting
- Darkness - lighting unknown
- Data missing or out of range

**Local_Authority_(District)**
- Geographical list of Districts

**Local_Authority_(Highway)**
- Geographical list of Local Highway Authorities

**Location_Easting_OSGR**
- Geographical information

**Location_Northing_OSGR**
- Geographical information

**Longitude**
- Geographical information

**LSOA_of_Accident_Location**
- Statistical location for Local Government

**Number_of_Casualties**
- Number or persons injured or killed

**Number_of_Vehicles**
- Number of vehicles involved in the accident

**Pedestrian_Crossing-Human_Control**
- None within 50 metres
- Control by school crossing patrol
- Control by other authorised person
- Data missing or out of range
- unknown (self reported)

**Pedestrian_Crossing-Physical_Facilities**
- No physical crossing facilities within 50 metres
- Zebra
- Pelican, puffin, toucan or similar non-junction pedestrian light crossing
- Pedestrian phase at traffic signal junction
- Footbridge or subway
- Central refuge
- Data missing or out of range
- unknown (self reported)

**Police_Force**
- List of Police Forces across England, Wales and Scotland

**Road_Surface_Conditions**
- Dry
- Wet or damp
- Snow
- Frost or ice
- Flood over 3cm. deep
- Oil or diesel
- Mud
- Data missing or out of range
- unknown (self reported)

**Road_Type**
- Roundabout
- One way street
- Dual carriageway
- Single carriageway
- Slip road
- Unknown
- One way street/Slip road

**Special_Conditions_at_Site**
- None
- Auto traffic signal - out
- Auto signal part defective
- Road sign or marking defective or obscured
- Roadworks
- Road surface defective
- Oil or diesel
- Mud
- Data missing or out of range
- unknown (self reported)

**Speed_limit**
- 20, 30, 40, 50, 60, 70 are the only valid speed limits on public highways

**Time**
- Time of accident

**Urban_or_Rural_Area**
- Urban
- Rural
- Unallocated
- Data missing or out of range

**Weather_Conditions**
- Fine no high winds
- Raining no high winds
- Snowing no high winds
- Fine + high winds
- Raining + high winds
- Snowing + high winds
- Fog or mist
- Other
- Unknown
- Data missing or out of range

**Year**
- Year of accident

**InScotland**
- In Scotland or not
    """)

st.markdown(
    """
#### Map showing how to code the roads at a roundabout and slip roads.

![stats20-Map.jpg](data:image/jpeg;base64,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ACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiq99qFrpdrJdXlzDaW0eN807hEXJwMseByR+dcp4i+M3gXwrpcmo6p4s0m3s4yqtIt0shBJwPlQk9faolUhDWTsdVDCYjFNRoU5TbdtE3r206nZ0V4frP7Zvwp06CI2GvSeI7uWQRpY6LavPcNkE5CkLkDHrWNc/thprFzaWHg/wCG/i3xFqs7sDa3dp9gUIFLFvMbcCeOhx1/CuOWPwy050/TX8rn0VPhPPKi5nhZRXefuJW6tz5Ul5vQ+iaK+dpvi98c/FF9bWfh74SWvhp8O8114ov/ADYCBjCr5O0huvUHPtQ+jftKeMNRiS68Q+FPAFnDE5M2j2f9oG4cldqss+doA3HKke4PGF9djL4ISl8rfnY1/wBWatO31rFUaXXWopW+VPnbv5J+dj6JpjTRq2GkUH0JFfPNv+zP458QXd1feLvjh4vkvpCojXwzcf2XbqgXHMK5Xd7gDPfnmn2f7DHw1upLq78U/wBteONYuZfMk1bW9VnNywCKqoWjZAwAXgkE84zgAA9tiZfDRt6yS/JSD+zMko39vmLl/wBe6Upa+XPKlp56PyPTdW+Onw60DUrnTtS8c+H7C/tnMc1tcalEkkbDqGUtkGvPr79uL4PWF9c2reJJp2glaIyW2nzyxMVJBKOqFWXjhgSCORXVeG/2YPhR4W0xbC08AaDcwqxfzNSskvZcn/ppMGcj2zgV6BoXh/S/C+lw6Zo2m2mkabDu8qzsYFhhj3MWbaigAZYknA6kmi2Mlu4x+Tf6oHU4ao35ader6yhT+ekan3fieDaf+19c65ZpfaP8IvH2r6XNk299aaaGinQEgOpz0OKZafFL9oTWNJiurb4QaXafaYRJGLrW0jlj3DI3xtgqwzypwexr6MopfVq0vjrP5JL9GDzrLKV/q+WU9/tzqT0+UofPTXsj5yk8J/tL6ro7Ry+MvB2nS3MG1xFZSiWBmXnDAEblJ6jIyKdffsv+Nta02Sx1P44+Jrq0uE8u4gFvGodT95chsj619F1kaj4v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4Rav/bcEcnmRaNqM4t7yE7lC+VOSFbaMsWYqeOATX231zh7iVWx8FgsT/PCLdKX+KG8H5x06tdublq0fh95dnufqfRX4xfCH9vb40fswawnhDxrY3utWGnpFHJoPiaJ7fULOMxxiMIzDcFEYBVWHO7JPNfpn+zj+1x8P/wBpnQ47jw3qaWeuKjyXXhy/kRNQtlVlUu0YPKEumHXKktjOa+ZzrhnMMkjGtWSnRn8NSDUoS9JL8nZm1OtCpot+3U9qooor5Q3CiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+Vf8Aglx/yYn8Mv8AuJ/+nS7r6qr86vgP8TNQ+EP/AAR40/xNpNyLPVobHUrazut4TyJp9XuYUkBPAKmTd/wGplLlVy4R55KPc+64fit4JuPGMnhKLxjoEviuMZfQk1OA3yjAPMG7eOCO3euqr84o1/ZduvCei+GJ/h74o0LTx9lWD44WvhY28LX4kULcLrDRl2dpefMZTEc9cYr9F7SPyrWFPOa52oo86QgtJgfeOABk9eABWzjyq/nby/ryfk+pipcz+Xz/AK/4Ymr5V/aI/wCT7P2Rv+5u/wDTXFX1VXyr+0R/yfZ+yN/3N3/prirMs+qqKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPnb/AIJ+/wDJqvhr/sKa7/6eb2ij/gn7/wAmq+Gv+wprv/p5vaKAOt+DH/JUvj3/ANjfZ/8AqP6RXrleR/Bj/kqXx7/7G+z/APUf0ivXKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvn79qz/kYvgl/2Pdh/wCzV9A182/tneILDwo/wj1rVJvs2m6f4zs7q5m2M+yNFdmbaoJOADwATXn5g1HDSb8vzR9jwjCVTOqMIK7amklu3ySPpKiiivQPjgooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA85+P3xJl+F/w1vdQsF87xBfSJpejW643TX0x2xBdwKkr8z4bghCO9W/gj8NovhR8NdI8Pq3m3MaGe8mAIEtw53SPtJO3LE8A4HavOdO/4vj+0fNqP+v8AB3w7Rre2dfmhu9WmH7xlYbkk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8FfECSS53sZIdetVMezHyqhiAOc8ZavnmT9kP9qP9n/UrXU/D2ja9a6heRSW4ufB2oNcSJGChZJDGfkUnbgHrtPpVPIeHsw1yvM1Tf8ALXi4d/tx5o/l+Ng9rVh8cL+h+lv/AAT9/wCTVfDX/YU13/083tFZX/BNL7WP2K/AIv8Azft3n6v9o877/mf2tebt3vnOaK/NpLlbR2HoXwY/5Kl8e/8Asb7P/wBR/SK9cryP4Mf8lS+Pf/Y32f8A6j+kV65UgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFfMPx1+Ovifx147m+CPwRmjbxu0at4j8WMnmWfhO1bje3Z7phny4s5zycAEhPjr8dfE/jrx3P8ABH4Izxv43aNW8R+LGTzLPwnatxvbs90wz5cWc55OACR638CvgV4Y/Z78CQ+GfDMErhpGub/U7x/MvNTum/1lzcSdXkY/gBgAAACgA+BXwK8Mfs9+BIPDPhmCRw0jXN/qd43mXmp3TcyXNxJ1eRj+AGAAAAK9EoooAKKKKACiiigAooooAKKKKACiiigAooooA+dv+Cfv/Jqvhr/sKa7/AOnm9oo/4J+/8mq+Gv8AsKa7/wCnm9ooA634Mf8AJUvj3/2N9n/6j+kV65XkfwY/5Kl8e/8Asb7P/wBR/SK9coAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvmrxZ/wUe/Z28D+KtZ8N638Q/sWtaPezaffW39iajJ5M8UjRyJuS3KthlIypIOOCRX0rXyr+wL/AM3G/wDZZvEf/tvQAf8AD0b9mL/opn/lA1T/AORq8Z/aP/4Kn/D3VtM0zwn8J/HUen3muSGDUPG97pF75GgW/wDHIkBhEs05GQgVdoJBZh2/QLWNc03w/apdapqFrpts8sdus15MsSGSRwkaAsQNzMyqB1JIA5NXQoDFgAGPU45NAHwl8Cv20v2O/wBnvwJB4Z8M/EaRw0jXN/qd5oeqSXmp3TcyXNxJ9ly8jH8AMAAAAV6J/wAPRv2Yv+imf+UDVP8A5Gr6qr5V/wCCo/8AyYn8Tf8AuGf+nS0oA+qqKKKACiiigAooooAKKKKACiiigAooooAKKKKAPnb/AIJ+/wDJqvhr/sKa7/6eb2ij/gn7/wAmq+Gv+wprv/p5vaKAOt+DH/JUvj3/ANjfZ/8AqP6RXrleR/Bj/kqXx7/7G+z/APUf0ivXKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAr5V/YF/5uN/7LN4j/wDbevqqvlX9gX/m43/ss3iP/wBt6APOf20Ph/4/uPFXgLxd4s8axvoFv8RNGtND8J6JAYrVIWuv9fdu+WmnIA4G1E5xnNfeFfMf7e3/ACJfww/7KNoH/pQa+nKqn/At2m//AEimVU+NPvFf+lTCvlX/AIKj/wDJifxN/wC4Z/6dLSvqqvlX/gqP/wAmJ/E3/uGf+nS0qST6qooooAKKKKACiiigAooooAKKKKACiiigAooooA+dv+Cfv/Jqvhr/ALCmu/8Ap5vaKP8Agn7/AMmq+Gv+wprv/p5vaKAOt+DH/JUvj3/2N9n/AOo/pFeuV5H8GP8AkqXx7/7G+z/9R/SK9coAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvlX9gX/m43/ss3iP8A9t6+qq+P9N/Y3+MXgfxV48v/AIe/tHf8IZovizxNf+J5tI/4Qazv/JnupAzDzZpizYVY142g7M7QSaAPsCivlX/hnf8Aad/6O5/8xrpf/wAco/4Z3/ad/wCjuf8AzGul/wDxygD6qr5V/wCCo/8AyYn8Tf8AuGf+nS0o/wCGd/2nf+juf/Ma6X/8crlPil+xN8dvjR4E1Pwb4y/am/tnw3qXlfa7L/hXthB5nlypKnzxTK4w8aHhhnGDwSKAPtWiiigAooooAKKKKACiiigAooooAKKKKACiiigD52/4J+/8mq+Gv+wprv8A6eb2ij/gn7/yar4a/wCwprv/AKeb2igDrfgx/wAlS+Pf/Y32f/qP6RXrleR/Bj/kqXx7/wCxvs//AFH9Ir1ygAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPnb/AIJ+/wDJqvhr/sKa7/6eb2ij/gn7/wAmq+Gv+wprv/p5vaKANfxV+yyPEHjzxL4p0r4q/ETwZN4guYby907w7qFnFaGaO1htg6rJayMCY7ePOWPI7dKof8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQAf8ADJ+r/wDRwXxg/wDBtp//AMg0f8Mn6v8A9HBfGD/wbaf/APINFFAB/wAMn6v/ANHBfGD/AMG2n/8AyDR/wyfq/wD0cF8YP/Btp/8A8g0UUAH/AAyfq/8A0cF8YP8Awbaf/wDINH/DJ+r/APRwXxg/8G2n/wDyDRRQB6P8EfhDpXwH+GeleCNFv9S1PTtOkuZUvNXkjkupWnuJbiQu0aIp+eZsYUcY6nmiiigD/9k=)

| Acc | Description | 1.12 | 1.13 | 1.14 | 1.16 | 1.17 | 1.18 | 1.19 | | Veh | 2.7 | 2.10 |
|-----|-------------|------|------|------|------|------|------|------|--- |-----|-----|------|
| 1 | Veh 1 hits offside of Veh 2 whilst moving to nearside lane on main carriageway of motorway, 100 metres from junction | 1 | 4 | 3 | 00 | | | | | 001 002 | 11 18 | 0 0 |
| | | | | | | | | | | | | |
| 2 | Veh 1 crashes on exit slip road, 50m. from junction with motorway main carriageway and 50m. from roundabout | 1 | 4 | 7 | 00 | | | | | | 001 | 18 | 0 |
| | | | | | | | | | | | | |
| 3 | Veh 1 crashes into rear of Veh 2 which is waiting in queue on slip road, 10m from junction with roundabout | 1 | 4 | 7 | 01 | 2 | 3 | 3102 | | 001 002 | 04 03 | 1 1 |
| | | | | | | | | | | | | |
| 4 | Veh 1 pulls out from slip road onto roundabout and hits nearside of Veh 2 | 3 | 3102 | 1 | 01 | 2 | 1 | 4 | | 001 002 | 18 04 | 4 8 |
| | | | | | | | | | | | | |
| 5 | Veh 1 crashes into rear of Veh 2 which brakes suddenly on roundabout more than 20m. from any entry/exit roads | 3 | 3102 | 1  | | | | | | 001 002 | 18 04 | 0 0 |
| | | | | | | | | | | | | |
| 6 | Veh 1 joining main carriageway of motorway from entry slip collides with Veh 2 which is in nearside lane | 1 | 4 | 3 | 05 | 4 | 1 | 4 | | 001 002 | 12 18 | 7 8 |
    """)

st.markdown(
    '''
## Import Dependencies

Lets import the modules we will be using, and add a little global styling.

#### Data manipulation
conda install -c conda-forge pandas  
`import pandas as pd`  
  
conda install -c conda-forge numpy  
`import numpy as np`  
  
conda install -c conda-forge openpyxl  
`import openpyxl`  
  
conda install -c conda-forge missingno  
`import missingno as msno`  
  
`import os`  
  
`import io`
#### Data visualisation
conda install -c conda-forge matplotlib  
`import matplotlib.pyplot as plt`  
`from matplotlib import rcParams`  
conda install -c conda-forge seaborn  
`import seaborn as sns`  
  
conda install -c conda-forge folium  
`import folium as flm`  
`import altair as alt`  
  
`from datetime import datetime`  
`import statistics` 
  
#### Apply some cool styling
`rcParams['figure.figsize'] = (12,  6)`  
`sns.set(style='darkgrid', palette='pastel', font_scale=1)`  

#### For showing all columns in Pandas
`pd.set_option('display.max_columns', None)`  

#### This ignores the depreciation warnings etc
`import warnings`  
`warnings.filterwarnings("ignore")`
    ''')

st.markdown('## Importing and Loading Data into DataFrame')

st.markdown('Lets import the dataset using pandas `pd.read_parquet` as `df_accident`, and have a quick look at some random rows.')

df_accident = pd.read_parquet(
    'data/sample.parquet')

# I like to use this method as it shows random rows.
st.dataframe(df_accident.sample(n=10))

st.markdown(
    '''
### Data Shape
Lets have a look ar the shape of the data.
    ''')

st.dataframe(df_accident.shape)

g = msno.matrix(df_accident)
st.pyplot(g.figure)


st.markdown(
    '''
We have 500,000 rows (values) and 34 columns (features).

### Data Types
Lets have a look at the data types.
    ''')

buffer = io.StringIO()
df_accident.info(buf=buffer)
s1 = buffer.getvalue()

st.text(s1)

st.markdown(
    '''
#### Data Types
|Data Type | Number|
|---|---:|
|object | 21|
|float | 10|
|integer | 3 |

## Data Cleaning
We will check the data for missing values, duplicates and incorrect data types.
- Missing Values - We will examine the missing values and use CCA to determine removal or replacement
- Duplicates - We will remove any duplicates
- Data Types - We will correct any data type errors, for example dates and times, as we go through the features

### Missing Values
Lets check for missing data.
    ''')

st.dataframe(df_accident.isnull().sum().sort_values(ascending=False))

st.write(
    'We have a total of:',
    sum(df_accident.isnull().sum().sort_values(ascending=False)),
    'missing values.'
    )

st.markdown(
    '''
We have a lot of missing or NaN data (1,014,501). We will need to take a good look to see what is going on.  
  
We will start with the columns (features).

### Dropping Columns

Before we look at missing row data, data types and errors lets investigate the features and take out what we do not need.  
  
Lets highlight the columns we can drop and the reasons behind our decision:  
- **LSOA_of_Accident_Location** - As this is for statistical purposes it is not needed.
- **2nd_Road_Number** - Not needed as we have the class of the road and the geographical location.
- **Did_Police_Officer_Attend_Scene_of_Accident** - This happened after the event.
- **Location_Northing_OSGR** - Secondary geographical locator.
- **Location_Easting_OSGR**  - Secondary geographical locator.
- **InScotland** - Not needed as we have geolocation and Local Authority information.
- **1st_Road_Number** - Not needed as we have the class of the road and the geographical location.
- **Local_Authority_(Highway)** - Not needed as we have Local_Authority_(District) .
- **Accident_Index** - Not required for EDA however will leave in place so we can merge later if and when required.

Lets have a look **'Did_Police_Officer_Attend_Scene_of_Accident'**.
    ''')

st.dataframe(df_accident['Did_Police_Officer_Attend_Scene_of_Accident'].value_counts(dropna=False, ascending=False))

st.markdown('Lets name the values according to the Police Form.')

df_accident[
    'Did_Police_Officer_Attend_Scene_of_Accident'] = df_accident[
    'Did_Police_Officer_Attend_Scene_of_Accident'
].map({1.0: 'Yes', 2.0: 'No', 3.0: 'Self Reported'})

df_accident['Did_Police_Officer_Attend_Scene_of_Accident'].replace(np.NaN, 'Self Reported', inplace=True)

st.dataframe(df_accident[
    'Did_Police_Officer_Attend_Scene_of_Accident'
].value_counts(dropna=False, ascending=False))


st.markdown('There is only 5,637 instances of **Self Reporting**, the remaining 2,041,6190 were reported on site at the accident or at a Police station, by a Police Officer or member of civilian staff.')

df_accident.groupby('Did_Police_Officer_Attend_Scene_of_Accident')['Accident_Severity'].value_counts(ascending=False)

df_accident.drop([
    'Location_Northing_OSGR',
    'Location_Easting_OSGR',
    'LSOA_of_Accident_Location',
    '1st_Road_Number',
    '2nd_Road_Number',
    'Did_Police_Officer_Attend_Scene_of_Accident',
    'Local_Authority_(Highway)',
    'InScotland'
    ], axis=1, inplace=True)

df_accident.columns

df_accident.shape

st.write(
    'We now have a total of:',
    sum(df_accident.isnull().sum().sort_values(ascending=False)),
    'missing values.'
    )

df_accident.isnull().sum().sort_values(ascending=False)


st.markdown(
    '''
Lets take a look at the columns with the most missing data.
| Column Name         | Count        |
| --------------------|--------------|
| 1st_Road_Class      | 305589       |
| 2nd_Road_Class      | 844272       |
| Weather_Conditions  | 21392       |
| Road_Type           | 7266       |
| Light_Conditions    | 2084       |
| Road_Surface_Conditions | 1189   |

### 1st_Road_Class
    ''')

st.dataframe(df_accident['1st_Road_Class'].value_counts(dropna=False, ascending=False))

st.markdown('### 2nd_Road_Class')

st.dataframe(df_accident['2nd_Road_Class'].value_counts(dropna=False, ascending=False))

st.markdown(
    '''
After investigating **1st_** and **2nd_road_class** features we found that in the [Police form](https://www.gov.uk/government/publications/stats19-forms-and-guidance) it shows us that there are two options which are very similar:  
  
- **Not Known**
- **Unclassified**
  
**Not Known** would indicate that the Police Officer does not know the road classification.  
**Unclassified** relates to a road of no classification, a side road or housing estate road.  
It would be logical to believe that there must be a **1st_Road_Class**, therefore the  
**1st_Road_Class**, **None**, value represents an **Unclassified** road or a **Not Known** road.  
  
The **2nd_road_class**, **None**, value represents that only one road was involved, and no junction.
  
Lets replace the values in **1st_Road_Class** with **Unclassified** and **2nd_road_class** with **No 2nd Road**  
  
We also found that a **2nd_road_class** can only be added if there is a **junction** involved.  
    ''')

df_accident['1st_Road_Class'].replace(np.NaN, 'Unclassified', inplace=True)
df_accident['1st_Road_Class'].value_counts(dropna=False, ascending=False)

df_accident['2nd_Road_Class'].replace(np.NaN, 'No 2nd Road', inplace=True)
df_accident['2nd_Road_Class'].value_counts(dropna=False, ascending=False)

st.markdown('### Weather_Conditions')

df_accident['Weather_Conditions'].value_counts(dropna=False, ascending=False)

st.markdown("We will change **'Weather_Conditions'** NaN values with the `mode`.")

df_accident['Weather_Conditions'].replace(
    'Data missing or out of range', 'Fine no high winds', inplace=True)
df_accident['Weather_Conditions'].value_counts(
    dropna=False, ascending=False)

st.markdown("We have not made any significant change using **`mode`** ('Fine no high winds').")

st.markdown('### Light_Conditions')

df_accident['Light_Conditions'].value_counts(dropna=False, ascending=False)

st.markdown("We will change **'Light_Conditions'** NaN values with the **`mode`** ('Daylight').")

df_accident['Light_Conditions'].replace(
    'Data missing or out of range', 'Daylight', inplace=True)
df_accident['Light_Conditions'].value_counts(
    dropna=False, ascending=False)

st.markdown("We have not made any significant change using **`mode`** ('Daylight').")

st.markdown('### Road_Surface_Conditions')

df_accident['Road_Surface_Conditions'].value_counts(
                                            dropna=False, ascending=False
                                        )

st.markdown("We will change **'Road_Surface_Conditions'** NaN values with the **`mode`** ('Dry').")

df_accident['Road_Surface_Conditions'].replace(
    'Data missing or out of range', 'Dry', inplace=True)
df_accident['Road_Surface_Conditions'].value_counts(
    dropna=False, ascending=False)

st.markdown("We have not made any significant change using **`mode`** ('Dry').")

df_accident.shape

st.write(
    'We now have a total of:',
    sum(df_accident.isnull().sum().sort_values(ascending=False)),
    'missing values.'
)

st.markdown(
    '''
So far we have lost no rows and reduced the missing value count down to 7,022.  
  
We will drop the rows containing the left over NaN values.
    ''')

st.dataframe(df_accident.head(n=10))

df_accident['Light_Conditions'].value_counts()

length_before = len(df_accident)
df_accident.dropna(inplace=True)
lnth = (length_before - len(df_accident)) / length_before
df_accident.reset_index(drop=True)
st.write(f'We have dropped only {lnth:.2%} of the rows.')

st.markdown('We have only lost 0.22% of the original rows.')

st.markdown('Lets confirm we have no missing values.')

st.write(
    'We now have a total of:',
    sum(df_accident.isnull().sum().sort_values(ascending=False)),
    'missing values.'
    )

f = msno.matrix(df_accident)
st.pyplot(f.figure)

st.markdown(
    '''
### Duplicates
Lets check for and remove any duplicate rows.
    ''')
st.write(f'Number of duplicated rows:', df_accident.duplicated().sum())
st.write(f'Rows and columns:', df_accident.shape)

st.markdown(
    '''
### Data Types and Formatting
We will convert all column names to lowercase change special characters, **-** with **_** and **()** with **''** and remove white space, if any, for standardisation.
    ''')

df_accident.columns = df_accident.columns.str.strip().str.lower()
df_accident.columns = df_accident.columns.str.replace('-', '_')
df_accident.columns = df_accident.columns.str.replace('(', '')
df_accident.columns = df_accident.columns.str.replace(')', '')

st.code(df_accident.columns)

st.write('Now we can check datatypes.')

buffer = io.StringIO()
df_accident.info(buf=buffer)
s2 = buffer.getvalue()

st.text(s2)

st.write('Lets take a sample row and have a look at each feature in more detail.')

st.dataframe(df_accident.sample(1))

st.write("Lets convert the categorical data to `category` and the numerical data to `int` and `float`.")

for col in [
    'accident_index',
    '1st_road_class', '2nd_road_class', 'speed_limit',
    'accident_severity', 'carriageway_hazards',
    'junction_control', 'junction_detail',
    'light_conditions', 'local_authority_district',
    'pedestrian_crossing_human_control',
    'pedestrian_crossing_physical_facilities',
    'police_force', 'road_surface_conditions',
    'road_type', 'special_conditions_at_site',
    'urban_or_rural_area', 'weather_conditions']:
    df_accident[col] = df_accident[col].astype('category')

for col in ['number_of_casualties', 'number_of_vehicles']:
    df_accident[col] = df_accident[col].astype('int')

for col in ['latitude', 'longitude']:
    df_accident[col] = df_accident[col].astype('float64')

buffer = io.StringIO()
df_accident.info(buf=buffer)
s3 = buffer.getvalue()

st.text(s3)

st.markdown(
    """
Now things are starting to look much better. We will look at dates and times later, as we plan to make use of these columns.

## Average Counts
Lets see how the **RTC's** have tracked over our time period.
    """)

years = range(2005, 2018)
yr_slight = df_accident['accident_severity'] == 'Slight'
yr_serious = df_accident['accident_severity'] == 'Serious'
yr_fatal = df_accident['accident_severity'] == 'Fatal'

yearly_numbers = {}
for year in years:
    yr = df_accident['year'] == year
    yearly_numbers[year] = [
        yr.sum(), int(yr.sum() /12), int(yr.sum() / 52),
        int(yr.sum() /365), (yr & yr_slight).sum(),
        (yr & yr_serious).sum(), (yr & yr_fatal).sum()
    ]


df_avgs = pd.DataFrame.from_dict(yearly_numbers, ).T
df_avgs.reset_index(inplace=True)
df_avgs.columns = (
    'year', 'yearly', 'monthly', 'weekly',
    'daily', 'slight', 'serious', 'fatal')
df_avgs['year'] = df_avgs['year'].astype('str')
st.dataframe(df_avgs)

col1, col2 = st.columns((1, 1))
with col1:
    chart = alt.Chart(df_avgs).mark_bar().encode(
        x=alt.X('year', axis=alt.Axis(title='Collisions by Year')),
        y=alt.Y('yearly', axis=alt.Axis(title='Number of Collisions')))
    st.altair_chart(chart, use_container_width=True)
with col2:
    chart = alt.Chart(df_avgs).mark_bar().encode(
        x=alt.X('year', axis=alt.Axis(title='Slight Collisions by Year')),
        y=alt.Y('slight', axis=alt.Axis(title='Number of Slight Collisions')))
    st.altair_chart(chart, use_container_width=True)
col1, col2 = st.columns((1, 1))
with col1:
    chart = alt.Chart(df_avgs).mark_bar().encode(
        x=alt.X('year', axis=alt.Axis(title='Serious Collisions by Year')),
        y=alt.Y('serious', axis=alt.Axis(title='Number of Serios Collisions')))
    st.altair_chart(chart, use_container_width=True)
with col2:
    chart = alt.Chart(df_avgs).mark_bar().encode(
        x=alt.X('year', axis=alt.Axis(title='Fatal Collisions by Year')),
        y=alt.Y('fatal', axis=alt.Axis(title='Number of Fatal Collisions')))
    st.altair_chart(chart, use_container_width=True)

st.write("As expected we can see an overall decline in RTC's over the years, however in 2014 we see a rise in collisions. This rise continues for serious accidents.")

st.markdown('## Collision Severity Spread')

round(df_accident[
    'accident_severity'
].value_counts(ascending=False, normalize=True), 3) * 100

st.markdown(
    """
| Severity         | Percentage        |
| --------------------|--------------|
| **Slight**      | **84.7%**       |
| **Serious**      | **14.0%**      |
| **Fatal**  | **1.3%**    |
  
With less than 15% of Serious and Fatal accidents we have a very heavy bias towards **Slight** collisions.  
  
Lets see if we can uncover any large increases in the **Serious** and **Fatal** collisions.  
We have benchmarks of:
- 14.0% for **Serious** collision
- 1.3% for **Fatal** collisions
    """)

st.write('#### Final DataFrame')
st.dataframe(df_accident)
# ## Features

# Lets remind ourselves of our features.
#   

# st.code(df_accident.columns)

# ### Feature: **'1st_road_class'** and **'2nd_road_class'**

# st.write(print(df_accident['1st_road_class'].value_counts(
#                                                dropna=False, ascending=False
#                                                )))
# st.write(print(f'Total rows: ', df_accident['1st_road_class'].count()))

# st.write(print(df_accident[
#   '2nd_road_class'].value_counts(
#       dropna=False, ascending=False)))
# st.write(print(f'Total rows: ', df_accident['1st_road_class'].count()))

# # We can see that most RTC's happen on **'A Roads'** and **'Unclassified Roads'**. Nearly half of all RTC's do not involve a **second road**.

# fig, ax = plt.subplots(1, 2, figsize=(15, 10))
# for axes in ax.flatten():
#   plt.sca(axes)
#   plt.xticks(rotation = 45)

# sns.histplot(data=df_accident,
#            x='1st_road_class',
#            hue='accident_severity',
#            ax=ax[0]
#            )

# sns.histplot(data=df_accident,
#            x='2nd_road_class',
#            hue='accident_severity',
#            ax=ax[1]
#            )

# plt.show()

# # From the visualisation we can detect that most fatal RTC's happen on either an **'A' Road only**, or an **'A'** Road with an **'Unclassified'** 2nd Road (pulling out of, or into a side road).

# # ### Feature - **'accident_severity'**

# print(df_accident[
#   'accident_severity'].value_counts(
#       dropna=False, ascending=False))
# print(f'\n Total rows: ', df_accident['accident_severity'].count())

# # We can see that most RTC's are **'Slight'** RTC's with **'Fatal'** the lowest.

# data = df_accident['accident_severity'].value_counts()
# labels = ['Slight', 'Serious', 'Fatal']
# colors = sns.color_palette('pastel')[0:3]
# explode=[0, 0, 0.1]
# plt.pie(
#   data, labels = labels,
#   colors = colors,
#   explode=explode, autopct='%.1f%%'
# )

# plt.show()

# df_accident[
#   'accident_severity'].value_counts().plot(kind='bar', alpha=0.6, rot=0)

# st.markdown(
#   '''
#   ### Features - **'date'** and **'time'**
#   Lets take a look at the data type of the **'date'** and **'time'** columns.
#   ''')

# st.write(f'Data type of Date Column: {df_accident["date"].dtype}')
# st.write(f'Data type of Time Column: {df_accident["time"].dtype}')
# df_accident['date'][0]
# df_accident['time'][0]

# st.markdown("Let's convert our **'date'** and **'time'** columns using `pd.to_datetime`.")

# df_accident['date'] = pd.to_datetime(df_accident['date'], dayfirst=True)
# df_accident['time'] = pd.to_datetime(df_accident['time'], format='%H:%M')
# st.write(f"Data type of date Column: {df_accident['date'].dtype}")
# st.write(f"Data type of time Column: {df_accident['time'].dtype}")

# # #### New Feature - **'hour'**
# # Lets create the new feature **'hour'** in order to simplify the time view.

# df_accident['hour'] = df_accident['time'].apply(lambda x:x.hour)
# # df_accident.drop('time', axis=1, inplace=True)

# # ### Feature - **'hour'**

# plt.figure(figsize=(15,10))
# ax = sns.histplot(
#   data=df_accident, x='hour',
#   palette='pastel', bins=24, hue='accident_severity'
#                 )

# plt.show()

# # We can see two peaks, one in the morning and one in the afternoon.  
# #   
# # Lets take a look at the **RTC** severities.

# fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# sns.histplot(
#            df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='hour', color='lightgreen',
#            alpha=0.6, bins=24, ax=ax[0], kde=True
#            ).set(title=f'Accident_Severity: Slight')

# sns.histplot(
#            df_accident[df_accident['accident_severity'] == 'Serious'],
#            y='hour',
#            alpha=0.6, bins=24, ax=ax[1], kde=True
#            ).set(title=f'Accident_Severity: Serious')

# sns.histplot(
#            df_accident[df_accident['accident_severity'] == 'Fatal'],
#            y="hour", color='orange',
#            alpha=0.6, bins=24, ax=ax[2], kde=True
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # We can see most RTC's happen betwen 8am and 7pm, with two peaks.  
# # - For Slight and Serious we have peaks at 9am and 3-5pm. This would appear to be rush hour traffic.
# # - For **'Fatal'** we have a peak at 3-5pm  
# # 
# # **'Fatal'** RTC's do appear to maintain a higher rate until midnight, where it then starts to tail off.

# # ### Feature - **'day_of_week'**

# # Here we can look to see what week day trends we have.

# df_accident.day_of_week = pd.Categorical(
#   values=df_accident.day_of_week,
#   categories=[
#       'Monday', 'Tuesday', 'Wednesday', 'Thursday',
#       'Friday', 'Saturday', 'Sunday']) # Set the order for the columns.
# df_accident.sort_values(['day_of_week']) # Sort the columns.

# plt.figure(figsize=(15,15))
# ax = sns.histplot(
#   data=df_accident, x='day_of_week',
#   palette='pastel', bins=7, hue='accident_severity'
#   )

# plt.show()

# # There is a definite increase in **RTC's** on **Friday's**.  
# #   
# # Lets have a look at the severity on each day.

# fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='day_of_week', color='lightgreen',
#            alpha=0.6, bins=7, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Serious'],
#            y='day_of_week',
#            alpha=0.6, bins=7, ax=ax[1]
#            ).set(title=f'Accident_Severity: Serious')

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Fatal'],
#            y='day_of_week', color='orange',
#            alpha=0.6, bins=7, ax=ax[2]
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # We can see from the visuals that Friday's have the highest rates of **'Slight'** and **'Serious'** RTC's with the exception of **'Fatal'** RTC's.  
# #   
# # **'Fatal'** RTC's peak on Saturdays and Sundays.

# # #### New Features - **'day'** and **'month'**
# # Lets create the two features, **'day'** and **'month'**.

# df_accident["day"] = df_accident["date"].apply(lambda x:x.day)
# df_accident["month"] = df_accident["date"].apply(lambda x:x.month)
# df_accident['month'] = pd.to_datetime(
#                                     df_accident['month'], format='%m'
#                                    ).dt.month_name().str.slice(stop=3)
# df_accident.shape

# # ### Feature - **'day'**
# # Here we can have a look at the day of the month to see what trends we may have.

# plt.figure(figsize=(15,15))
# ax = sns.histplot(data=df_accident, x='day',
#                 palette='pastel', bins=31,
#                 hue='accident_severity', kde=True
#                 )

# plt.show()

# # There is a very slight drop off from the 25th to the 30th of the month.  
# #   
# # The drop offs on the 31st is expected due to only being 7 per year.  
# #   
# # Lets see how **RTC** severity looks.

# fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            x='day', color='lightgreen',
#            alpha=0.6, bins=31, ax=ax[0], kde=True,
#            ).set(title=f"Accident_Severitx: Slight")

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Serious'],
#            x='day',
#            alpha=0.6, bins=31, ax=ax[1], kde=True,
#            ).set(title=f'Accident_Severitx: Serious')

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Fatal'],
#            x='day', color='orange',
#            alpha=0.6, bins=31, ax=ax[2], kde=True,
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # The visualisation of **RTC** severities does not reveal much.

# # ### Feature - **'month'**

# # Lets check correlations against **'month'**.

# plt.figure(figsize=(15,15))
# ax = sns.histplot(data=df_accident, x='month',
#                 palette='pastel', bins=12, hue='accident_severity'
#                 )

# plt.show()

# # We can see a rise leading towards the end of year, lets try and explore what is happening.

# df_accident.month = pd.Categorical(values=df_accident.month,
#                                   categories=['Jan', 'Feb', 'Mar', 'Apr',
#                                               'May', 'Jun', 'Jul',
#                                               'Aug', 'Sep', 'Oct',
#                                               'Nov', 'Dec'
#                                               ]
#                                   ) # Set the order for the columns
# df_accident.sort_values(['month'])

# fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            x='month', color='lightgreen',
#            alpha=0.6, bins=12, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Serious'],
#            x='month',
#            alpha=0.6, bins=12, ax=ax[1],
#            ).set(title=f'Accident_Severity: Serious')

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Fatal'],
#            x='month', color='orange',
#            alpha=0.6, bins=12, ax=ax[2],
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # We can see from the visualisations that there is a upward trend heading towards the end of the year.  
# #   
# # We can note that January does see a rise..
# # 

# # #### New Feature **'season'**
# # Lets create a new feature **'season'** with values, **'Spring', 'Summer', 'Autumn', and 'Winter'** to see if we can reveal a trend.

# # ### Feature - **'season'**

# df_accident["season"] = df_accident["month"].map({'Mar': "Spring", 'Apr': "Spring",
#                                                 'May': "Spring", 'Jun': "Summer",
#                                                 'Jul': "Summer", 'Aug': "Summer",
#                                                 'Sep': "Autumn", 'Oct': "Autumn",
#                                                 'Nov': "Autumn", 'Dec': "Winter",
#                                                 'Jan': "Winter", 'Feb': "Winter"}
#                                                )
# df_accident.shape

# # Lets have a look at how the **RTC's** look.

# data = df_accident['season'].value_counts()
# labels = ['Autumn', 'Summer', 'Spring', 'Winter']
# colors = sns.color_palette('pastel')[0:4]
# explode = [0.1, 0, 0, 0]

# plt.pie(
#   data, labels = labels, colors = colors,
#   explode=explode, autopct='%.2f%%'
# )

# plt.show()

# df_accident.season = pd.Categorical(values=df_accident.season,
#                                   categories=['Winter', 'Spring',
#                                               'Summer', 'Autumn'
#                                               ]
#                                   ) # Set the order for the column as you want
# df_accident.sort_values(['season']) # Sort the column

# fig, ax = plt.subplots(1, 3, figsize=(20, 7))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            x='season', color='lightgreen',
#            alpha=0.6, bins=50, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            x='season',
#            alpha=0.6, bins=50, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            x='season', color='orange',
#            alpha=0.6, bins=50, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()  ## and plot

# # We can see by the visualisations that there is an increase across all of the severities of an **RTC** drawing towards **'Autumn'**.  
# #   
# # As from our previous visualisation we can see **'Serious'** **RTC's** starting to peak in **'Summer'**  
# #   
# # **'Summer'** and **'Autumn'** have the most RTC's..

# # #### New Feature **'bank_holidays'**
# # Lets have a look to see if UK Bank Holidays have an impact.

# # 

# # ### Feature - **'bank_holidays'**

# # Lets take a look at the bank holiday breaks.  
# # To do this we will create a function that takes in the date of the start and end of the break, for example, Mayday is on a Monday, so the start of the break would be the previous Saturday and the duration would be 3.

# def holidays(name, start, end, yr, days, svty):
#   '''
#   A function to compare average holiday RTC counts against
#   average daily counts with percentage rise or fall

#   Args:
#       name (str): Name of the holiday, for example 'Easter'
#       start (str): Start date of the holiday period. '%m/%d/%Y'
#       end (str): End date of holiday period. '%m/%d/%Y'
#       yr (int): year (range: 2005 - 2010)
#       days (int): Length of holiday period
#       svty (str): Severity (Slight, Serious, Fatal)
#   '''
#   holiday = len(
#       df_accident[
#           ((df_accident['accident_severity'] == svty)
#           & (df_accident['date'] >= start))
#           & ((df_accident['date'] <= end))
#       ]
#   ) / days
#   severity = len(
#       df_accident[
#           ((df_accident['accident_severity'] ==  svty)
#            & (df_accident['year'] == yr))
#       ]
#   ) / 365
#   perc = (int(holiday) - int(severity)) / int(severity)

#   print(
#       'The average', svty, 'RTCs over', name, yr, 'were',
#       int(holiday), 'per day.\nThe daily average for the year is',
#       int(severity), 'fatal accidents per day, a change of ' f'{perc:.1%}\n'
#   )

# holidays('Easter', '03/25/2005', '03/28/2005', 2005, 4, 'Fatal')
# holidays('Easter', '04/14/2006', '04/17/2006', 2006, 4, 'Fatal')
# holidays('Easter', '04/06/2007', '04/09/2007', 2007, 4, 'Fatal')
# holidays('Easter', '03/21/2008', '03/24/2008', 2008, 4, 'Fatal')
# holidays('Easter', '04/10/2009', '04/13/2009', 2009, 4, 'Fatal')
# holidays('Easter', '04/02/2010', '04/05/2010', 2010, 4, 'Fatal')

# # As we can see there is an increase in **fatal RTC's** over the **'Easter'** period for 4 of the 6 years.  
# # Over the six years it is about a **8%** rise.

# holidays('Mayday', '04/30/2005', '05/02/2005', 2005, 3, 'Fatal')
# holidays('Mayday', '04/29/2006', '05/01/2006', 2006, 3, 'Fatal')
# holidays('Mayday', '05/05/2007', '05/07/2007', 2007, 3, 'Fatal')
# holidays('Mayday', '05/03/2008', '05/05/2008', 2008, 3, 'Fatal')
# holidays('Mayday', '05/02/2009', '05/04/2009', 2009, 3, 'Fatal')
# holidays('Mayday', '05/01/2010', '05/03/2010', 2010, 3, 'Fatal')

# # As we can see there is an increase in **fatal RTC's** over the **'Mayday'** period for 3 of the 6 years.  
# # Over the six years it is about a **10%** rise.

# holidays('Spring', '05/26/2005', '05/28/2005', 2005, 3, 'Fatal')
# holidays('Spring', '05/25/2006', '05/27/2006', 2006, 3, 'Fatal')
# holidays('Spring', '05/24/2007', '05/26/2007', 2007, 3, 'Fatal')
# holidays('Spring', '05/22/2008', '05/24/2008', 2008, 3, 'Fatal')
# holidays('Spring', '05/21/2009', '05/23/2009', 2009, 3, 'Fatal')
# holidays('Spring', '05/27/2010', '05/29/2010', 2010, 3, 'Fatal')

# # As we can see there is an increase in **fatal RTC's** over the **'Spring'** period for 3 of the 6 years.  
# # Over the six years it is about a **3%** rise.

# holidays('Summer Break', '08/28/2005', '08/30/2005', 2005, 3, 'Fatal')
# holidays('Summer Break', '08/26/2006', '08/28/2006', 2006, 3, 'Fatal')
# holidays('Summer Break', '08/25/2007', '08/27/2007', 2007, 3, 'Fatal')
# holidays('Summer Break', '08/23/2008', '08/25/2008', 2008, 3, 'Fatal')
# holidays('Summer Break', '08/29/2009', '08/31/2009', 2009, 3, 'Fatal')
# holidays('Summer Break', '08/28/2010', '08/30/2010', 2010, 3, 'Fatal')

# # As we can see there is an increase in **fatal RTC's** over the **'Summer Bank Holiday'** period for 4 of the 6 years.  
# # Over the six years it is about a **34%** rise.

# holidays('Christmas and New Year', '12/24/2005', '01/02/2006', 2006, 10, 'Fatal')
# holidays('Christmas and New Year', '12/23/2006', '01/01/2007', 2007, 10, 'Fatal')
# holidays('Christmas and New Year', '12/25/2007', '01/01/2008', 2008, 8, 'Fatal')
# holidays('Christmas and New Year', '12/25/2008', '01/01/2009', 2009, 8, 'Fatal')
# holidays('Christmas and New Year', '12/25/2009', '01/01/2010', 2010, 8, 'Fatal')

# # As we can see there is an decrease in **fatal RTC's** over the **'Christmas and New Year'** period for 3 of the 5 years.  
# # Over the six years it is about a **19%** reduction. (excluded 2005 as not a full period and does not include Christmas 2010)

# # ### Feature - **'junction_detail'**

# # Lets check the the data type and values.

# df_accident['junction_detail'].dtype

# df_accident['junction_detail'].value_counts(dropna=False, ascending=False)

# # We appear to have a nice set of values with the exception of **'Data missing or out of range'**.  
# #   
# # As there are only four of these, we will replace with them with the `mode` (Not at junction or within 20 metres).

# df_accident['junction_detail'].replace('Data missing or out of range', 'Not at junction or within 20 metres', inplace=True)
# df_accident['junction_detail'].value_counts(dropna=False, ascending=False)

# # Lets visualise the data.

# plt.figure(figsize=(15,15))
# ax = sns.histplot(data=df_accident, x='junction_detail', bins=5, hue='accident_severity')

# ax.set_xticklabels(ax.get_xticklabels(),
#                  rotation=20,
#                  horizontalalignment='right'
#                  )
# plt.show()


# # We can see that **RTC's** are more inclined to happen **'Not at junction or within 20 metres'** or **'T or staggered junction'**.  
# #   
# # **'Serious'** and **'Fatal'** **'RTC's'** are also more inclined to happen at these junction types.

# df_accident.junction_detail = pd.Categorical(values=df_accident.junction_detail,
#                                            categories=[
#                                               'Not at junction or within 20 metres',
#                                               'T or staggered junction',
#                                               'Crossroads',
#                                               'Roundabout',
#                                               'Private drive or entrance',
#                                               'Other junction',
#                                               'Slip road',
#                                               'More than 4 arms (not roundabout)',
#                                               'Mini-roundabout',
#                                               'Authorised person'
#                                            ]
#                                            ) # Sort the attributes
# df_accident.sort_values(['junction_detail']) # Sort the column

# fig, ax = plt.subplots(2, 2, figsize=(25, 25))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='junction_detail', color='lightgreen',
#            alpha=0.6, bins=50, ax=ax[0, 0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='junction_detail',
#            alpha=0.6, bins=50, ax=ax[0, 1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='junction_detail', color='orange',
#            alpha=0.6, bins=50, ax=ax[1, 0]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # ### Feature - **'junction_control'**

# # Lets check the data type and values.

# df_accident['junction_control'].dtype

# df_accident['junction_control'].value_counts(dropna=False, ascending=False)

# # We appear to have two similar features **'Auto traffic signal'** and **'Auto traffic sigl'**. This appears to be a typo so lets merge the values.  
# #   
# # We also have two values, **'Not at junction or within 20 metres'** and **'Data missing or out of range'**. The Police reporting form only has four options:
# # - Give way or uncontrolled
# # - Auto traffic signal
# # - Stop sign
# # - Authorised person
# # 
# # We will assume that the absence of an entry on the form indicates none of the four options were valid.  
# #   
# # Lets merge **'Data missing or out of range'** with **'Not at junction or within 20 metres'**.

# df_accident[
#     "junction_control"
# ] = df_accident[
#     "junction_control"
# ].map({"Auto traffic sigl": "Auto traffic signal"}
#     ).fillna(df_accident["junction_control"])

# df_accident[
#     "junction_control"
# ] = df_accident[
#     "junction_control"
# ].map({"Data missing or out of range": "Not at junction or within 20 metres"}
#     ).fillna(df_accident["junction_control"])

# df_accident['junction_control'].value_counts(dropna=False, ascending=False)

# # Now we have cleaned up the feature lets see how this looks against **'Severity'**.

# plt.figure(figsize=(15,15))
# ax = sns.histplot(
#   data=df_accident, x='junction_control',
#   bins=5, hue='accident_severity'
#   )

# ax.set_xticklabels(ax.get_xticklabels(),
#                  rotation=20,
#                  horizontalalignment='right'
#                  )
# None

# plt.show()

# # We can see that most **RTC's** happen at a **'give way'** or **'uncontrolled junction'**.  
# #   
# # Lets check individual values against **'severity'**.

# df_accident.junction_control = pd.Categorical(
#      values=df_accident.junction_control,
#      categories=['Give way or uncontrolled',
#                  'Not at junction or within 20 metres',
#                  'Auto traffic signal',
#                  'Stop sign',
#                  'Authorised person'
#                 ]
# ) # Set the order for the column
# df_accident.sort_values(['junction_control']) # Sort the column

# fig, ax = plt.subplots(2, 2, figsize=(25, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='junction_control', color='lightgreen',
#            alpha=0.6, bins=50, ax=ax[0, 0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='junction_control',
#            alpha=0.6, bins=50, ax=ax[0, 1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='junction_control', color='orange',
#            alpha=0.6, bins=50, ax=ax[1, 0]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # Lets have a look at the correlations between **'junction_control'** and **'junction_detail'**.

# df_accident.groupby('junction_control')['junction_detail'].value_counts(ascending=True)

# # We can see some anomalies in the **Not at junction or within 20 metres** values.  
# #   
# # As stated in the Police instructions if **Not at junction or within 20 metres** is selected then **no** options in **junction_detail** should be selected.
# # 

# plt.figure(figsize=(18,10))

# ax = df_accident.groupby(
#   'junction_control'
# )['junction_detail'].value_counts(ascending=True).plot(kind='bar', alpha=0.6)
# for p in ax.patches:
#   ax.annotate(
#       str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
#               )

# plt.show()

# # We can see two areas of concern here, **Not at junction or within 20 metres** and **Give way or uncontrolled**.  
# #   
# # **Not at junction or within 20 metres** and **Not at junction or within 20 metres** have the highest correlation as expected.  
# #   
# # **Give way or uncontrolled** and **T or staggered junction** also have a high correlation.  
# #   
# # The junction type with the highest number of accidents is **T or staggered junction**.
# # 
# # 

# # ### Feature - **'carriageway_hazards'**

# df_accident['carriageway_hazards'].value_counts(dropna=False,ascending=False)

# plt.figure(figsize=(15,25))
# ax = sns.histplot(data=df_accident,
#                 x='carriageway_hazards',
#                 hue='accident_severity'
#                 )
# ax.set_xticklabels(ax.get_xticklabels(),
#                  rotation=45,
#                  horizontalalignment='right'
#                  )
# None

# plt.show()

# # We can see that most RTC's happen when there are no carriageway hazards.  
# #   
# # Lets take a look at each carriageway hazard to see if we can uncover anything.

# df_accident.groupby(
#   'carriageway_hazards'
# )['accident_severity'].value_counts(ascending=True)

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='carriageway_hazards', color='lightgreen',
#            alpha=0.6, bins=31, ax=ax[0],
#            ).set(title=f"Accident_Severitx: Slight")

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Serious'],
#            y='carriageway_hazards',
#            alpha=0.6, bins=31, ax=ax[1],
#            ).set(title=f'Accident_Severitx: Serious')

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Fatal'],
#            y='carriageway_hazards', color='orange',
#            alpha=0.6, bins=31, ax=ax[2],
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # We can see here from the visualisations that the severity of an RTC rises when there is both an object on the road and when there is a previous accident.  
# #   
# # Fatal accidents are more likely to happen when there is a previous accident.

# # ### Feature - **'special_conditions_at_site'**

# # Lets have a look at the values.

# df_accident['special_conditions_at_site'].value_counts(ascending=False)

# # We appear to have two sets of similar features:  
# # - **Auto signal part defective** and **Auto sigl part defective**
# # - **Auto traffic signal - out** and **Auto traffic sigl - out**  
# #   
# # This appears to be a typo so lets merge the values.  

# df_accident['special_conditions_at_site'] = df_accident['special_conditions_at_site'
#                                             ].map({'Auto sigl part defective': 'Auto signal part defective'}
#                                                   ).fillna(df_accident['special_conditions_at_site'])
# df_accident['special_conditions_at_site'] = df_accident['special_conditions_at_site'
#                                             ].map({'Auto traffic sigl - out': 'Auto traffic signal - out'}
#                                                   ).fillna(df_accident['special_conditions_at_site'])
# df_accident['special_conditions_at_site'].value_counts(dropna=False, ascending=False)

# # Now we have a clean set of values lets have a look at them against **'Severity'**.

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='special_conditions_at_site', color='lightgreen',
#            alpha=0.6, bins=31, ax=ax[0],
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Serious'],
#            y='special_conditions_at_site',
#            alpha=0.6, bins=31, ax=ax[1],
#            ).set(title=f'Accident_Severity: Serious')

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Fatal'],
#            y='special_conditions_at_site', color='orange',
#            alpha=0.6, bins=31, ax=ax[2],
#            ).set(title=f'Accident_Severity: Fatal')

# plt.show()

# # We can see that **'special_conditions_at_site'** do not have a great impact.  
# #   
# # We do see a slight rise in all severities when there are **'Roadworks'** present.

# # ### Feature - **'light_conditions'**

# # Lets have a look **'light_conditions'**.

# df_accident['light_conditions'].value_counts(ascending=False)

# plt.figure(figsize=(15,15))
# ax = sns.histplot(
#   data=df_accident, x='light_conditions', bins=5, hue='accident_severity'
# )

# ax.set_xticklabels(ax.get_xticklabels(),
#                  rotation=20,
#                  horizontalalignment='right'
#                  )
# None

# plt.show()

# # We can see that the **RTC's** are much more common during daylight hours.  
# #   
# # Lets have la look against **'Accident Severity'**.

# df_accident.light_conditions = pd.Categorical(values=df_accident.light_conditions,
#                                            categories=[
#                                               'Daylight',
#                                               'Darkness - lights lit',
#                                               'Darkness - no lighting',
#                                               'Darkness - lighting unknown',
#                                               'Darkness - lights unlit'
#                                            ]
#                                            ) # Sort the attributes
# df_accident.sort_values(['light_conditions']) # Sort the column

# fig, ax = plt.subplots(3, 1, figsize=(15, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='light_conditions', color='lightgreen',
#            alpha=0.6, bins=5, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='light_conditions',
#            alpha=0.6, bins=5, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='light_conditions', color='orange',
#            alpha=0.6, bins=5, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # It still appears that RTC's are much more common in the daylight.  
# #   
# # However we can see in the **'Fatal'** visual that a larger proportion happen during darkness.  

# # ### Feature - **'pedestrian_crossing_human_control'**

# # Lets take a look at the values.

# df_accident['pedestrian_crossing_human_control'].value_counts(dropna=False, ascending=False)

# # We can see that these have been entered as `floats`, which is out of line with the data.  
# #   
# # Lets convert them to match the Police Accident Form.

# df_accident['pedestrian_crossing_human_control'] = df_accident[
#   'pedestrian_crossing_human_control'
# ].map(
#   {0.0 : 'None within 50m', 1.0 : 'School Crossing Patrol',
#   2.0 : 'Other Authorised Control'}
# )

# df_accident[
#   'pedestrian_crossing_human_control'
# ].value_counts(dropna=False, ascending=False)


# # Nearly all **RTC's** are not affected by this feature.  
# #   
# # Now we have mapped to the correct names lets visualise against **'Severity'** anyway.

# df_accident.pedestrian_crossing_human_control = pd.Categorical(values=df_accident.pedestrian_crossing_human_control,
#                                            categories=[
#                                               'None within 50m',
#                                               'Other Authorised Control',
#                                               'School Crossing Patrol'
#                                            ]
#                                            ) # Sort the attributes
# df_accident.sort_values(['pedestrian_crossing_human_control']) # Sort the column

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='pedestrian_crossing_human_control',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='pedestrian_crossing_human_control',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='pedestrian_crossing_human_control',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # ### Feature - **'pedestrian_crossing_physical_facilities'**

# # Lets take a look at **'pedestrian_crossing_physical_facilities'** to see if they have any useful insight.

# df_accident[
#   'pedestrian_crossing_physical_facilities'
# ].value_counts(dropna=False, ascending=False)

# # We can see that these have been entered as `floats`, which is out of line with the data.  
# #   
# # Lets convert them to match the Police Accident Form.

# df_accident[
#   'pedestrian_crossing_physical_facilities'
# ] = df_accident[
#   'pedestrian_crossing_physical_facilities'
# ].map(
#   {0.0 : 'None within 50m',
#   1.0 : 'Zebra Crossing',
#   4.0 : 'Pelican, puffin, toucan or similar non-junction pedestrian light crossing',
#   5.0 : 'Pedestrian phase at traffic signal junction',
#   7.0 : 'Footbridge or subway',
#   8.0 : 'Central refuge — no other controls'}
# )

# df_accident[
#   'pedestrian_crossing_physical_facilities'
# ].value_counts(dropna=False, ascending=False)


# # Lets have a look against **'Severity'**.

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='pedestrian_crossing_physical_facilities',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='pedestrian_crossing_physical_facilities',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='pedestrian_crossing_physical_facilities',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # We can see that this feature has little impact.  
# #   
# # However we can see an increase in **'Pedestrian phase at traffic signal junction'** for all three severities.  
# #   
# # There is also a rise at **'Pelican, puffin, toucan or similar non-junction pedestrian light crossing'**.  
# #   
# # Lets take a closer look.

# (len(df_accident.loc[(
#   df_accident['accident_severity'] != 'Slight'
# )
# & ((
#   df_accident['pedestrian_crossing_physical_facilities']
#   == 'Pelican, puffin, toucan or similar non-junction pedestrian light crossing'
#    )
# | (
#   df_accident['pedestrian_crossing_physical_facilities']
#   == 'Pedestrian phase at traffic signal junction'
#    ))
# ]) / 6) / 52

# # We are having an average of **51** serious **RTC's** per **week** at a light controlled pedestrian crossing across the UK.  
# #   
# # There are **3** **Fatal** **RTC's** per **week** at a light controlled pedestrian crossing.

# # ### Feature - **'road_surface_conditions'**

# # Lets take a look to see what difference **'road_surface_conditions'** have.

# df_accident['road_surface_conditions'].value_counts(ascending=False, dropna=False)

# # We can see that nearly a third of all accidents happen in **'Wet or damp'** conditions.

# df_accident.groupby(
#   'road_surface_conditions'
# )['accident_severity'].value_counts(ascending=False)

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='road_surface_conditions',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='road_surface_conditions',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='road_surface_conditions',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # We can see that **'Wet or damp'** has an impact across all severities.

# df_accident.groupby('road_surface_conditions')['weather_conditions'].value_counts(ascending=False)

# # ### Feature - **'road_type'**

# round(df_accident.groupby(
#   'road_type'
# )['accident_severity'].value_counts(ascending=False, normalize=True), 3) * 100

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='road_type',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='road_type',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='road_type',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # ### Feature - **'special_conditions_at_site'**

# round(df_accident.groupby(
#   'special_conditions_at_site'
# )['accident_severity'].value_counts(ascending=False, normalize=True), 3) * 100

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='special_conditions_at_site',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='special_conditions_at_site',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='special_conditions_at_site',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # ### Feature - **'urban_or_rural_area'**

# df_accident['urban_or_rural_area'].value_counts(ascending=False)

# df_accident['urban_or_rural_area'].replace('Unallocated', 'Urban', inplace=True)

# round(df_accident.groupby(
#   'urban_or_rural_area'
# )['accident_severity'].value_counts(ascending=False, normalize=True), 3) * 100

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='urban_or_rural_area',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='urban_or_rural_area',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='urban_or_rural_area',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # We can see that the **Fatal** **RTC's** in **Rural** areas are much higher than **Urban** areas.

# # 

# # 

# df_accident['local_authority_district'].value_counts(ascending=False)

# # ### Feature - **'weather_conditions'**

# df_accident['weather_conditions'].value_counts(ascending=False)

# df_accident['weather_conditions'].value_counts().plot(kind='barh')

# round(df_accident.groupby(
#   'weather_conditions'
# )['accident_severity'].value_counts(ascending=False, normalize=True), 3) * 100

# fig, ax = plt.subplots(3, 1, figsize=(20, 15))

# sns.histplot(df_accident[df_accident['accident_severity'] == 'Slight'],
#            y='weather_conditions',
#            color='lightgreen', alpha=0.6, bins=3, ax=ax[0]
#            ).set(title=f"Accident_Severity: Slight")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Serious"],
#            y='weather_conditions',
#            alpha=0.6, bins=3, ax=ax[1]
#            ).set(title=f"Accident_Severity: Serious")

# sns.histplot(df_accident[df_accident["accident_severity"] == "Fatal"],
#            y='weather_conditions',
#            color='orange', alpha=0.6, bins=3, ax=ax[2]
#            ).set(title=f"Accident_Severity: Fatal")

# plt.show()

# # ### Feature - **'number_of_casualties'**

# # A casualty is classed as any human death or personal injury. Lets have a look at the counts.

# df_accident['number_of_casualties'].sum()

# df_accident.groupby('accident_severity')['number_of_casualties'].sum()

# df_accident.groupby('police_force')['number_of_casualties'].sum()

# (df_accident.groupby('accident_severity')['number_of_casualties'].sum() / 6) / 365

# # We can see that during the **2005** to **2010** time period, the average death toll on UK roads was approx **12** per day, **one death every two hours**.

# df_accident['number_of_casualties'].describe()

# # We can see we have a large range of numbers, however the **min**, **25%**, **50%** and **75%** are all **1**, and the **max** is **68**.  
# #   
# # Lets plot the numbers

# sns.boxplot(data=df_accident, x='number_of_casualties')

# plt.show()

# # On further investigation we found a collision with 68 casualties involving a coach overturning trying to exit a motorway.  
# #   
# # [Link for Coach Accident](https://www.theguardian.com/uk/2007/jan/04/transport.world1)

# # ### Feature - **'number_of_vehicles'**

# # Lets take a look at **'number_of_vehicles'**.

# df_accident['number_of_vehicles'].describe()

# round(df_accident.groupby(
#   'number_of_vehicles'
# )['accident_severity'].value_counts(ascending=False, normalize=True), 3) * 100

# sns.boxplot(data=df_accident, x='number_of_vehicles')

# plt.show()

# round((df_accident.groupby('accident_severity')['number_of_vehicles'].sum() / 6) / 365, 1)

# # As we can see, on average over **850** vehicles per day are involved in an **RTC** during the 2005 to 2010 time period.

# df_accident['accident_severity'].value_counts()

# (df_accident['number_of_vehicles'].sum() / 6) / 365

# # ### Feature - **'speed_limit'**

# # Lets take a look at **'speed_limit'**.

# df_accident['speed_limit'].value_counts(ascending=False)

# thirty = len(df_accident[df_accident['speed_limit'] == 30]) / len(df_accident)
# print(f'The 30mph has an accident rate of', f'{thirty:.1%}')

# # As we can see there is a 63.8% rate of **RTC's** in the **30mph** speed limit areas.  
# #   
# # These are usual urban areas.

# df_accident_severity = df_accident.groupby(
#   'speed_limit')['accident_severity'].value_counts(ascending=False)
# df_accident_severity.plot(
#   kind='bar',
#   title='Number of Accidents by Speed Limit and Severity',
#   xlabel='Speed Limit', ylabel='Number of Accidents')
# plt.show()

# # ### Features - **'local_authority_district'**

# df_accident.columns

# df_accident['local_authority_district'].value_counts(ascending=False).head(10)

# # We can see that **Birmingham** has the highest rate of accidents.

# # ### Features - **'police_force'**

# df_accident['police_force'].value_counts(ascending=False).head(10)

# # We can see that **Metropolitan Police** has the highest count beating its next in line by over 200,000.

# # ### Features - **'latitude' and 'longitude'**

# df_accident['date'] = df_accident['date'].dt.strftime('%d/%m/%Y')

# def map_rtc(data, year, pforce):
#   cond = (data['year'] == year) & (data['police_force'] == pforce)

#   lat = data[cond]['latitude'].tolist()
#   lon = data[cond]['longitude'].tolist()
#   nam = data[cond]['police_force'].tolist()
#   sev = data[cond]['accident_severity'].tolist()
#   cas = data[cond]['number_of_casualties'].tolist()
#   veh = data[cond]['number_of_vehicles'].tolist()
#   dat = data[cond]['date'].tolist()

#   def color_producer(status):
#       if 'Slight' in status:
#           return 'green'
#       elif 'Serious' in status:
#           return 'blue'
#       else:
#           return 'orange'

#   html = '''<h4>Collision Information</h4>
#   <b>%s</b>
#   <b>Severity: </b> %s
#   <b>Casualties: </b> %s
#   <b>Vehicles: </b> %s
#   <b>Date: </b> %s
#   '''
#   map = flm.Map(location=[lat[1], lon[1]], zoom_start=12, scrollWheelZoom=False)

#   fg = flm.FeatureGroup(name='RTC-Map')

#   for lt, ln, nm, st, ca, ve, da in zip((lat), (lon), (nam), (sev), (cas), (veh), (dat)):
#       iframe = flm.IFrame(html = html % ((nm), (st), (ca), (ve), (da)), height = 165)
#       popup = flm.Popup(iframe, min_width=200, max_width=500)
#       fg.add_child(flm.CircleMarker(location = [lt, ln], popup = (popup), fill_color=color_producer(st), color='None', radius=10, fill_opacity = 0.7))
#       map.add_child(fg)

#   map.save('map.html')
#   return map

# map_rtc(df_accident, 2017, 'Metropolitan Police')

# plt.scatter(df_accident['longitude'], df_accident['latitude'], cmap='Reds')
# plt.colorbar()
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Heatmap')

# # ## Conclusions

# 