import folium as flm
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

APP_TITLE = 'Predicting RTC severity using Machine Learning'
st.set_page_config(page_title='Home', layout='wide')


def display_accidents_count(
    df, year, severity_status, accident_severity, pforce, metric_title
):
    df = df[(df['Year'] == year)]
    if pforce:
        df = df[df['Police_Force'] == pforce]
        df.drop_duplicates(inplace=True)
        total = df[accident_severity].count()
        # st.metric(metric_title,'{:,}'.format(total))
    if severity_status:
        df = df[df['Accident_Severity'] == severity_status]
        df.drop_duplicates(inplace=True)
        total = df[accident_severity].count()
        st.metric(metric_title, '{:,}'.format(total))


def display_casualties_count(
    df, year, severity_status, no_of_casualties, pforce, metric_title
):
    df = df[(df['Year'] == year)]
    if pforce:
        df = df[df['Police_Force'] == pforce]
        df.drop_duplicates(inplace=True)
        total = df[no_of_casualties].sum()

    if severity_status:
        df = df[df['Accident_Severity'] == severity_status]
        df.drop_duplicates(inplace=True)
        total = df[no_of_casualties].sum()
        st.metric(metric_title, '{:,}'.format(total))


def display_year(df, year, metric_title):
    st.metric(metric_title, '{:}'.format(year))


def display_severity_status(df, severity_status, metric_title):
    st.metric(metric_title, '{:}'.format(severity_status))


def map_rtc(data, year, pforce, severity):
    cond = (
        data['Year'] == year
    ) & (
        data['Police_Force'] == pforce
    ) & (
        data['Accident_Severity'] == severity
    )

    lat = data[cond]['Latitude'].tolist()
    lon = data[cond]['Longitude'].tolist()
    nam = data[cond]['Police_Force'].tolist()
    sev = data[cond]['Accident_Severity'].tolist()
    cas = data[cond]['Number_of_Casualties'].tolist()
    veh = data[cond]['Number_of_Vehicles'].tolist()
    dat = data[cond]['Date'].tolist()
    tim = data[cond]['Time'].tolist()

    def color_producer(status):
        if 'Slight' in status:
            return 'green'
        elif 'Serious' in status:
            return 'blue'
        else:
            return 'orange'

    html = '''<h3>Collision Information</h3>
    <p><b>%s</b></p>
    <table style="width:250px">
    <tr>
    <td><b>Severity: </b></td>
    <td><b>%s</b></td>
    </tr>
    <tr>
    <td><b>Casualties: </b></td>
    <td><b>%s</b></td>
    </tr>
    <tr>
    <td><b>Vehicles: </b></td>
    <td><b>%s</b></td>
    </tr>
    <tr>
    <td><b>Date: </b></td>
    <td><b>%s</b></td>
    </tr>
    <tr>
    <td><b>Time: </b></td>
    <td><b>%s</b></td>
    </tr>
    </table>
    '''
    map = flm.Map(
        location=[lat[0], lon[0]], zoom_start=10, scrollWheelZoom=False
    )

    fg = flm.FeatureGroup(name='My V Map')

    for lt, ln, nm, sv, ca, ve, da, ti in zip(
        (lat), (lon), (nam), (sev), (cas), (veh), (dat), (tim)
    ):
        iframe = flm.IFrame(
            html=html % ((nm), (sv), (ca), (ve), (da), (ti)), height=190
        )
        popup = flm.Popup(iframe, min_width=275, max_width=500)
        fg.add_child(
            flm.CircleMarker(
                location=[lt, ln], popup=(popup),
                fill_color=color_producer(sv), color='None',
                radius=7, fill_opacity=0.7
            )
        )
        map.add_child(fg)

    st_map = st_folium(map, width=1600)
    return st_map


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

    # Create lists for dropdowns.
    year_list = list(df['Year'].unique())
    year_list.sort()
    year = st.sidebar.selectbox(
        'Year of Accident', year_list, len(year_list) - 1
    )

    pforce_list = list(df['Police_Force'].unique())
    pforce_list.sort()
    pforce = st.sidebar.selectbox(
        'Police Force', pforce_list, len(pforce_list) - 1
        )
    # Create radio buttons.
    severity_status = st.sidebar.radio(
        'Severity Status', ['Slight', 'Serious', 'Fatal']
        )

    # Set columns.
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_year(df, year, 'Year of Accidents')  # f'Year{year}')

    with col2:
        display_accidents_count(
            df, year, severity_status, 'Accident_Severity',
            pforce, 'Number of Accidents'
                )

    with col3:
        display_casualties_count(
            df, year, severity_status, 'Number_of_Casualties',
            pforce, 'Number of Casualties'
            )

    with col4:
        display_severity_status(
            df, severity_status, 'Severity Status'
            )
    with st.container():
        map_rtc(df, year, pforce, severity_status)


if __name__ == "__main__":
    main()
