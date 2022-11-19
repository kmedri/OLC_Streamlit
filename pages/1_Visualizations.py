import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
############################### FUNCTIONS ####################################
##############################################################################


def get_data(url):
    df = pd.read_parquet(url)

    return df

##############################################################################


def accidents_to_years_Bar_subplots(df):
    df_slight = df[df['Accident_Severity'] == 'Slight'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_slight.reset_index(inplace=True)

    df_serious = df[df['Accident_Severity']=='Serious'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_serious.reset_index(inplace=True)

    df_fatal = df[df['Accident_Severity'] == 'Fatal'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_fatal.reset_index(inplace=True)

    fig = make_subplots(rows=3, cols=1, x_title='Years', y_title='Accident Counts')

    fig.append_trace(go.Bar(
                x=df_slight['Year'], y=df_slight['Accident_Severity'], name='Slight Accidents'
        ), row=1, col=1)
    fig.append_trace(go.Bar(
                x=df_serious['Year'], y=df_serious['Accident_Severity'], name='Serious Accidents'
        ), row=2, col=1)
    fig.append_trace(go.Bar(
                x=df_fatal['Year'], y=df_fatal['Accident_Severity'], name='Fatal Accidents'
        ), row=3, col=1)

    fig.update_layout(width=1100, height=800)


# fig.update_xaxes(title_text="<b>Years</b>")
# fig.update_yaxes(title_text="<b>Accidents Count</b> ", secondary_y=False)
# fig.update_yaxes(title_text="<b>secondary</b> Y - axis ", secondary_y=True)
    st.header('Accidents vs Years (Bar Graphs)')
    st.write(fig)
##############################################################################


def accidents_to_years_Line_subplots(df):
    df_slight=df[df['Accident_Severity']=='Slight'][['Accident_Severity','Year']].groupby('Year').count()
    df_slight.reset_index(inplace=True)

    df_serious=df[df['Accident_Severity']=='Serious'][['Accident_Severity','Year']].groupby('Year').count()
    df_serious.reset_index(inplace=True)

    df_fatal=df[df['Accident_Severity']=='Fatal'][['Accident_Severity','Year']].groupby('Year').count()
    df_fatal.reset_index(inplace=True)

    fig=make_subplots(rows=3, cols=1, x_title='Years', y_title='Accidents Count')

    fig.add_trace(go.Scatter(
            x=df_slight['Year'], y=df_slight['Accident_Severity'], name='Slight Accidents'
            ), row=1, col=1)
    fig.add_trace(go.Scatter(
            x=df_serious['Year'], y=df_serious['Accident_Severity'], name='Serious Accidents'
            ), row=2, col=1)
    fig.add_trace(go.Scatter(
            x=df_fatal['Year'], y=df_fatal['Accident_Severity'], name='Fatal Accidents'
            ), row=3, col=1)

    fig.update_layout(width=1100, height=800)#, title='Accidents vs Years')

    # fig.update_xaxes(title_text="<b>Years</b>")
    # fig.update_yaxes(title_text="<b>Accidents Count</b> ", secondary_y=False)
    # fig.update_yaxes(title_text="<b>secondary</b> Y - axis ", secondary_y=True)

    st.header('Accidents vs Years (Line Graphs)')
    st.write(fig)
##############################################################################


def accidents_to_years_Line_overlap(df):
    df_slight = df[df['Accident_Severity']=='Slight'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_slight.reset_index(inplace=True)

    df_serious = df[df['Accident_Severity']=='Serious'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_serious.reset_index(inplace=True)

    df_fatal = df[df['Accident_Severity']=='Fatal'][['Accident_Severity', 'Year']].groupby('Year').count()
    df_fatal.reset_index(inplace=True)

    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(
            x=df_slight['Year'], y=df_slight['Accident_Severity'], name='Slight Accidents'
            ))  # , row=1, col=1)
    fig.add_trace(go.Scatter(
            x=df_serious['Year'], y=df_serious['Accident_Severity'], name='Serious Accidents'
            ))  # , row=2, col=1)
    fig.add_trace(go.Scatter(
            x=df_fatal['Year'], y=df_fatal['Accident_Severity'], name='Fatal Accidents'
            ))  # , row=3, col=1)

    fig.update_layout(width=1100, height=800)#, title='Accidents vs Years')

    fig.update_xaxes(title_text="<b>Years</b>")
    fig.update_yaxes(title_text="<b>Accidents Count</b> ", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> Y - axis ", secondary_y=True)

    st.header('Accidents vs Years (Line Graphs Overlap)')
    st.write(fig)
##############################################################################


def main():
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
            </style>
            """, unsafe_allow_html=True
    )
    url = 'data/full_accident_data_time_series.parquet'
    df = get_data(url)
    col1, col2 = st.columns((1, 5))
    with col1:
        st.image("https://raw.githubusercontent.com/kmedri/OLC_Streamlit/style/assets/Omdena-Logo.png?raw=true")
    with col2:
        st.write('# Liverpool Chapter')
    accidents_to_years_Bar_subplots(df)
    accidents_to_years_Line_subplots(df)
    accidents_to_years_Line_overlap(df)


##############################################################################
###################################Main#######################################
##############################################################################

if __name__ == "__main__":
    st.set_page_config(page_title='Visualizations', layout='wide')
    main()