import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

############################### FUNCTIONS ###################################
def get_data(url):
    df = pd.read_parquet(url)

    return df

##############################################################################

def accidents_to_years(df):
        df_slight=df[df['Accident_Severity']=='Slight'][['Accident_Severity','Year']].groupby('Year').count()
        df_slight.reset_index(inplace = True)

        df_serious=df[df['Accident_Severity']=='Serious'][['Accident_Severity','Year']].groupby('Year').count()
        df_serious.reset_index(inplace = True)

        df_fatal=df[df['Accident_Severity']=='Fatal'][['Accident_Severity','Year']].groupby('Year').count()
        df_fatal.reset_index(inplace = True)

        fig=make_subplots(rows=3, cols=1)

        fig.append_trace(go.Bar( 
                x=df_slight['Year'], y=df_slight['Accident_Severity']
                ), row=1, col=1)
        fig.append_trace(go.Bar( 
                x=df_serious['Year'], y=df_serious['Accident_Severity']
                ), row=2, col=1)
        fig.append_trace(go.Bar( 
                x=df_fatal['Year'], y=df_fatal['Accident_Severity']
                ), row=3, col=1)
        
        fig.update_layout(width=1100, height=800)

        st.write(fig)
##############################################################################
def main():
        url = 'data/full_accident_data_time_series.parquet'
        df = get_data(url)
        accidents_to_years(df)
        


##############################################################################
                                  #####
##############################################################################

if __name__ == "__main__":
    st.set_page_config(page_title='Visualizations', layout='wide')
    main()