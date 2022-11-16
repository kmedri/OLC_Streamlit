import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

############################### FUNCTIONS ###################################
def get_data(url):
    df = pd.read_parquet(url)

    return df
##############################################################################
def years_to_accidents_graph(df):
        # analysing the count of accident sevearity date wise
        df_fatal = df[df['Accident_Severity'] == 'Fatal'][['Accident_Severity','Date']].groupby(by = 'Date').count()
        df_serious = df[df['Accident_Severity'] == 'Serious'][['Accident_Severity','Date']].groupby(by = 'Date').count()
        df_slight = df[df['Accident_Severity'] == 'Slight'][['Accident_Severity','Date']].groupby(by = 'Date').count()

        figure, axes = plt.subplots(3, figsize=( 100, 25))
        plt.grid('both')

        sns.lineplot(  y="Accident_Severity", x= df_fatal.index, data=df_fatal,  ax=axes[0], color = 'red')

        sns.lineplot(  y="Accident_Severity", x= df_serious.index, data=df_serious,  ax=axes[1], color = 'blue')

        sns.lineplot(  y="Accident_Severity", x= df_slight.index, data=df_slight,  ax=axes[2], color = 'black')

        axes[0].set_title('Fatal Accident with time',fontsize= 40)
        axes[1].set_title('Serious Accident with time',fontsize=40)
        axes[2].set_title('Slight Accident with time', fontsize= 40)
        
        return figure
##############################################################################

def years_to_accidents_graph3(df):
        years_options=df['Year'].unique().tolist()
        pf_options=df['Police_Force'].unique().tolist()

        year=st.selectbox('Select Year', years_options)
        pf=st.selectbox('Select Police Force', pf_options)

        df=df[df['Police_Force']==pf]
        df=df[df['Year']==year]
        total_casualties=df['Number_of_Casualties'].sum()

        fig=px.bar(df, x='Police_Force', y='Number_of_Casualties')

        fig.update_layout(width=800)

        st.metric('Casualties', '{:,}'.format(total_casualties))


        st.write(fig)
##############################################################################

def years_to_accidents_fatal(df):
        df_fatal=df[df['Accident_Severity']=='Fatal'][['Accident_Severity','Year']].groupby('Year').count()
        df_fatal.reset_index(inplace = True)
        fig = px.bar(df_fatal, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Fatal Accidents Count",
                 },
                 width=1100)

        fig2 = px.line(df_fatal, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Fatal Accidents Count",
                 },
                 width=1100)

        st.header("Fatal Accidents VS Years")
        st.write(fig,fig2)

##############################################################################

def years_to_accidents_serious(df):
        df_serious=df[df['Accident_Severity']=='Serious'][['Accident_Severity','Year']].groupby('Year').count()
        df_serious.reset_index(inplace = True)
        fig = px.bar(df_serious, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Serious Accidents Count",
                 },
                 width=1100)

        fig2 = px.line(df_serious, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Serious Accidents Count",
                 },
                 width=1100)

        st.header("Serious Accidents VS Years")
        st.write(fig,fig2)

##############################################################################

def years_to_accidents_slight(df):
        df_slight=df[df['Accident_Severity']=='Slight'][['Accident_Severity','Year']].groupby('Year').count()
        df_slight.reset_index(inplace = True)
        fig = px.bar(df_slight, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Slight Accidents Count",
                 },
                 width=1100)

        fig2 = px.line(df_slight, x='Year', y='Accident_Severity', log_x=True,
             labels={
                     "Year": "Year",
                     "Accident_Severity": "Slight Accidents Count",
                 },
                 width=1100)

        st.header("Slight Accidents VS Years")
        st.write(fig,fig2)

##############################################################################
def main():
        url = 'data/full_accident_data_time_series.parquet'
        df = get_data(url)
        #st.pyplot(years_to_accidents_graph(df))
        #years_to_accidents_graph2(df)
        years_to_accidents_slight(df)
        years_to_accidents_serious(df)
        years_to_accidents_fatal(df)
        


##############################################################################
                                  #####
##############################################################################

if __name__ == "__main__":
    st.set_page_config(page_title='Visualizations', layout='wide')
    main()