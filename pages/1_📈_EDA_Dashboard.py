# Import libraries
# Import libraries
import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page configuration
st.set_page_config(page_title="EDA", 
                   page_icon="📈",
                   layout="wide")


# Create the main dashboard title
st.markdown("# Exploratory Data Analysis Dashboard")

# Sidebar header
st.sidebar.header("Sao Paulo Crime Data Analysis")

# Read data
CRIME_DATA_FILE = "datasets/crime_data_by_year_without_crime_type.csv"
CURADA_WEATHER_CRIME_DATA_FILE = "datasets/SP_Monthly_weather_crime_data_2001_2021.csv"
COMM_WEATHER_CRIME_POPULATION_DATA_FILE = "datasets/SP_crime_weather_population_data.csv"
COMM_INTERPOLATED_POPULATION_DATA = "datasets/COMM_population_data.csv"

SP_crime_rate_by_year_data = pd.read_csv(CRIME_DATA_FILE, index_col=0)
SP_crime_weather_data = pd.read_csv(CURADA_WEATHER_CRIME_DATA_FILE, index_col=0)
SP_crime_weather_pop_data = pd.read_csv(COMM_WEATHER_CRIME_POPULATION_DATA_FILE, index_col=0)
SP_population_data = pd.read_csv(COMM_INTERPOLATED_POPULATION_DATA, index_col=0)
# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Overview", "Regional Level", "Municipal Level"])

# Define functions for each tab
def create_yearly_crime_rate_graph(df):
    """
    Create a line graph for the yearly trend of crime rates by region.

    Parameters:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    fig (Figure): The Plotly figure containing the line graph.
    """
    df = SP_crime_rate_by_year_data.groupby(['Year', 'Region'], as_index=False)['Crime Rate'].mean()

    fig = px.line(df, x='Year', y='Crime Rate', color='Region',
                title="Yearly Crime Rate by Region <br><sup>Note: Population Data in 2007 and 2010 are not available</sup>",
                labels={'x': "Year", 'y': 'Crime Rate'})

    # Customize the graph layout (optional)
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Crime Rate (%)',  # Update the y-axis label to include percentage
        showlegend=True
    )
    return fig

def create_donut_chart(df):
    """
    Create a donut chart for crime type distribution.

    Parameters:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    fig (Figure): The Plotly figure containing the donut chart.
    """
    grouped_ctype_df = df.groupby('Crime Type', as_index=False)['Monthly Crime Count'].sum().rename(columns={'Monthly Crime Count': 'Crime Count'})

    fig = go.Figure(data=[go.Pie(labels=grouped_ctype_df['Crime Type'], values=grouped_ctype_df['Crime Count'], hole=0.3, textinfo='label+percent', showlegend=False)])

    fig.update_traces(textposition='inside')
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        title_text="Crime Type Distribution",
    )

    return fig

def create_line_chart(df):
    """
    Create a line chart for monthly crime counts.

    Parameters:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    fig (Figure): The Plotly figure containing the line chart.
    """
    grouped_date_df = df.groupby('Date', as_index=False)['Monthly Crime Count'].sum()

    fig = go.Figure(data=[go.Scatter(x=grouped_date_df['Date'], y=grouped_date_df['Monthly Crime Count'], mode='lines')])

    fig.update_layout(
        title_text="Monthly Crime Counts",
        xaxis_title="Date",
        yaxis_title="Monthly Crime Count",
    )

    return fig

def show_yearly_crime_rate_by_region(crime_data, region):
    """
    Show yearly crime rate by region.

    Parameters:
    crime_data (DataFrame): The DataFrame containing the crime data.
    region (str): The selected region.

    Returns:
    fig (Figure): The Plotly figure containing the line graph.
    """
    # Filter the data for the specified region
    df = crime_data.groupby(['Year', 'Region'], as_index=False)['Crime Rate'].mean()
    filtered_df = df[df['Region'].isin([region])]

    # Create a line graph for the yearly trend of crime rates
    fig = px.line(
        filtered_df, x='Year', y='Crime Rate', color='Region',
        title=f"Yearly Crime Rate in {region}<br><sup>Note: Population Data in 2007 and 2010 are not available</sup>",
        labels={'x': "Year", 'y': 'Crime Rate'}
    )

    # Customize the graph layout (optional)
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Crime Rate (%)',  # Update the y-axis label to include percentage
        showlegend=False
    )

    return fig

def show_municipalities_crime_count(crime_data, region):
    """
    Show the top 5 municipalities with the highest crime count in a region.

    Parameters:
    crime_data (DataFrame): The DataFrame containing the crime data.
    region (str): The selected region.

    Returns:
    fig (Figure): The Plotly figure containing the bar graph.
    """
    # Filter the data for the specified region
    filtered_df = crime_data[crime_data['Region'].isin([region])]
    df = filtered_df.groupby('Municipality', as_index=False)['Crime Count'].sum()

    # Sort the data by Crime Count in descending order
    df = df.sort_values(by='Crime Count', ascending=False)

    # Select the top 5 municipalities
    top_municipalities = df.head(5)

    # Create a bar graph for the top 5 municipalities
    fig = px.bar(
        top_municipalities, x='Municipality', y='Crime Count',
        title=f"Top 5 Municipalities with Highest Crime Count in {region}",
        labels={'x': "Municipality", 'y': 'Crime Count'}
    )

    return fig

def show_municipality_crime_count_based_on(crime_data, region, crime_type):
    """
    Show the top 5 municipalities with the highest recorded crimes of a specific type in a region.

    Parameters:
    crime_data (DataFrame): The DataFrame containing the crime data.
    region (str): The selected region.
    crime_type (str): The selected crime type.

    Returns:
    fig (Figure): The Plotly figure containing the bar graph.
    """
    # Filter the data for the specified region and crime type
    filtered_df = crime_data[(crime_data['Region'] == region) & (crime_data['Crime Type'] == crime_type)]
    df = filtered_df.groupby('Municipality', as_index=False)['Monthly Crime Count'].sum().rename(columns={'Monthly Crime Count': 'Crime Count'})

    # Sort the data by Crime Count in descending order and select the top 5 municipalities
    top_municipalities = df.sort_values(by='Crime Count', ascending=False).head(5)

    # Create a bar graph for the top 5 municipalities
    fig = px.bar(
        top_municipalities, y='Municipality', x='Crime Count',
        title=f"Municipalities in {region} with high recorded {crime_type} crimes",
        labels={'y': "Municipality", 'x': 'Crime Count'},
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    return fig

def show_monthly_crime_trend_based_on(crime_data, region, crime_type):
    """
    Show the monthly crime trend for a specific crime type in a region.

    Parameters:
    crime_data (DataFrame): The DataFrame containing the crime data.
    region (str): The selected region.
    crime_type (str): The selected crime type.

    Returns:
    fig (Figure): The Plotly figure containing the line graph.
    """
    # Filter the data for the specified region and crime type
    filtered_df = crime_data[(crime_data['Region'] == region) & (crime_data['Crime Type'] == crime_type)]
    grouped_df = filtered_df.groupby(['Date'], as_index=False)['Monthly Crime Count'].sum()

    # Create a line graph for the monthly crime trend
    fig = px.line(
        grouped_df, x='Date', y='Monthly Crime Count',
        title=f"When does {crime_type} crimes frequently occur in {region}?",
        labels={'x': 'Month', 'y': 'Crime Count'},
    )

    return fig

def plot_crimes_percent_mun_reg_ov(df, municipality, year):
    """Plot Crimes as Percent of all crimes for Municipality compared to Region and Overall"""
    
    pct_crime_overall_grp = (df.groupby(['Year', 'Crime Type'])['Monthly Crime Count'].sum() / df.groupby(['Year'])['Monthly Crime Count'].sum() * 100).reset_index(name='overall_pct')
    pct_crime_reg_grp = (df.groupby(['Year', 'Region', 'Crime Type'])['Monthly Crime Count'].sum() / df.groupby(['Year', 'Region'])['Monthly Crime Count'].sum() * 100).reset_index(name='regional_pct')
    pct_crime_mun_grp = (df.groupby(['Year', 'Region', 'Municipality', 'Crime Type'])['Monthly Crime Count'].sum() / df.groupby(['Year', 'Region', 'Municipality'])['Monthly Crime Count'].sum() * 100).reset_index(name='mun_pct')

    pct_df = pct_crime_mun_grp.merge(pct_crime_reg_grp, on=['Year', 'Region', 'Crime Type'])
    pct_df = pct_df.merge(pct_crime_overall_grp, on=['Year', 'Crime Type'])

    cols = pct_df.select_dtypes(include='float').columns
    pct_df[cols] = pct_df[cols].round(2)
    pct_df_melt = pct_df.melt(id_vars=['Year', 'Region', 'Municipality', 'Crime Type'], var_name='Pct_of', value_name='Percentage')

    df = pct_df_melt[(pct_df_melt['Municipality']==municipality) & (pct_df_melt['Year']==year)].sort_values(by='Percentage')
    colors={'mun_pct':'red',
            'regional_pct':'blue',
            'overall_pct':'orange'}

    fig=go.Figure()
    for p in df['Pct_of'].unique():
        ds = df[df['Pct_of']==p]
        fig.add_traces(go.Bar( x=ds['Percentage'], y=ds['Crime Type'],
                            orientation='h',  marker_color=colors[p], name=p, text=ds['Percentage'], texttemplate='%{x:.1f}%', textposition='outside' ))
    fig.update_layout(title_text=f"<b>Crimes as Percent of all crimes for {municipality} compared to Region and Overall for {year}<b>", title_x=0.5, title_font=dict(size=20), showlegend=True, height=800, width=1000)

    return fig

def generate_region_stats(data, year, population_df):
    # Calculate crime per capita and per 1000 people
    per_capita_df = data.groupby(['Year', 'Region', 'Municipality']).agg(
        crime_count=('Monthly Crime Count', 'sum')).reset_index()
    per_capita_df = per_capita_df.merge(population_df, on=['Year', 'Municipality'], how='left')
    per_capita_df['CrimePerCapita'] = per_capita_df['crime_count'] / per_capita_df['Population']
    per_capita_df['CrimePer1000'] = per_capita_df['crime_count'] / per_capita_df['Population'] * 1e3
    per_capita_df.replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate region statistics for the given year
    region_ds = per_capita_df[per_capita_df['Year'] == year].groupby(['Region'])[['Population', 'crime_count']].sum().reset_index()
    region_ds['perCapita'] = region_ds['crime_count'] / region_ds['Population']
    region_ds['CrimePer1000'] = region_ds['crime_count'] / region_ds['Population'] * 1000

    # Create a subplot with three horizontal bar charts
    fig = make_subplots(1,3, shared_yaxes=True,
                        subplot_titles=('Population','Crime', 'per 1000 people'),
                        horizontal_spacing=0.01)

    fig.add_trace(go.Bar(x=region_ds['Population'], y=region_ds['Region'], orientation='h', name='Population',
                        text=region_ds['Population']/1e6,  texttemplate='%{text:.2f}M', textposition='outside' ),       row=1, col=1 )
    fig.add_trace(go.Bar(x=region_ds['crime_count'], y=region_ds['Region'], orientation='h', name='Crime Count',
                        text=region_ds['crime_count']/1e3,  texttemplate='%{text:.1f}K',textposition='outside' ),      row=1, col=2 )
    fig.add_trace(go.Bar(x=region_ds['CrimePer1000'], y=region_ds['Region'], orientation='h', name='per 1000',
                        text=region_ds['CrimePer1000'], texttemplate='%{x:.1f}', textposition='outside' ),   row=1, col=3 )
    fig.update_layout(title_text="<b>Region Stats: 2020<b>", title_x=0.5, title_font=dict(size=20), showlegend=False, height=400, width=1000)
    fig.update_yaxes(showgrid=True, row=1, col=1)
    fig.update_yaxes(showgrid=True, row=1, col=2)
    fig.update_yaxes(showgrid=True, row=1, col=3)

    # expand x ticks so the bar callout is visible for max value
    max_x_pop = region_ds['Population'].max() * 1.25
    max_x_crime = region_ds['crime_count'].max() * 1.25
    max_x_perCapita = region_ds['CrimePer1000'].max() * 1.25
    fig.update_xaxes(range=[0, max_x_pop], showticklabels=False, row=1, col=1)
    fig.update_xaxes(range=[0, max_x_crime], showticklabels=False, row=1, col=2)
    fig.update_xaxes(range=[0, max_x_perCapita], showticklabels=False, row=1, col=3)

    return fig

# Define the content for each tab
with tab1:
    st.header("Analysis Highlights")
    
    # Show Yearly Crime Rate by Region
    st.plotly_chart(create_yearly_crime_rate_graph(SP_crime_rate_by_year_data), use_container_width=True)

    # Show Donut Chart for Crime Type Distribution
    col1_, col2_ = st.columns(2)
    col1_.plotly_chart(create_donut_chart(SP_crime_weather_data), use_container_width=True)
    col2_.plotly_chart(create_line_chart(SP_crime_weather_data), use_container_width=True)

with tab2:
    st.header("Regional Analysis")
    year_opt = st.slider('Slide to pick a year:', min_value=2001, max_value=2020, value=2020)
                         
    st.plotly_chart(generate_region_stats(SP_crime_weather_pop_data, year_opt, SP_population_data), use_container_width=True)

    # Select a region for analysis
    regions = st.selectbox(
        'Select region/s to evaluate',
        list(SP_crime_rate_by_year_data['Region'].unique())
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(show_yearly_crime_rate_by_region(SP_crime_rate_by_year_data, regions), use_container_width=True)

    with col2:
        st.plotly_chart(show_municipalities_crime_count(SP_crime_rate_by_year_data, regions), use_container_width=True)

    # Select a crime type for further analysis
    crime_type_options = st.selectbox('Select crime type to analyze:', 
                                      list(SP_crime_weather_data['Crime Type'].unique()))

    # Show Municipalities with High Recorded Crime Counts based on Crime Type
    col_1, col_2 = st.columns(2)
    col_1.plotly_chart(show_municipality_crime_count_based_on(SP_crime_weather_data, regions, crime_type_options), use_container_width=True)
    col_2.plotly_chart(show_monthly_crime_trend_based_on(SP_crime_weather_data, regions, crime_type_options), use_container_width=True)

with tab3:
    st.header("Municipal Analysis")
    c1, c2 = st.columns(2)
    with c1:
        municipality_options = st.selectbox('Choose a municipality to explore:',
                                            list(SP_crime_rate_by_year_data['Municipality'].unique()))
    with c2:
        year_mun_options = st.selectbox('Pick a year to view:', 
                                        list(SP_crime_rate_by_year_data['Year'].unique()))

    st.plotly_chart(plot_crimes_percent_mun_reg_ov(SP_crime_weather_pop_data, municipality_options, year_mun_options))

