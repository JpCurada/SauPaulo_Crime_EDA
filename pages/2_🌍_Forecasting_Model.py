import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Forecasting", 
                   page_icon="🌍",
                   layout="wide")

st.markdown("# Sao Paulo Crime Forecasting Model")
st.sidebar.header("Forecasting Model")