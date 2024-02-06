# app.py
from datetime import datetime
import streamlit as st
from Neuraltesting import plot_stock_price_with_predictions

# Streamlit app header
st.title("Stock Price Analysis with Streamlit")

# Section for user input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., SBIN.NS):")
start_date = st.date_input("Enter Start Date:", datetime(2020, 1, 1))

# Button to trigger analysis
if st.button("Run Stock Analysis"):
    # Execute the imported function
    plot_stock_price_with_predictions(stock_symbol, start_date)

# Additional sections or widgets can be added as needed
