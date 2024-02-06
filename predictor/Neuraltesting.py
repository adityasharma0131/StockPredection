import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet

def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][-1]    #can use 1mo to make it work out during during holidays
        company_name = stock.info['longName']
        exchange = stock.info['exchange']
        currency = stock.info['currency']
        volume = stock.history(period='1d')['Volume'][-1]           #can use 1mo to make it work out during during holidays
       

        return {
            'Ticker': ticker,
            'Company Name': company_name,
            'Current Price': current_price,
            'Exchange': exchange,
            'Currency': currency,
            'Volume': volume,
        }
    except Exception as e:
        return e



def plot_stock_price_with_predictions(stock_symbol, start_date, end_date=None, periods=10, width=1000, height=600):
    if end_date is None:
        end_date = datetime.now()

    # Download stock data using yfinance
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval='1mo')
    
    stock_details = get_stock_details(stock_symbol)

    # Display stock details in a tabular format side by side
    if isinstance(stock_details, dict):
        st.subheader(f"Stock Details for {stock_symbol}:")

        # Create columns for displaying data side by side
        col1, col2 = st.columns(2)

        # Display details in the first column
        with col1:
            st.markdown(f'<p style="color:orange;">ATTRIBUTE</p>', unsafe_allow_html=True)
            for key in stock_details.keys():
                st.write(key)

        # Display values in the second column
        with col2:
            st.markdown(f'<p style="color:orange;">VALUE</p>', unsafe_allow_html=True)
            for value in stock_details.values():
                st.write(value)
    else:
        st.write(stock_details)

    # Describing Data
    st.subheader('Data From 2020 - 2024')

    # Add more spacing to the descriptive statistics table
    data_description = stock_data.describe().T.style.set_table_styles([
        {'selector': 'tr:hover',
         'props': [('background-color', '#ffffb3')]},
        {'selector': 'th',
         'props': [('background-color', '#e6e6e6')]},
        {'selector': 'td',
         'props': [('border', '2px solid #cccccc')]},
        {'selector': 'th:hover',
         'props': [('background-color', '#ffffb3')]},
        {'selector': 'tr:nth-child(even)',
         'props': [('background-color', '#f2f2f2')]},
    ])

    st.write(data_description)

    # Extract relevant columns for NeuralProphet
    stocks = stock_data[['Close']]
    stocks.reset_index(inplace=True)
    stocks.columns = ['ds', 'y']

    # NeuralProphet time series forecasting
    model = NeuralProphet()
    model.fit(stocks)
    future = model.make_future_dataframe(stocks, periods=periods)
    forecast = model.predict(future)
    actual_prediction = model.predict(stocks)

    # Plotly candlestick chart with NeuralProphet predictions
    fig = go.Figure()

    # Add Candlestick chart
    fig.add_trace(go.Candlestick(x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    increasing=dict(line=dict(color="green")),
                    decreasing=dict(line=dict(color="red")),
                    name='Actual'))

    # Add NeuralProphet predictions
    fig.add_trace(go.Scatter(x=actual_prediction['ds'], y=actual_prediction['yhat1'], mode='lines', name='Prediction (Actual)', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Future Predictions', line=dict(color='orange')))

    # Update chart layout for dark mode and adjust width and height
    fig.update_layout(title=f"{stock_symbol} Stock Price - Monthly Candlestick Chart with NeuralProphet Predictions",
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark",  # Set the dark mode template
                      showlegend=False)  # Hide the legend
    fig.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# Example usage:
# plot_stock_price_with_predictions("SBIN.NS", "2020-01-01", width=1200, height=800)
