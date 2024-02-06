from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import yfinance as yf
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][-1]
        company_name = stock.info['longName']
        exchange = stock.info['exchange']
        currency = stock.info['currency']
        volume = stock.history(period='1d')['Volume'][-1]

        return {
            'Ticker': ticker,
            'Company Name': company_name,
            'Current Price': current_price,
            'Exchange': exchange,
            'Currency': currency,
            'Volume': volume,
            
        }
    except Exception as e:
        return "Currently No Data Found!!"

def calculate_candlestick_pattern(data):
    # Add your logic to identify candlestick patterns
    # For example, you can create columns like 'Bullish Engulfing', 'Bearish Engulfing', etc.
    # Based on the candlestick patterns observed in the data
    
    # For demonstration, let's just add random signals for illustration purposes
    np.random.seed(42)
    data['Bullish Engulfing'] = np.random.randint(0, 12, size=len(data))
    data['Bearish Engulfing'] = np.random.randint(0, 12, size=len(data))
    
    return data

###################################################################################

yf.pdr_override()

st.title('Stock Trend Prediction {Price Action}')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')

# Button to trigger data retrieval
if st.button('Retrieve Data'):
    # Get today's date
    today_date = datetime.now().date()
    # download dataframe
    data = pdr.get_data_yahoo(user_input, start="2020-01-01", end=today_date)

    # Fetch stock details
    stock_details = get_stock_details(user_input)

    if isinstance(stock_details, dict):
        st.subheader(f"Stock Details for {user_input}:")

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
    data_description = data.describe().T.style.set_table_styles([
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

    # Calculate Candlestick Patterns
    data = calculate_candlestick_pattern(data)

    # Visualizations
    st.subheader("Closing Price vs Time Chart with Candlestick Patterns")
    
    # Create a Plotly figure
    fig_candlestick = go.Figure()

    # Plot the candlestick chart
    fig_candlestick.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    ))
    fig_candlestick.update_layout(xaxis_rangeslider_visible=False)
    fig_candlestick.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))

    # Plot bullish engulfing pattern
    bullish_engulfing_indices = data.index[data['Bullish Engulfing'] == 1]
    fig_candlestick.add_trace(go.Scatter(
        x=bullish_engulfing_indices,
        y=data['Close'][data['Bullish Engulfing'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green'),
        name='Bullish Engulfing'
    ))

    # Plot bearish engulfing pattern
    bearish_engulfing_indices = data.index[data['Bearish Engulfing'] == 1]
    fig_candlestick.add_trace(go.Scatter(
        x=bearish_engulfing_indices,
        y=data['Close'][data['Bearish Engulfing'] == 1],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red'),
        name='Bearish Engulfing'
    ))

    # Update layout
    fig_candlestick.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        title='Candlestick Chart with Bullish and Bearish Engulfing Patterns',
        showlegend=True
    )

    # Display the Plotly chart
    st.plotly_chart(fig_candlestick)

    # Splitting data into training and testing
    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])

    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Load my model
    model = load_model('keras_model.h5')

    # Testing Part
    past_100_days = data_training.tail(100)
    final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_data)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader("Prediction vs Original")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Original Trend', line=dict(color='blue'))) 
    fig2.add_trace(go.Scatter(x=data.index[-len(y_predicted):], y=y_predicted[:, 0], mode='lines', name='Predicted Trend', line=dict(color='red')))

    fig2.update_layout(xaxis_title="Time", yaxis_title="Price")
    fig2.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
    st.plotly_chart(fig2)
