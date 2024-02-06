from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

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



###################################################################################

yf.pdr_override() # <== that's all it takes :-)

st.title('Stock Trend Prediction (Volume)')

user_input = st.text_input('Enter Stock Ticker', 'SBIN.NS')

# Button to trigger data retrieval
if st.button('Retrieve Data'):
    # Get today's date
    today_date = datetime.now().date()
    # download dataframe
    data = pdr.get_data_yahoo(user_input, start="2020-01-01", end=today_date)

    # Fetch stock details
    stock_details = get_stock_details(user_input)

    # Display stock details in a tabular format side by side
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

    # Calculate Volume Change Percentage
    data['VolumeChangePercentage'] = data['Volume'].pct_change() * 100
    data['VolumeChangePercentage'] = data['VolumeChangePercentage'].fillna(0)

    # Visualizations
    st.subheader("Closing Price vs Time Chart")
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    increasing_line_color="green",
                    decreasing_line_color="red")])
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
    st.plotly_chart(fig)

    st.subheader("Volume Change Percentage")
    fig_volume = go.Figure()

# Increase the width of bars for better visibility
    bar_width = 3

    # Add the closing price line
    fig_volume.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', showlegend=False))

    # Add the volume bars
    fig_volume.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=np.where(data['VolumeChangePercentage'] > 0, 'green', 'red'),
        opacity=1,
        width=bar_width,
        showlegend=False,
        name='Volume'
    ))

    # Set labels
    fig_volume.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
    )
    fig_volume.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))

    # Display the Plotly chart
    st.plotly_chart(fig_volume)



    st.subheader("Stock and Volume Chart")
    
    # Add the stock price line on the primary y-axis
    fig_combined = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    showlegend=False,
                    increasing_line_color="green",
                    decreasing_line_color="red")])

    # Add the volume bars on the secondary y-axis
    fig_combined.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=np.where(data['VolumeChangePercentage'] > 0, 'green', 'red'),
        opacity=0.5,
        name='Volume',
        showlegend=False,
        yaxis='y2'
    ))

    # Set layout
    fig_combined.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price', color='blue', side='left', showgrid=False),
        yaxis2=dict(title='Volume', color='black', side='right', overlaying='y', showgrid=False)
    )
    fig_combined.update_layout(xaxis_rangeslider_visible=False)
    fig_combined.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
    # Display the combined      chart
    st.plotly_chart(fig_combined)
    
    
    
    
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
    st.subheader("Prediction vs Original with Volume")
    
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
