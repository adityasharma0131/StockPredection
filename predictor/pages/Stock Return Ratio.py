import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

def fetch_stock_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1mo")
    return data

# Streamlit app
def main():
    st.title('Stock Return Ratio Visualization')

    # Input for stock ticker symbol
    ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., TCS.NS)", "TCS.NS")

    # Input for starting date
    start_date = st.date_input("Select Start Date", datetime(2020, 1, 1))

    # Button to trigger the retrieval of charts and return ratio graph
    if st.button("Retrieve Charts and Return Ratio Graph"):
        # Fetching stock data
        end_date = datetime.now().date()  # Accessing today's date properly
        data = fetch_stock_data(ticker_symbol, start_date, end_date)

        # Displaying stock prices with candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        increasing=dict(line=dict(color="green")),
                        decreasing=dict(line=dict(color="red")),
                        name='Actual'))
        fig.update_layout(xaxis_title='Date',
                          yaxis_title='Stock Prices',
                          xaxis_rangeslider_visible=False,
                          template="plotly_dark")  # Set the dark mode template
        fig.update_layout(
                            xaxis=dict(
                                title='Date',
                                showgrid=True  # Disable grid lines for x-axis
                            ),
                            yaxis=dict(
                                title='Price',
                                showgrid=True  # Disable grid lines for y-axis
                            ))
        st.subheader("Stock Price - Monthly Candlestick Chart")
        st.plotly_chart(fig, use_container_width=True)

        # Calculating returns
        prices = data['Close']
        returns = prices.pct_change().dropna()

        st.subheader("Return Ratio of " + ticker_symbol)

        # Create a Plotly figure for the line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', line=dict(color='orange')))
        fig_line.update_layout(xaxis_title="Date", yaxis_title="Returns")
        fig_line.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
        st.plotly_chart(fig_line, use_container_width=True)
    
if __name__ == "__main__":
    main()
