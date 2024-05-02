import streamlit as st
from pymongo import MongoClient
import subprocess

# Connect to MongoDB
client = MongoClient('mongodb+srv://adityasharma0431:anant99@cluster0.7uugmow.mongodb.net/?retryWrites=true&w=majority')
db = client['TradingMonk']
collection = db['users']

def login(username, password):
    # Check if the username and password match
    user = collection.find_one({'username': username, 'password': password})
    if user:
        st.success("Login successful.")
        # Open the welcome.py file located in the predictor folder in a new Streamlit app
        subprocess.Popen(["streamlit", "run", "predictor/Dashboard.py"])
    else:
        st.error("Invalid username or password.")

def main():
    st.title("Trading Monk Login")
    
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", key="login_button"):
        login(username, password)

if __name__ == "__main__":
    main()
