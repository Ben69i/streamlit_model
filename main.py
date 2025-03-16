# import psycopg
from dotenv import load_dotenv
import streamlit_authenticator as stauth
import streamlit as st
import os
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page

# Load environment variables from .env
# load_dotenv()
#
# # Fetch variables
# USER = os.getenv("db_user")
# PASSWORD = os.getenv("db_password")
# HOST = os.getenv("db_host")
# PORT = os.getenv("db_port")
# DBNAME = os.getenv("db_dbname")
#
# st.cache_resource(show_spinner="Connecting to database...",ttl=600)
# def connect_db():
#     # Connect to the database
#
#     try:
#         connection = psycopg.connect(
#             user=USER,
#             password=PASSWORD,
#             host=HOST,
#             port=PORT,
#             dbname=DBNAME
#         )
#         print("Connection successful!")
#
#         return connection
#
#     except Exception as e:
#         pass


# c=connect_db()
# print(c.cursor().execute("SELECT * FROM streamlit_auth").fetchone())
# print(stauth.Hasher().hash('password').encode())
# c.close()
keys=st.session_state.keys()
if all(i not in keys for i in ["n_samples","data_preprocessing","n_features","centers_std","seed","lock_seed"]):
    st.session_state.n_samples=200
    st.session_state.n_features=5
    st.session_state.centers=2
    st.session_state.centers_std=1
    st.session_state.seed=42
    st.session_state.lock_seed=True
    st.session_state.data_preprocessing=None
col1,col2=st.columns(2)
col2.image("./logo.jpeg",width=150)
if col2.button("Click MeeeeeeeEEEE",on_click=lambda:rain(emoji="🍗🍔",font_size=80,animation_length=1,falling_speed=6),type="primary"):
    col2.text("You clicked me")
if st.button("Go to experimental",type="secondary"):
    switch_page("Experimental")
if st.button("Go to regression",type="secondary"):
    switch_page("Regress")

st.sidebar.button("clear cache",on_click=lambda:st.cache_data.clear())

