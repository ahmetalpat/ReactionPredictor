import streamlit as st
import sqlite3
from passlib.hash import bcrypt

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY, password_hash TEXT)
    ''')
    conn.commit()
    conn.close()

def show_auth_page():
    st.title("Welcome to the Reaction Predictor")
    st.write("Please log in or register to continue to the demo.")
    
    init_db()

    choice = st.radio("Choose an action:", ["Login", "Register"], horizontal=True)

    if choice == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                conn.close()
                if result and bcrypt.verify(password, result[0]):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    elif choice == "Register":
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if new_username and new_password:
                    hashed_password = bcrypt.hash(new_password)
                    try:
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (new_username, hashed_password))
                        conn.commit()
                        st.success("Registration successful! Please go to the Login tab.")
                    except sqlite3.IntegrityError:
                        st.error("Username already exists. Please choose another one.")
                    finally:
                        conn.close()
                else:
                    st.warning("Please enter both a username and password.")