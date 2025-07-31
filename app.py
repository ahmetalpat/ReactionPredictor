import streamlit as st
import subprocess
import os
import time

# Check an environment variable to decide which app to run
APP_MODE = os.environ.get("STREAMLIT_APP_MODE")

if APP_MODE == "landing":
    from landing_page import show_landing_page
    show_landing_page()

elif APP_MODE == "predictor":
    # This must be the first Streamlit command
    st.set_page_config(page_title="Reaction Predictor", layout="centered")
    
    from auth_app import show_auth_page
    from predictor_app import show_predictor_page

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        show_predictor_page()
    else:
        show_auth_page()

else:
    # This is the "launcher" part that runs when the container starts
    print("Launcher script started. Starting Streamlit services...")
    
    env_vars = os.environ.copy()

    # Start the landing page app on port 8502
    env_vars["STREAMLIT_APP_MODE"] = "landing"
    subprocess.Popen([
        "streamlit", "run", "app.py",
        "--server.port=8502",
        "--server.headless=true", # Important for running in a container
    ], env=env_vars)

    # Start the main predictor app on port 8501
    env_vars["STREAMLIT_APP_MODE"] = "predictor"
    subprocess.Popen([
        "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.headless=true", # Important for running in a container
        # This allows Nginx to correctly proxy to the /app path
        "--server.baseUrlPath=app",
    ], env=env_vars)

    # Keep the main script alive so the container doesn't exit
    while True:
        time.sleep(3600)