import streamlit as st
import requests
import os

st.set_page_config(page_title="Diagnosys: Content Intelligence System", layout="wide")
st.title("📊 Diagnosys: Content Intelligence System")

API_URL = os.environ.get("API_URL", "http://localhost:8000/process_request")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_area("Please enter your research request:", "", height=100)

if st.button("Send"):
    if user_input.strip():
        # Clear old messages
        st.session_state.messages = []
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call the backend
        with st.spinner("Processing request..."):
            try:
                response = requests.post(API_URL, json={"request": user_input})
                data = response.json()

                if data.get("success"):
                    agent_output = data.get("final_output", "No output returned")
                else:
                    agent_output = f"❌ Error: {data.get('error', 'Unknown error')}"

                # Add agent response
                st.session_state.messages.append({"role": "agent", "content": agent_output})

            except Exception as e:
                st.session_state.messages.append({"role": "agent", "content": f"Error: {str(e)}"})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")
