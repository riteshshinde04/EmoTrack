import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate("emotrack-ad232-firebase-adminsdk-rhn2m-858c1179d9.json")

# Initialize Firebase app if not already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

def app():
    st.title("Welcome to :violet[EmoTrack] 😎")

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "useremail" not in st.session_state:
        st.session_state.useremail = ""

    def login(email, password):
        try:
            user = auth.get_user_by_email(email)
            if password:  
                st.session_state.logged_in = True
                st.session_state.username = user.uid
                st.session_state.useremail = user.email
                st.success("Login Successful!")
            else:
                st.warning("Invalid credentials.")
        except:
            st.error("Login Failed: Invalid email or user does not exist.")

    def signup(email, password, username):
        try:
            user = auth.create_user(email=email, password=password, uid=username)
            st.success("Account created successfully!")
            st.info("Please log in using your email and password.")
        except Exception as e:
            st.error(f"Sign-Up Failed: {e}")

    def logout():
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.useremail = ""
        st.info("Logged out successfully.")

    if not st.session_state.logged_in:
        
        choice = st.radio("Login/Sign Up", ["Login", "Sign Up"])

        if choice == "Login":
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                login(email, password)
        else:
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            username = st.text_input("Enter Your Unique Username")
            if st.button("Create Account"):
                signup(email, password, username)
    else:
        
        st.success(f"Logged in as: {st.session_state.username}")
        st.text(f"Email: {st.session_state.useremail}")
        if st.button("Log Out"):
            logout()