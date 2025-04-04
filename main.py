import streamlit as st
from streamlit_option_menu import option_menu
import home, account

st.set_page_config(
    page_title="EmoTrack",
)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({"title": title, "function": function})

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title="EmoTrack",
                options=["Home", "Account"],
                icons=["house-fill", "person-circle"],
                menu_icon="chat-text-fill",
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": "black"},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                },
            )


        if app == "Home":
            if st.session_state.get("logged_in"):
                home.app()
            # else:
            #     st.warning("You need to log in to access the Home page.")
            #     st.write("Go to the Account page to log in.")
        # elif app == "Account":
        #     account.app()


app = MultiApp()
app.add_app("Home", home.app)
app.add_app("Account", account.app)

if __name__ == "__main__":
    app.run()