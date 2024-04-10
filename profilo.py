import streamlit as st
from streamlit_option_menu import option_menu
import random

st.set_page_config(
    page_title='profilo',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar menu options
options = ["Home", "About", "Certificate", "Project", "Contact"]

# Define content for each option
content = {
    "Home": "<h1 style='font-size: 3em;'>HI, I am Kishan Kumar. DATA SCIENCE ENTHUSIASTIC</h1>",
    "About": "This is the about page.",
    "Certificate": {
        "Python for Data Science": (
        "pyd.jpg", "https://www.credly.com/badges/f7efca15-02ec-46be-a0ac-84c62fea02b3/linked_in_profile"),
        "Data Visualization": (
        "dv.jpg", "https://www.credly.com/badges/a33539eb-e491-449c-8a12-6f1f925248ba/linked_in_profile"),
        "Data Analysis": (
        "da.jpg", "https://www.credly.com/badges/ef16ffb5-db3c-4ded-b41b-fcaa35b2d2da/linked_in_profile"),
        "Prompt Engineering": (
        "pe2.jpg", "https://courses.cognitiveclass.ai/certificates/e605bffd4da945149049fe4a2955efd4"),
        "Machine Learning": (
        "ml.jpg", "https://courses.cognitiveclass.ai/certificates/a3fde26a28a04c60adab05199e3294d4"),
        "Applied Data Science": (
        "ds.jpg", "https://www.credly.com/badges/7b399bf8-3691-4ede-b4a9-77cfa15fb325/linked_in_profile"),
    },
    "Project": "This is the project page.",
    "Contact": "This is the contact page."
}

# Icons for the sidebar menu
icons = ["house-door", "book", "file-earmark-text", "archive", "envelope"]

with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard",
        options=options,
        icons=icons,
        menu_icon="cast",
        default_index=0)

# Display content based on the selected option
if selected == "Certificate":
    # Get selected certificates
    certificates = list(content[selected].items())

    # Calculate the number of rows required
    num_rows = len(certificates) // 3 + (1 if len(certificates) % 3 != 0 else 0)

    # Display certificates in a 2x3 grid
    for i in range(num_rows):
        col1, col2, col3 = st.columns(3)
        for col, cert in zip([col1, col2, col3], certificates[i * 3:i * 3 + 3]):
            if cert is not None:
                cert_name, (cert_image, cert_link) = cert
                with col:
                    st.image(cert_image, caption=cert_name)
                    st.write(f"Description: {cert_name}")
                    st.write(f"Link: <a style='color:red' href='{cert_link}' target='_blank'>{cert_link}</a>", unsafe_allow_html=True)
else:
    st.write(content[selected], unsafe_allow_html=True)
