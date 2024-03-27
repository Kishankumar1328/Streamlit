import streamlit as st

st.header("Click Me")

if st.button("Say Hello"):
    st.write("why hello there")
else:
    st.write("Goodbye")
