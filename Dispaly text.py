import streamlit as st

st.text("hello")

st.markdown("hello")

st.caption("hello")

st.latex("hello")

st.write("hello")

st.write(["hi","hello","how are you?"])

st.title("hello")

st.header("hello")

st.subheader("hello")

code = """
import pandas as pd
import streamlit as st
st.title("hello")
"""
st.code(code)
