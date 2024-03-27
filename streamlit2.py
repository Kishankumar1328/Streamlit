import streamlit as st
col1,col2,col3=st.columns(3)

col1.metric("Temperature","70°F","1.2°F")
col2.metric("wind","9 mph","-13%")
col3.metric("Humidity","86%","4%")
