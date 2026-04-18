# Imported Libraries
import pickle
import streamlit as st
import numpy as np


st.title("Flower Classification App")

with open("model.pkl", "rb") as f:
    lr_model = pickle.load(f)

sl = st.number_input("Insert a sepel length")
sw = st.number_input("Insert a sepel width")
pl = st.number_input("Insert a petal length")
pw = st.number_input("Insert a petal width")

if st.button("Predict"):
    pred = lr_model.predict(np.array([[sl, sw, pl, pw]]))
    st.write("The flower is :", pred[0])

# pip install streamlit numpy scikit-learn
# python -m streamlit run main.py
