import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

#page setup

st.set_page_config(
  page_title="Fraud Detection Sustem",
  page_icon="🔒",
  layout="wide"
)
# Load model
@st.cache_resource
def load_model():
  try:
    model=xgb.XGBClassifier()
    model.load_model("xgb_fraud_model.json")
    return model
  except FileNotFoundError:
    st.error("Model File 'xgp_fraud_model.json' not found!")
    return None
  except Exception as e:
    st.error(f"Error loading model:{str(e)}")
    return None