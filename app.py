import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)
# Load model
@st.cache_resource
def load_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model("xgb_fraud_model.json")
        return model
    except FileNotFoundError:
        st.error("Model file 'xgb_fraud_model.json' not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
# Main app
def main():
    st.title("🔒 Fraud Detection System")
    st.write("Upload a CSV file or enter data manually to detect fraud")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Choose input method
    option = st.radio("Choose input method:", ["Upload CSV File", "Manual Input"])
    
    if option == "Upload CSV File":
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Show basic info
            st.write(f"**File uploaded:** {uploaded_file.name}")
            st.write(f"**Number of records:** {len(df)}")
            st.write(f"**Number of features:** {len(df.columns)}")
            
            # Show first few rows
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Detect Fraud"):
                try:
                    # Make predictions
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    # Add results to dataframe
                    df_results = df.copy()
                    df_results['Prediction'] = predictions
                    df_results['Fraud_Probability'] = probabilities[:, 1]
                    df_results['Result'] = df_results['Prediction'].map({0: 'Safe', 1: 'Fraud'})
                    
                    # Show summary
                    fraud_count = sum(predictions)
                    safe_count = len(predictions) - fraud_count
                    
                    st.subheader("📈 Results Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(predictions))
                    with col2:
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        st.metric("Safe Transactions", safe_count)
                    
                    # Show detailed results
                    st.subheader("📋 Detailed Results")
                    
                    # Color code the results
                    def color_result(val):
                        if val == 'Fraud':
                            return 'background-color: #ffcdd2'
                        elif val == 'Safe':
                            return 'background-color: #c8e6c9'
                        return ''
                    
                    styled_df = df_results.style.applymap(color_result, subset=['Result'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
    else:  # Manual Input
        st.subheader("✏️ Manual Input")
        st.write("Enter transaction details below:")
        
        # Create input fields - modify these based on your model's features
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=100.0)
            time = st.number_input("Time", min_value=0.0, value=0.0)
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
        
        with col2:
            v14 = st.number_input("V14", value=0.0)
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v18 = st.number_input("V18", value=0.0)
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)
        
        if st.button("🔍 Check Transaction"):
            try:
                # Create input array
                input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, 
                                      v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Show result
                st.subheader("🎯 Prediction Result")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-result">
                        ⚠️ FRAUD DETECTED ⚠️<br>
                        Fraud Probability: {probability[1]:.4f} ({probability[1]*100:.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-result">
                        ✅ TRANSACTION SAFE ✅<br>
                        Safe Probability: {probability[0]:.4f} ({probability[0]*100:.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fraud Probability", f"{probability[1]:.4f}")
                with col2:
                    st.metric("Safe Probability", f"{probability[0]:.4f}")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()