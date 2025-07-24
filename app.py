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

# Simple CSS for colored results
st.markdown("""
            <style>
            .fraud-result{
            background-color:#ffebee;
            border:2px solid #f44336;
            border-radius:5px;
            padding:10px;
            margin:10px 0;
            color:#d32f2f;
            font-weight:bold;
            text-align:center;
            }
            .safe-result{
            background-color:#e8f5e8;
            border:2px solid #4caf50;
            border-radius:5px;
            padding:10px;
            margin:10px 0;
            color:#2e7d32;
            font-weight:bold;
            text-align:center;
            }
           </style>
           """, unsafe_allow_html=True )

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
            df = pd.read_csv(uploaded_file)

            st.write(f"**File uploaded:** {uploaded_file.name}")
            st.write(f"**Number of records:** {len(df)}")
            st.write(f"**Number of features:** {len(df.columns)}")

            st.subheader("📊 Data Preview")
            st.dataframe(df.head())

            if st.button("🔍 Detect Fraud"):
                try:
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)

                    df_results = df.copy()
                    df_results['Prediction'] = predictions
                    df_results['Fraud_Probability'] = probabilities[:, 1]
                    df_results['Result'] = df_results['Prediction'].map({0: 'Safe', 1: 'Fraud'})

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

                    st.subheader("📋 Detailed Results")

                    def color_result(val):
                        if val == 'Fraud':
                            return 'background-color:#CD1C18'
                        elif val == 'Safe':
                            return 'background-color:#06402B'
                        return ''

                    styled_df = df_results.style.applymap(color_result, subset=['Result'])
                    st.dataframe(styled_df, use_container_width=True)

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

        col1, col2 = st.columns(2)

        with col1:
            step = st.number_input("Step", min_value=1, value=1)
            amount = st.number_input("Amount", min_value=0.0, value=100.0)
            oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
            newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
            oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)

        with col2:
            newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)
            isFlaggedFraud = st.selectbox("Is Flagged Fraud", [0, 1])
            transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

        if st.button("🔍 Check Transaction"):
            try:
                type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
                type_DEBIT = 1 if transaction_type == "DEBIT" else 0
                type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
                type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

                input_data = np.array([[step, amount, oldbalanceOrg, newbalanceOrig,
                                        oldbalanceDest, newbalanceDest, isFlaggedFraud,
                                        type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER]])

                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]

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

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fraud Probability", f"{probability[1]:.4f}")
                with col2:
                    st.metric("Safe Probability", f"{probability[0]:.4f}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
