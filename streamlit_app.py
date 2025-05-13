import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Anomaly Detector", layout="wide")
st.title("🔍 General-Purpose Anomaly Detection (Unsupervised ML)")

st.markdown("""
Upload your CSV file and let the model identify potential anomalies using the Isolation Forest algorithm.
This tool works for any kind of numeric dataset — financial, sensor logs, refinery systems, and more.
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        if df.isnull().sum().sum() > 0:
            st.warning("Missing values detected! Filling with column means...")
            df.fillna(df.mean(numeric_only=True), inplace=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns found. Please upload a valid dataset.")
        else:
            st.success("Numeric columns detected: " + ", ".join(numeric_cols))

            # Scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])

            # Isolation Forest
            clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            preds = clf.fit_predict(scaled_data)

            df['Anomaly'] = preds
            df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

            st.subheader("✅ Results with Anomaly Column")
            st.dataframe(df.head(20))

            st.subheader("📊 Anomaly Count")
            st.write(df['Anomaly'].value_counts())

            st.subheader("📉 Feature Distribution")
            selected_col = st.selectbox("Select column to plot", numeric_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df, x=selected_col, hue='Anomaly', kde=True, ax=ax)
            st.pyplot(fig)

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result CSV", csv, "anomaly_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting CSV upload...")
