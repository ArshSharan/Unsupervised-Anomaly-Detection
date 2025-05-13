import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px
import time

# Page config with futuristic theme
st.set_page_config(
    page_title="NEURALINK ANOMALY DETECTOR",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🖥️"
)

# Custom CSS for futuristic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    
    :root {
        --primary: #00f0ff;
        --secondary: #ff00aa;
        --dark: #0a0a1a;
        --light: #e0e0ff;
        --accent: #7b2dff;
    }
    
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: var(--primary) !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }
    
    .stApp {
        background-color: var(--dark);
        color: var(--light);
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(123, 45, 255, 0.1) 0%, transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(0, 240, 255, 0.1) 0%, transparent 25%);
    }
    
    .stDataFrame {
        background-color: rgba(10, 10, 26, 0.8) !important;
    }
    
    .stSelectbox, .stFileUploader, .stButton>button {
        border: 1px solid var(--primary) !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: var(--light) !important;
    }
    
    .stButton>button:hover {
        border: 1px solid var(--secondary) !important;
        background-color: rgba(255, 0, 170, 0.1) !important;
    }
    
    .css-1aumxhk {
        background-color: rgba(10, 10, 26, 0.5);
        border-radius: 10px;
        padding: 20px;
        border-left: 3px solid var(--primary);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 240, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
    }
    
    .glow {
        text-shadow: 0 0 10px currentColor;
    }
</style>
""", unsafe_allow_html=True)

# App header with animated elements
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style='margin-bottom: 0;'>NEURALINK <span class='glow' style='color: var(--secondary);'>ANOMALY</span> DETECTOR</h1>
        <div style='height: 2px; background: linear-gradient(90deg, var(--primary), var(--secondary)); margin: 5px 0 20px 0;'></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align: right; font-family: Orbitron; color: var(--light);'>
            <div style='font-size: 0.8em; color: var(--primary);'>SYSTEM STATUS</div>
            <div style='color: var(--primary);'>ONLINE</div>
            <div style='font-size: 0.7em;'>v2.4.1</div>
        </div>
        """, unsafe_allow_html=True)

# Futuristic file uploader
with st.container():
    st.markdown("""
    <h3 style='margin-bottom: 10px;'>DATA INPUT CONSOLE</h3>
    <div style='height: 1px; background: linear-gradient(90deg, var(--primary), transparent); margin: 0 0 20px 0;'></div>
    """, unsafe_allow_html=True)
    
    with stylable_container(
        key="futuristic_uploader",
        css_styles="""
            {
                border: 1px solid var(--primary);
                border-radius: 5px;
                padding: 20px;
                background-color: rgba(0, 0, 0, 0.3);
            }
        """,
    ):
        uploaded_file = st.file_uploader("SELECT DATASET FOR ANALYSIS", type=["csv"], help="Upload CSV file containing numeric data")

# Main processing section
if uploaded_file:
    try:
        with st.spinner('DECRYPTING DATA STREAM...'):
            time.sleep(1)
            df = pd.read_csv(uploaded_file)
            
        # Data preview section
        with st.expander("📡 RAW DATA STREAM", expanded=True):
            st.dataframe(df.head().style.applymap(lambda x: "color: white"))
            
            # Data quality metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                with stylable_container(
                    key="metric1",
                    css_styles="""
                        {
                            color: white;
                            border: 1px solid var(--primary);
                            border-radius: 5px;
                            padding: 5px;
                            background-color: rgba(0, 0, 0, 0.3);
                        }
                    """,
                ):
                    st.metric("ROWS", df.shape[0])
            with col2:
                with stylable_container(
                    key="metric2",
                    css_styles="""
                        {
                            border: 1px solid var(--primary);
                            border-radius: 5px;
                            padding: 5px;
                            background-color: rgba(0, 0, 0, 0.3);
                        }
                    """,
                ):
                    st.metric("COLUMNS", df.shape[1])
            with col3:
                missing = df.isnull().sum().sum()
                with stylable_container(
                    key="metric3",
                    css_styles=f"""
                        {{
                            border: 1px solid {'var(--secondary)' if missing > 0 else 'var(--primary)'};
                            border-radius: 5px;
                            padding: 5px;
                            background-color: rgba(0, 0, 0, 0.3);
                        }}
                    """,
                ):
                    st.metric("MISSING VALUES", missing)
            
            if missing > 0:
                with st.spinner('IMPUTING MISSING VALUES...'):
                    time.sleep(1)
                    df.fillna(df.mean(numeric_only=True), inplace=True)
                    st.success("MISSING VALUES IMPUTED USING COLUMN MEANS")

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 0:
            st.error("⚠️ NO NUMERIC COLUMNS DETECTED. UPLOAD VALID DATASET.")
        else:
            st.success(f"NUMERIC FEATURES IDENTIFIED: {', '.join(numeric_cols)}")
            
            # Analysis section
            with st.container():
                st.markdown("""
                <h3 style='margin-bottom: 10px;'>NEURAL ANALYSIS ENGINE</h3>
                <div style='height: 1px; background: linear-gradient(90deg, var(--primary), transparent); margin: 0 0 20px 0;'></div>
                """, unsafe_allow_html=True)
                
                with stylable_container(
                    key="analysis_container",
                    css_styles="""
                        {
                            border: 1px solid var(--primary);
                            border-radius: 5px;
                            padding: 20px;
                            background-color: rgba(0, 0, 0, 0.3);
                        }
                    """,
                ):
                    with st.spinner('INITIATING ISOLATION FOREST ALGORITHM...'):
                        # Scaling
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[numeric_cols])
                        
                        # Isolation Forest
                        clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                        preds = clf.fit_predict(scaled_data)
                        
                        df['ANOMALY'] = preds
                        df['ANOMALY'] = df['ANOMALY'].map({1: 'NORMAL', -1: 'ANOMALY'})
                        
                        time.sleep(1)
                    
                    # Results
                    st.markdown("""
                    <h4 style='margin-bottom: 10px;'>ANOMALY DETECTION RESULTS</h4>
                    """, unsafe_allow_html=True)
                    
                    # Anomaly metrics
                    anomaly_counts = df['ANOMALY'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        with stylable_container(
                            key="anomaly_metric",
                            css_styles="""
                                {
                                    border: 1px solid var(--secondary);
                                    border-radius: 5px;
                                    padding: 15px;
                                    background-color: rgba(0, 0, 0, 0.3);
                                }
                            """,
                        ):
                            st.metric("ANOMALIES DETECTED", anomaly_counts.get('ANOMALY', 0), delta_color="off")
                    with col2:
                        with stylable_container(
                            key="normal_metric",
                            css_styles="""
                                {
                                    border: 1px solid var(--primary);
                                    border-radius: 5px;
                                    padding: 15px;
                                    background-color: rgba(0, 0, 0, 0.3);
                                }
                            """,
                        ):
                            st.metric("NORMAL POINTS", anomaly_counts.get('NORMAL', 0), delta_color="off")
                    
                    # Interactive visualization
                    st.markdown("""
                    <h4 style='margin-bottom: 10px;'>INTERACTIVE DATA VISUALIZATION</h4>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["FEATURE DISTRIBUTION", "3D SCATTER PLOT"])
                    
                    with tab1:
                        selected_col = st.selectbox("SELECT FEATURE FOR ANALYSIS", numeric_cols, key="hist_col")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.set_style("dark")
                        ax.set_facecolor('#0a0a1a')
                        fig.patch.set_facecolor('#0a0a1a')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        
                        sns.histplot(df, x=selected_col, hue='ANOMALY', 
                                    palette={"NORMAL": "#00f0ff", "ANOMALY": "#ff00aa"}, 
                                    kde=True, ax=ax, element="step")
                        ax.set_title(f'DISTRIBUTION OF {selected_col.upper()}', color='white')
                        st.pyplot(fig)
                    
                    with tab2:
                        if len(numeric_cols) >= 3:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                x_axis = st.selectbox("X-AXIS", numeric_cols, index=0, key="x_axis")
                            with col2:
                                y_axis = st.selectbox("Y-AXIS", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y_axis")
                            with col3:
                                z_axis = st.selectbox("Z-AXIS", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="z_axis")
                            
                            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                                              color='ANOMALY',
                                              color_discrete_map={"NORMAL": "#00f0ff", "ANOMALY": "#ff00aa"},
                                              hover_data=df.columns)
                            fig.update_layout(
                                scene=dict(
                                    xaxis=dict(title=x_axis, gridcolor='rgba(0, 240, 255, 0.1)'),
                                    yaxis=dict(title=y_axis, gridcolor='rgba(0, 240, 255, 0.1)'),
                                    zaxis=dict(title=z_axis, gridcolor='rgba(0, 240, 255, 0.1)'),
                                    bgcolor='#0a0a1a'
                                ),
                                paper_bgcolor='#0a0a1a',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("3D VISUALIZATION REQUIRES AT LEAST 3 NUMERIC FEATURES")
                    
                    # Download results
                    st.markdown("""
                    <h4 style='margin-bottom: 10px;'>DATA EXPORT</h4>
                    """, unsafe_allow_html=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    with stylable_container(
                        key="download_button",
                        css_styles="""
                            button {
                                border: 1px solid var(--primary) !important;
                                background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
                                color: black !important;
                                font-weight: bold !important;
                            }
                            button:hover {
                                border: 1px solid var(--secondary) !important;
                                background: linear-gradient(90deg, var(--secondary), var(--primary)) !important;
                            }
                        """,
                    ):
                        st.download_button("DOWNLOAD ANALYSIS RESULTS", csv, "anomaly_detection_results.csv", "text/csv")
    
    except Exception as e:
        st.error(f"SYSTEM ERROR: {str(e)}")
else:
    with stylable_container(
        key="waiting_container",
        css_styles="""
            {
                border: 1px dashed var(--primary);
                border-radius: 5px;
                padding: 50px;
                text-align: center;
                background-color: rgba(0, 0, 0, 0.1);
                margin-top: 50px;
            }
        """,
    ):
        st.markdown("""
        <div style='color: var(--primary); font-size: 1.2em;'>
            AWAITING DATA UPLOAD
        </div>
        <div style='color: var(--light); margin-top: 10px;'>
            UPLOAD CSV FILE TO INITIATE ANOMALY DETECTION SEQUENCE
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: var(--light); font-size: 0.8em;'>
    NEURALINK ANOMALY DETECTOR v2.4 | POWERED BY ISOLATION FOREST ALGORITHM
</div>
""", unsafe_allow_html=True)
