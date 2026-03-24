"""
Streamlit Dashboard for Churn Prediction System.
Displays customer analytics, prediction tool, and model insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.85; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 32px; }
    .risk-high { color: #ff4444; font-weight: bold; font-size: 24px; }
    .risk-medium { color: #ffaa00; font-weight: bold; font-size: 24px; }
    .risk-low { color: #00cc66; font-weight: bold; font-size: 24px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")
IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_importance.csv")


# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data
def load_feature_importance():
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    return None


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown("---")
st.sidebar.markdown("**Project 2** — ML System")
st.sidebar.markdown("Predicts customer churn probability using trained models.")
st.sidebar.markdown("---")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.sidebar.success(f"🟢 API Online — {health.get('model_name', 'N/A')}")
    else:
        st.sidebar.warning("🟡 API Online — Model not loaded")
except Exception:
    st.sidebar.error("🔴 API Offline")

st.sidebar.markdown("---")
st.sidebar.markdown("**Stack:** Scikit-learn, LightGBM, FastAPI, Streamlit")

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Analyze customer behavior, predict churn risk, and explore model insights.")

# Load data
try:
    df = load_dataset()
except FileNotFoundError:
    st.error("Dataset not found. Run the data generation script first.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Predict", "🧠 Insights"])


# ──────────────────────────────────────────────
# TAB 1: Overview
# ──────────────────────────────────────────────
with tab1:
    st.subheader("Customer Overview")

    # Metrics row
    total = len(df)
    churned = df["Churn_Binary"].sum()
    active = total - churned
    churn_rate = churned / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Customers</h3>
            <h1>{total:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>Churned</h3>
            <h1>{churned:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>Active</h3>
            <h1>{active:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <h3>Churn Rate</h3>
            <h1>{churn_rate:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        # Churn distribution
        churn_counts = df["Churn"].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            color_discrete_sequence=["#4facfe", "#f5576c"],
            hole=0.4,
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Churn by contract type
        contract_churn = df.groupby("Contract")["Churn_Binary"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Churn Rate"] = contract_churn["Churn Rate"] * 100
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            title="Churn Rate by Contract Type",
            color="Contract",
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
            text_auto=".1f",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # Monthly charges distribution
        fig = px.histogram(
            df, x="MonthlyCharges", color="Churn",
            title="Monthly Charges Distribution",
            nbins=40,
            color_discrete_map={"No": "#4facfe", "Yes": "#f5576c"},
            barmode="overlay",
            opacity=0.7,
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        # Tenure distribution
        fig = px.histogram(
            df, x="tenure", color="Churn",
            title="Tenure Distribution (months)",
            nbins=36,
            color_discrete_map={"No": "#4facfe", "Yes": "#f5576c"},
            barmode="overlay",
            opacity=0.7,
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 2: Predict
# ──────────────────────────────────────────────
with tab2:
    st.subheader("🔮 Predict Customer Churn")
    st.markdown("Enter customer details below and get a churn prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Account**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=5.0)
            total = st.number_input("Total Charges ($)", 0.0, 9000.0, 840.0, step=50.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prob = result["churn_probability"]
                risk = result["risk_level"]

                st.markdown("---")

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Churn Probability", f"{prob:.1%}")
                with col_r2:
                    risk_class = f"risk-{risk.lower()}"
                    st.markdown(f'<p class="{risk_class}">Risk: {risk}</p>', unsafe_allow_html=True)
                with col_r3:
                    st.metric("Prediction", "Will Churn" if result["churn_prediction"] else "Will Stay")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Churn Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 45], "color": "#00cc66"},
                            {"range": [45, 75], "color": "#ffaa00"},
                            {"range": [75, 100], "color": "#ff4444"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the API server is running (`uvicorn api.main:app`).")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ──────────────────────────────────────────────
# TAB 3: Insights
# ──────────────────────────────────────────────
with tab3:
    st.subheader("🧠 Model Insights")

    importance_df = load_feature_importance()

    if importance_df is not None:
        # Top 15 features
        top_features = importance_df.head(15)
        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"autorange": "reversed"},
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available. Train the model first.")

    st.markdown("---")

    # Additional insights from data
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        # Churn by internet service
        internet_churn = df.groupby("InternetService")["Churn_Binary"].mean().reset_index()
        internet_churn.columns = ["Internet Service", "Churn Rate"]
        internet_churn["Churn Rate"] = internet_churn["Churn Rate"] * 100
        fig = px.bar(
            internet_churn,
            x="Internet Service",
            y="Churn Rate",
            title="Churn Rate by Internet Service",
            color="Internet Service",
            color_discrete_sequence=["#667eea", "#f093fb", "#4facfe"],
            text_auto=".1f",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_i2:
        # Churn by payment method
        payment_churn = df.groupby("PaymentMethod")["Churn_Binary"].mean().reset_index()
        payment_churn.columns = ["Payment Method", "Churn Rate"]
        payment_churn["Churn Rate"] = payment_churn["Churn Rate"] * 100
        fig = px.bar(
            payment_churn,
            x="Payment Method",
            y="Churn Rate",
            title="Churn Rate by Payment Method",
            color="Payment Method",
            color_discrete_sequence=["#764ba2", "#f5576c", "#00f2fe", "#fee140"],
            text_auto=".1f",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric columns
    st.markdown("---")
    st.markdown("### Numeric Feature Correlations with Churn")
    numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_Binary"]]
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, use_container_width=True)
