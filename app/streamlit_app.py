"""
RiskLens AI — Interactive Credit Scoring & Underwriting Dashboard
==================================================================
Includes a premium glassmorphism authentication gate and Role-Based Access 
Control (RBAC) for Credit Underwriters and Risk Managers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import polars as pl

from src.scoring.predictor import RiskLensScorer
from src.features.engineering import _add_features

# Set page config
st.set_page_config(
    page_title="RiskLens AI — Secured Credit Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for the application when logged in
st.markdown("""
<style>
    /* Main body background styling */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    
    /* Styled Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 18px 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }
    
    /* Premium style details */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 15px;
    }
    
    /* Button micro-animations */
    div.stButton > button {
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTHENTICATION GATEWAY (GLASSMORPHISM DESIGN)
# ============================================================
def render_login_page():
    # Centered login page CSS and glassmorphism styling
    st.markdown("""
    <style>
        /* Hide sidebar completely on login screen */
        [data-testid="sidebar-content"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            display: none !important;
        }
        .stApp {
            background: radial-gradient(circle at 10% 20%, #080c14 0%, #111a2e 90.2%) !important;
            color: #f1f5f9 !important;
        }
        
        /* Style the st.container border wrapper to be a glassmorphic login card */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.02) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            padding: 45px 35px !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
            max-width: 460px !important;
            margin: 80px auto !important;
        }
        
        .login-title {
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            letter-spacing: -1.2px;
            text-align: center;
        }
        .login-subtitle {
            color: #64748b;
            font-size: 13.5px;
            margin-bottom: 35px;
            font-weight: 500;
            text-align: center;
        }
        /* Style standard streamlit input labels */
        div[data-testid="stTextInput"] label {
            color: #cbd5e1 !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            margin-bottom: 6px !important;
        }
        div[data-baseweb="input"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 2px !important;
        }
        div[data-baseweb="input"] input {
            color: #f8fafc !important;
            font-size: 14.5px !important;
        }
        /* Styled Login Button */
        div.stButton > button {
            background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            width: 100% !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3) !important;
            margin-top: 18px;
            letter-spacing: 0.5px;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #0369a1 0%, #4338ca 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
        }
        .cred-info {
            background: rgba(15, 23, 42, 0.4);
            border-radius: 12px;
            padding: 18px;
            margin-top: 30px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 12.5px;
            color: #64748b;
            text-align: left;
            line-height: 1.5;
        }
        .cred-info code {
            background: rgba(255, 255, 255, 0.05);
            color: #cbd5e1;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: monospace;
        }
        .cred-info strong {
            color: #94a3b8;
        }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 4, 1])
    with col_c:
        with st.container(border=True):
            st.markdown('<div class="login-title">RiskLens AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-subtitle">Secured Credit Risk Analytics & Scoring Portal</div>', unsafe_allow_html=True)
            
            email_input = st.text_input("Corporate Username / Email", placeholder="underwriter@risklens.ai")
            pass_input = st.text_input("Password", type="password", placeholder="••••••••")
            
            if st.button("Access Portal Gateway", use_container_width=True):
                # Simulated credentials database
                user_db = {
                    "underwriter@risklens.ai": ("underwriter123", "Credit Underwriter", "Jane Doe"),
                    "riskmanager@risklens.ai": ("manager123", "Risk Manager", "John Smith")
                }
                
                if email_input in user_db and user_db[email_input][0] == pass_input:
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = email_input
                    st.session_state["user_role"] = user_db[email_input][1]
                    st.session_state["user_name"] = user_db[email_input][2]
                    st.success("Access Granted. Loading Portal...")
                    st.rerun()
                else:
                    st.error("🔒 Access Denied: Invalid Username or Password Credentials.")
                    
            # Info box removed for production-grade security gateway


# Verify Authentication
if not st.session_state.get("authenticated", False):
    render_login_page()
    st.stop()


@st.cache_resource
def load_scorer():
    # Cache the scorer object as it loads models from disk and initializes SHAP explainers
    return RiskLensScorer(use_shap=True)


# ============================================================
# CORE SCORING & PIPELINE PREPARATION
# ============================================================
try:
    scorer = load_scorer()
except FileNotFoundError as exc:
    st.error(f"Models not found. Please run `python pipelines/train.py` first.\n\n{exc}")
    st.stop()

@st.cache_data
def get_median_template():
    # Load 1,000 samples to establish baseline medians
    df_sample = RiskLensScorer.load_sample_from_test_set(n=1000, issue_year=2018)
    
    # Calculate median for numeric columns
    median_series = df_sample.median(numeric_only=True)
    median_df = pd.DataFrame([median_series])
    
    # Fill categoricals with mode or first non-null
    for col in df_sample.columns:
        if col not in median_df.columns:
            mode_vals = df_sample[col].mode()
            median_df[col] = mode_vals.iloc[0] if not mode_vals.empty else df_sample[col].iloc[0]
            
    return median_df[df_sample.columns]

def update_engineered_features(df_pd: pd.DataFrame) -> pd.DataFrame:
    df_pl = pl.DataFrame(df_pd)
    df_pl = _add_features(df_pl)
    return df_pl.to_pandas()


# ============================================================
# MAIN PORTAL HEADER & NAVIGATION
# ============================================================
role_color = "#38bdf8" if st.session_state["user_role"] == "Credit Underwriter" else "#a78bfa"
st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.7) 100%); 
            padding: 24px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); 
            margin-bottom: 25px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #f8fafc; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">RiskLens AI Portal</h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">Welcome back, <strong>{st.session_state['user_name']}</strong>.</p>
        </div>
        <div style="background-color: rgba(255, 255, 255, 0.03); padding: 8px 16px; border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.08);">
            <span style="color: {role_color}; font-weight: 800; font-size: 12px; letter-spacing: 0.8px;">🛡️ {st.session_state['user_role'].upper()} MODE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown(f"""
<div style="background-color: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 20px;">
    <p style="margin:0; font-size:12px; color:#64748b;">Operator Profile</p>
    <p style="margin:2px 0; font-size:14px; font-weight:700; color:#f1f5f9;">{st.session_state['user_name']}</p>
    <p style="margin:0; font-size:12px; color:#94a3b8;">{st.session_state['user_email']}</p>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Select Scoring Interface",
    ["Quick Demo (Random Test Loan)", "Manual Underwriting Form"]
)

# Persona Preset Definitions
personas = {
    "Custom (Manual Adjustments)": None,
    "Prime Borrower (Very Low Risk)": {
        "loan_amnt": 15000.0,
        "term": 36,
        "annual_inc": 125000.0,
        "dti": 10.0,
        "revol_bal": 6000.0,
        "revol_util": 12.5,
        "delinq_2yrs": 0,
    },
    "Subprime Borrower (High Risk)": {
        "loan_amnt": 28000.0,
        "term": 60,
        "annual_inc": 42000.0,
        "dti": 32.5,
        "revol_bal": 25000.0,
        "revol_util": 88.0,
        "delinq_2yrs": 2,
    },
    "Debt-Stressed Applicant": {
        "loan_amnt": 30000.0,
        "term": 60,
        "annual_inc": 55000.0,
        "dti": 48.5,
        "revol_bal": 14000.0,
        "revol_util": 72.0,
        "delinq_2yrs": 0,
    },
    "High Risk (Multiple Delinquencies)": {
        "loan_amnt": 10000.0,
        "term": 36,
        "annual_inc": 48000.0,
        "dti": 18.0,
        "revol_bal": 7500.0,
        "revol_util": 35.0,
        "delinq_2yrs": 4,
    }
}

# Persona Selector Callback
def on_persona_change():
    selected = st.session_state["selected_persona"]
    if selected and personas[selected]:
        for k, v in personas[selected].items():
            st.session_state[k] = v

# Initialize widgets states
initial_values = {
    "loan_amnt": 15000.0,
    "term": 36,
    "annual_inc": 75000.0,
    "dti": 15.0,
    "revol_bal": 12000.0,
    "revol_util": 35.0,
    "delinq_2yrs": 0
}
for key, val in initial_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

sample_row = None

if mode == "Quick Demo (Random Test Loan)":
    st.sidebar.subheader("Dataset Ingest")
    if st.sidebar.button("Load Random 2018 Test Loan", type="primary"):
        st.session_state["sample"] = RiskLensScorer.load_sample_from_test_set(n=1, issue_year=2018)
        st.session_state["result"] = None

    if st.session_state.get("sample") is not None:
        sample_row = st.session_state["sample"]
        st.sidebar.success("Loaded historical borrower profile.")
    else:
        st.sidebar.info("Ingest a random borrower from the dataset above.")

else:
    st.sidebar.subheader("Interactive Underwriter")
    st.sidebar.selectbox(
        "Load Borrower Preset",
        options=list(personas.keys()),
        key="selected_persona",
        on_change=on_persona_change
    )

    st.sidebar.divider()
    st.sidebar.subheader("Underwriting Parameters")
    
    loan_amnt = st.sidebar.slider("Loan Amount ($)", 1000, 40000, step=500, key="loan_amnt")
    term = st.sidebar.selectbox("Term (Months)", [36, 60], key="term")
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=5000, max_value=500000, step=1000, key="annual_inc")
    dti = st.sidebar.slider("Debt-to-Income (DTI) Ratio (%)", 0.0, 60.0, step=0.5, key="dti")
    revol_bal = st.sidebar.number_input("Outstanding Revolving Balance ($)", min_value=0.0, max_value=250000.0, step=500.0, key="revol_bal")
    revol_util = st.sidebar.slider("Revolving Utilization Rate (%)", 0.0, 120.0, step=0.5, key="revol_util")
    delinq_2yrs = st.sidebar.slider("Delinquencies (Last 2 Years)", 0, 12, step=1, key="delinq_2yrs")

    # Combine sliders with median template
    base_df = get_median_template().copy()
    base_df["loan_amnt"] = float(loan_amnt)
    base_df["term"] = int(term)
    base_df["annual_inc"] = float(annual_inc)
    base_df["dti"] = float(dti)
    base_df["revol_bal"] = float(revol_bal)
    base_df["revol_util"] = float(revol_util)
    base_df["delinq_2yrs"] = int(delinq_2yrs)

    # Dynamic Feature engineering
    sample_row = update_engineered_features(base_df)
    st.session_state["sample"] = sample_row

st.sidebar.divider()
if st.sidebar.button("🚪 Terminate Session (Logout)", type="secondary", use_container_width=True):
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["user_role"] = None
    st.session_state["user_name"] = None
    st.rerun()


# ============================================================
# MAIN VIEW LAYOUT
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Borrower Profile Details")
    st.write(f"Scoring Engine: **{scorer.model_name}** (threshold={scorer.threshold:.3f})")

    if sample_row is not None:
        # Group and categorize features for clean tabular presentation
        with st.expander("Explore Borrower Profile Attributes", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Loan Details", 
                "💼 Capacity & Income", 
                "💳 Credit History", 
                "📈 Derived ML Ratios"
            ])
            
            with tab1:
                st.markdown("**Loan Specifications**")
                loan_disp = sample_row[["loan_amnt", "term", "purpose", "addr_state"]].copy()
                loan_disp.columns = ["Loan Amount ($)", "Term (Months)", "Loan Purpose", "State"]
                st.dataframe(loan_disp.astype(str).T, use_container_width=True)
                
            with tab2:
                st.markdown("**Borrower Capacity**")
                cap_disp = sample_row[["annual_inc", "dti", "emp_length_num"]].copy()
                cap_disp.columns = ["Annual Income ($)", "Debt-to-Income (%)", "Employment Length (Years)"]
                st.dataframe(cap_disp.astype(str).T, use_container_width=True)
                
            with tab3:
                st.markdown("**Credit Bureau Records**")
                credit_disp = sample_row[["revol_bal", "revol_util", "delinq_2yrs", "open_acc", "total_acc", "credit_history_months"]].copy()
                credit_disp.columns = ["Revolving Balance ($)", "Revolving Util (%)", "Delinquencies (2y)", "Open Accounts", "Total Accounts", "Credit Age (Months)"]
                st.dataframe(credit_disp.astype(str).T, use_container_width=True)
                
            with tab4:
                st.markdown("**Engineered Ratio Features**")
                ratio_disp = sample_row[["income_to_loan", "loan_pct_of_income", "monthly_debt_dollar", "credit_util_x_balance", "delinq_score", "revol_bal_vs_limit"]].copy()
                ratio_disp.columns = ["Income-to-Loan", "Loan % of Income", "Monthly Debt Burden ($)", "Utilization x Balance", "Delinquency Score", "Revol Bal vs Limit"]
                st.dataframe(ratio_disp.astype(str).T, use_container_width=True)

        include_shap = st.checkbox("Include SHAP Risk Explanations", value=True)
        if st.button("Score Borrower Application", type="primary", use_container_width=True):
            with st.spinner("Analyzing credit profile..."):
                result = scorer.score_application(sample_row, include_shap=include_shap)
            st.session_state["result"] = result
    else:
        st.info("👈 Please load a test loan or switch to Manual Underwriting mode in the sidebar to begin.")

with col2:
    st.subheader("Decision & Underwriting Output")
    result = st.session_state.get("result")
    
    if result:
        pd_prob = result["probability_of_default"]
        
        # Color coding metrics based on decision
        decision = result["policy_decision"]["decision"]
        if decision == "Approve":
            decision_color = "#10b981"
            bg_color = "rgba(16, 185, 129, 0.08)"
        elif decision == "Refer":
            decision_color = "#f59e0b"
            bg_color = "rgba(245, 158, 11, 0.08)"
        else:
            decision_color = "#ef4444"
            bg_color = "rgba(239, 68, 68, 0.08)"

        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 22px; border-radius: 10px; border: 2px solid {decision_color}; margin-bottom: 22px;">
            <h3 style="color: {decision_color}; margin-top: 0; font-weight: 800; font-size: 20px;">RECOMMENDATION: {decision.upper()}</h3>
            <p style="margin: 0; font-size: 15px; color:#f1f5f9;"><strong>Probability of Default (PD):</strong> {pd_prob:.2%}</p>
            <p style="margin: 5px 0 0 0; font-size: 13.5px; color:#94a3b8;"><strong>Credit Policy Status:</strong> {result['policy_decision']['comments']}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Tier", result["risk_bucket"]["risk_tier"])
        m2.metric("Credit Grade", result["risk_bucket"]["credit_grade"])
        m3.metric("Rec. Rate Band", result["risk_bucket"]["recommended_interest_rate"])

        st.info(f"**Risk Profile Note:** {result['risk_bucket']['risk_description']}")

        # Display Policy block warning triggers if any exist
        triggered_rules = result["policy_decision"]["triggered_rules"]
        if triggered_rules:
            st.warning("⚠️ **Compliance & Adverse Action Reasons:**\n" + "\n".join(f"- {r}" for r in triggered_rules))

        # Show SHAP breakdown if requested
        if result.get("shap_top_factors"):
            st.markdown("---")
            st.subheader("Top Risk Drivers (Local SHAP Explanations)")
            shap_df = pd.DataFrame(result["shap_top_factors"])
            
            # Map contributions to readable labels
            shap_df["Impact Direction"] = shap_df["shap_value"].apply(
                lambda x: "📈 Increases Risk" if x > 0 else "📉 Decreases Risk"
            )
            shap_disp = shap_df[["feature", "feature_value", "Impact Direction"]].copy()
            shap_disp.columns = ["Credit Metric", "Borrower Value", "Risk Contribution Impact"]
            st.dataframe(shap_disp, use_container_width=True)
    else:
        st.write("Score the borrower's profile to view the underwriting decision output.")


# ============================================================
# ROLE-BASED ACCESS CONTROL (RBAC) REPORTS SECTION
# ============================================================
if st.session_state["user_role"] == "Risk Manager":
    st.divider()
    st.subheader("📊 Global Portfolio & Validation Reports (Admin Only)")
    
    # 1. Expected Loss Dashboard Section
    reports_dir = project_root / "reports"
    summary_csv = reports_dir / "portfolio_summary.csv"
    
    if summary_csv.exists():
        try:
            df_summary = pd.read_csv(summary_csv)
            total_el = df_summary["total_expected_loss"].sum()
            total_exposure = df_summary["total_exposure"].sum()
            el_rate = (total_el / total_exposure) * 100
            
            st.markdown("### Portfolio Expected Loss Summary")
            
            # Draw KPI Cards for Risk Manager
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Exposure (EAD)", f"${total_exposure:,.0f}")
            c2.metric("Total Expected Loss (EL)", f"${total_el:,.0f}")
            c3.metric("Expected Loss Rate", f"{el_rate:.2f}%")
            
            # Format and show tabular breakdown
            st.markdown("**Expected Loss Breakdown by Risk Tier**")
            df_disp = df_summary.copy()
            df_disp["total_exposure"] = df_disp["total_exposure"].apply(lambda x: f"${x:,.0f}")
            df_disp["total_expected_loss"] = df_disp["total_expected_loss"].apply(lambda x: f"${x:,.0f}")
            df_disp["avg_pd"] = df_disp["avg_pd"].apply(lambda x: f"{x:.2%}")
            df_disp["actual_default_rate"] = df_disp["actual_default_rate"].apply(lambda x: f"{x:.2%}")
            df_disp["pct_of_portfolio"] = df_disp["pct_of_portfolio"].apply(lambda x: f"{x:.1f}%")
            df_disp["pct_of_exposure"] = df_disp["pct_of_exposure"].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_disp, use_container_width=True)
            st.markdown("---")
        except Exception as e:
            st.error(f"Error loading portfolio Expected Loss summary: {e}")
            
    # 2. Charts dropdown section
    if reports_dir.exists():
        charts = list(reports_dir.glob("*.png"))
        if charts:
            st.markdown("### Model Validation Curves & Feature Importances")
            selected_chart = st.selectbox("Select Portfolio Chart / Curve to view", [p.name for p in charts])
            st.image(str(reports_dir / selected_chart), use_container_width=True)
        else:
            st.info("Run `python scripts/generate_reports.py` to pre-generate global analysis plots.")
    else:
        st.info("No global reports directory found.")
else:
    st.divider()
    st.info("ℹ️ **Risk Manager Privilege Required:** Global validation curves, Expected Loss (EL) analysis, and portfolio concentration tables are restricted to Risk Manager roles. Current role: `Credit Underwriter`.")
