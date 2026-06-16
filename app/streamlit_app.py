"""
RiskLens AI — Interactive Demo (Streamlit)
===========================================
Local web UI for hackathon demo and interviews.

Run:
    streamlit run app/streamlit_app.py

Requires trained models in data/models/ (run pipelines/train.py first).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.scoring.predictor import RiskLensScorer

st.set_page_config(page_title="RiskLens AI", page_icon="📊", layout="wide")

st.title("RiskLens AI")
st.caption("Credit default prediction · risk bucketing · policy decisions")

@st.cache_resource
def load_scorer():
    return RiskLensScorer(use_shap=True)


try:
    scorer = load_scorer()
except FileNotFoundError as exc:
    st.error(f"Models not found. Run `python pipelines/train.py` first.\n\n{exc}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Score a Test Loan")
    st.write(f"Champion model: **{scorer.model_name}** (threshold={scorer.threshold:.3f})")

    if st.button("Load random 2018 test loan", type="primary"):
        st.session_state["sample"] = RiskLensScorer.load_sample_from_test_set(n=1, issue_year=2018)

    sample = st.session_state.get("sample")
    if sample is not None:
        with st.expander("Input features (sample)"):
            st.dataframe(sample.T, use_container_width=True)

        include_shap = st.checkbox("Include SHAP explanation", value=True)
        if st.button("Score application"):
            with st.spinner("Scoring..."):
                result = scorer.score_application(sample, include_shap=include_shap)
            st.session_state["result"] = result

with col2:
    st.subheader("Decision Output")
    result = st.session_state.get("result")
    if result:
        pd_prob = result["probability_of_default"]
        st.metric("Probability of Default", f"{pd_prob:.1%}")

        bucket = result["risk_bucket"]
        policy = result["policy_decision"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Tier", bucket["risk_tier"])
        m2.metric("Credit Grade", bucket["credit_grade"])
        m3.metric("Policy", policy["decision"])

        st.info(bucket["risk_description"])
        st.write(f"**Recommended rate:** {bucket['recommended_interest_rate']}")
        st.write(f"**Bucket action:** {bucket['recommended_action']}")
        st.write(f"**Policy comment:** {policy['comments']}")

        if policy["triggered_rules"]:
            st.warning("Adverse action reasons:\n" + "\n".join(f"- {r}" for r in policy["triggered_rules"]))

        if result.get("shap_top_factors"):
            st.subheader("Top Risk Factors (SHAP)")
            shap_df = pd.DataFrame(result["shap_top_factors"])
            st.dataframe(shap_df[["feature", "feature_value", "impact", "shap_value"]], use_container_width=True)
    else:
        st.write("Load a sample loan and click **Score application**.")

st.divider()
st.subheader("Reports")
reports = project_root / "reports"
if reports.exists():
    charts = list(reports.glob("*.png"))
    if charts:
        selected = st.selectbox("Chart", [p.name for p in charts])
        st.image(str(reports / selected), use_container_width=True)
    else:
        st.write("Run `python scripts/generate_reports.py` to create charts.")
else:
    st.write("No reports folder yet.")
