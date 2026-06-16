"""
RiskLens AI — Credit Policy Decision Engine
============================================
Applies a rule-based policy engine (knockout rules) on top of the machine learning
model prediction to determine final credit approval or denial.

Provides compliance-compliant Adverse Action reasons when a loan is denied,
satisfying regulatory requirements (e.g. ECOA / Adverse Action Notices).
"""

from typing import Dict, Any, List

def evaluate_policy_rules(
    default_prob: float,
    dti: float,
    inq_last_6mths: float,
    delinq_2yrs: float,
    annual_inc: float,
    loan_amnt: float
) -> Dict[str, Any]:
    """
    Apply policy knockout rules to decide on loan approval or denial.
    
    Knockout Rules:
    1. Model PD > 35% -> Deny (High Model Risk)
    2. Debt-To-Income (DTI) > 50% -> Deny (Excessive Debt Burden)
    3. Inquiries in last 6 months > 5 -> Deny (Desperate Credit Seeking)
    4. Delinquencies in last 2 years > 3 -> Deny (Unstable Payment History)
    5. Loan Amount > 45% of annual income -> Deny (Over-Leveraged)
    
    Returns:
        decision_dict: Contains 'decision' (Approve/Deny), 'triggered_rules' (reasons),
                       and 'metrics' assessed.
    """
    triggered_rules = []
    
    # 1. Model Risk Cutoff
    if default_prob >= 0.35:
        triggered_rules.append("Default probability exceeds maximum credit risk threshold (35%).")
        
    # 2. DTI Cutoff
    if dti > 50.0:
        triggered_rules.append("Debt-to-Income (DTI) ratio exceeds credit policy limit (50%).")
        
    # 3. Credit Seeking (Inquiries)
    if inq_last_6mths > 5:
        triggered_rules.append("Recent credit inquiries (past 6 months) exceed acceptable limit (max 5).")
        
    # 4. Delinquencies
    if delinq_2yrs > 3:
        triggered_rules.append("Number of delinquencies in past 2 years exceeds policy threshold (max 3).")
        
    # 5. Over-leverage (Loan-to-Income)
    loan_pct_of_income = (loan_amnt / max(1.0, annual_inc)) * 100.0
    if loan_pct_of_income > 45.0:
        triggered_rules.append(f"Requested loan amount is too high relative to income ({loan_pct_of_income:.1f}% of income, limit is 45%).")
        
    # Final Decision
    if triggered_rules:
        decision = "Deny"
        comments = "Application denied based on credit policy rules."
    elif default_prob >= 0.22:
        decision = "Refer"
        comments = "Application referred to manual underwriting due to moderate credit risk."
    else:
        decision = "Approve"
        comments = "Application meets all credit policy criteria."
        
    return {
        "decision": decision,
        "triggered_rules": triggered_rules,
        "adverse_action_reasons": triggered_rules if decision == "Deny" else [],
        "comments": comments,
        "policy_metrics": {
            "dti": float(dti),
            "inq_last_6mths": int(inq_last_6mths),
            "delinq_2yrs": int(delinq_2yrs),
            "loan_to_income_pct": round(float(loan_pct_of_income), 1)
        }
    }
