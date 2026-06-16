"""
RiskLens AI — Risk Bucketing Module
====================================
Calibrates model default probabilities into 5 standardized credit risk tiers
(Very Low, Low, Medium, High, Very High) and provides pricing guidelines
(interest rate premium) based on the risk level.
"""

from typing import Dict, Any

def assign_risk_bucket(default_prob: float) -> Dict[str, Any]:
    """
    Map a probability of default (PD) to a standardized credit risk tier,
    including grading, pricing recommendations (interest rate), and risk descriptions.

    Design: Tiers are calibrated around the LendingClub average baseline default rate (~20%).
    """
    # Clamp probability
    prob = max(0.0, min(1.0, float(default_prob)))
    
    if prob < 0.05:
        tier = "Very Low"
        grade = "A"
        interest_rate_range = "5.0% - 8.5%"
        description = "Excellent credit profile. Extremely low likelihood of default."
        action = "Approve"
    elif prob < 0.12:
        tier = "Low"
        grade = "B"
        interest_rate_range = "8.6% - 12.0%"
        description = "Solid credit profile. Below average default risk."
        action = "Approve"
    elif prob < 0.22:
        tier = "Medium"
        grade = "C"
        interest_rate_range = "12.1% - 17.5%"
        description = "Moderate credit risk. Close to baseline default rate."
        action = "Approve (Review)"
    elif prob < 0.35:
        tier = "High"
        grade = "D"
        interest_rate_range = "17.6% - 24.0%"
        description = "Elevated risk profile. Substantially above average default probability."
        action = "Conditional Approve"
    else:
        tier = "Very High"
        grade = "E/F"
        interest_rate_range = "N/A"
        description = "Critical risk profile. Default probability exceeds acceptable thresholds."
        action = "Deny"
        
    return {
        "probability_of_default": prob,
        "risk_tier": tier,
        "credit_grade": grade,
        "recommended_interest_rate": interest_rate_range,
        "risk_description": description,
        "recommended_action": action
    }
