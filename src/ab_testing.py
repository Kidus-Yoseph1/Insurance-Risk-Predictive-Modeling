import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest


try:
    df = pd.read_csv('../data/external/MachineLearningRating_v3.txt', sep='|')
    print("Data re-loaded for Task 3 execution.")
except FileNotFoundError:
    print("Error: Ensure the file path '../data/external/MachineLearningRating_v3.txt' is correct.")
  

# DEFINE GROUP NAMES 
# Based on your LR output: Northern Cape (low risk) and Gauteng (high risk)
NORTHERN_CAPE = 'Northern Cape' 
GAUTENG = 'Gauteng'
ALPHA = 0.05

# DATA PREPARATION FUNCTION (Ensures Clean Data is Used) 
def prepare_claim_data(df, group_col, group_name):
    """Filters the DataFrame for the specified group and returns an array of non-negative claim values."""
    # Crucial step: Filters for the group AND ensures TotalClaims are non-negative (as per EDA findings)
    df_group = df[(df[group_col] == group_name) & (df['TotalClaims'] >= 0)].copy()
    
    # Return the raw array of claims per policy
    return df_group['TotalClaims'].values 

# PERFORM HYPOTHESIS TEST
def run_z_test_province(df, group_A_name, group_B_name, alpha=ALPHA):
    """Performs a two-sample one-tailed Z-test to compare Average Claim per Policy between two provinces."""
    
    claims_A = prepare_claim_data(df, 'Province', group_A_name)
    claims_B = prepare_claim_data(df, 'Province', group_B_name)
    
    if len(claims_A) < 30 or len(claims_B) < 30:
        print("Error: Insufficient sample size for Z-test after data cleaning.")
        return 
        
    # Z-test: Compares means of two independent samples.
    # 'alternative="smaller"' tests if mean_A < mean_B (H1: Northern Cape is smaller)
    z_stat, p_value = ztest(x1=claims_A, x2=claims_B, value=0, alternative='smaller')
    
    # Calculate means and sizes for reporting
    mean_A = claims_A.mean()
    mean_B = claims_B.mean()
    
    print(f"\n--- A/B Hypothesis Test Results: Avg Claim per Policy ---")
    print(f"Group A ({group_A_name} - Low Risk): Mean Claim = {mean_A:,.2f} ZAR (N={len(claims_A):,})")
    print(f"Group B ({group_B_name} - High Risk): Mean Claim = {mean_B:,.2f} ZAR (N={len(claims_B):,})")
    print("-" * 65)
    print(f"Z-Statistic: {z_stat:.4f}")
    print(f"P-Value: {p_value:.10f}") 
    print(f"Significance Level (α): {alpha}")
    print("-" * 65)
    
    # Draw Conclusion
    if p_value < alpha:
        print(f"CONCLUSION: Reject the Null Hypothesis (H0).")
        print(f"The average claim per policy in {group_A_name} is STATISTICALLY SIGNIFICANTLY LOWER than in {group_B_name}. The observed difference is not due to chance.")
    else:
        print(f"CONCLUSION: Fail to Reject the Null Hypothesis (H0).")
        print(f"There is NO statistical evidence that the average claim in {group_A_name} is lower than in {group_B_name} at the {alpha*100}% level.")


run_z_test_province(df, NORTHERN_CAPE, GAUTENG)

