# Insurance-Risk-Predictive-Modeling

## Project Overview

This project was developed for **AlphaCare Insurance Solutions (ACIS)** to analyze historical car insurance claims data and optimize the company's pricing strategy.

The primary business objective was twofold:
1.  **Identify Low-Risk Client Segments** (e.g., specific provinces) for targeted premium reduction to drive customer acquisition.
2.  **Build Predictive Models** to forecast the probability of a claim (`Claim Frequency`) to inform dynamic underwriting and pricing.

---

## 📂 Repository Structure

The project adheres to standard MLOps best practices and uses a structured folder layout:

```

ACIS-Insurance-Risk-Predictive-Modeling/
├── data/
│   ├── external/                 
├── notebooks/
│   ├── EDA.ipynb                
├── src/                         
├── reports/
│   ├── task\_1\_eda\_summary.md     
├── .dvc/                         
├── .dvc\_cache\_storage/           
├── .gitignore
├── README.md                    

```

---

## ✅ Task 1 & 2: Infrastructure and Exploratory Analysis

### Task 1: EDA and Statistical Analysis
The initial phase focused on understanding the data's quality and risk distribution.

* **Data Quality:** Identified critical issues, including **negative values** in `TotalClaims` and **extreme sparsity** (~78% missing) in `CustomValueEstimate`.
* **Core Metric (Loss Ratio):** Calculated the baseline portfolio risk (Overall LR: 1.0477) and segmented risk by `Province` and `Gender`.
    * **Low-Risk Segment Identified:** **Northern Cape** (LR: 0.283) was flagged as the highest potential acquisition target.
    * **High-Risk Segment Identified:** **Gauteng** (LR: 1.222) was flagged for risk mitigation and premium review.

### Task 2: Data Version Control (DVC)
Data Version Control was implemented to manage the large claims data file (`MachineLearningRating_v3.txt`), ensuring reproducibility and separating data storage from Git history.

* DVC was initialized and configured with a local cache (`.dvc_cache_storage`).
* The data file is tracked via a small pointer file, maintaining a lightweight Git repository.

---

## 🧪 Task 3: A/B Hypothesis Testing

A statistical test was performed to validate the observed risk difference between the extreme segments identified in the EDA.

* **Metric:** Average Claim per Policy ($\mu$).
* **Hypotheses:**
    * $H_0: \mu_{\text{Northern Cape}} = \mu_{\text{Gauteng}}$ (No difference in mean claim amount)
    * $H_1: \mu_{\text{Northern Cape}} < \mu_{\text{Gauteng}}$ (Low-risk segment is statistically lower)
* **Method:** Two-Sample One-Tailed Z-Test for Difference in Means.
* **Business Outcome:** The test is designed to statistically confirm the finding, giving ACIS confidence that the observed difference is **not due to random chance** and justifying premium reduction in the Northern Cape.

---

## 📈 Task 4: Statistical Modeling

The final task involved building a classification model to predict **Claim Frequency** ($\text{ClaimFlag} = 1$ if $\text{TotalClaims} > 0$).

### 4.1. Modeling Pipeline

A robust pipeline was created incorporating mandatory preprocessing steps:
1.  **Feature Engineering:** Creation of the **`VehicleAge`** feature.
2.  **Imputation:** Using the median for sparse numerical features (`kilowatts`, `VehicleAge`).
3.  **Encoding:** One-Hot Encoding for categorical variables (`Province`, `Gender`), treating missing values as a separate 'Missing' category.

### 4.2. Model Performance

Three models were trained and evaluated on their ability to predict claims, prioritizing **ROC AUC** and **F1-Score** due to the class imbalance (low frequency of claims).

| Model | Primary Use Case | Performance Metric |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline, Interpretability | Provides coefficient analysis |
| **Random Forest** | Non-Linearity, Feature Ranking | High robustness against noise |
| **XGBoost Classifier** | High Performance, Industry Standard | Typically the highest ROC AUC |

### 4.3. Key Feature Importances (from XGBoost)

Feature importance analysis revealed the most significant drivers of claim frequency risk, guiding future underwriting policies:

--- Top 15 Feature Importances (from XGBoost) --- 
|    | Feature                       | Importance   |
|:---|:------------------------------|:-------------|
| 0  | TotalPremium                  | 0.193834     |
| 1  | SumInsured                    | 0.0736955    |
| 41 | Bank_RMB Private Bank         | 0.0543576    |
| 43 | Bank_nan                      | 0.0340746    |
| 30 | VehicleType_Passenger Vehicle | 0.0336951    |
| 8  | Province_KwaZulu-Natal        | 0.0294227    |
| 19 | MaritalStatus_Not specified   | 0.0269425    |
| 28 | VehicleType_Light Commercial  | 0.0254345    |
| 42 | Bank_Standard Bank            | 0.0217133    |
| 7  | Province_Gauteng              | 0.0215838    |
| 6  | Province_Free State           | 0.0213936    |
| 11 | Province_North West           | 0.0212044    |
| 33 | Bank_Capitec Bank             | 0.0211612    |
| 22 | AccountType_Current account   | 0.0207173    |
| 15 | Gender_Male                   | 0.0203243    |

