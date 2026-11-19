# Credit Risk Scoring & Default Prediction

## Project Overview

An end-to-end Data Science project focused on credit risk assessment and default prediction using machine learning, statistical analysis, financial risk modeling, and explainable AI.

The objective is to estimate the Probability of Default (PD) of credit card customers and explore data-driven strategies for credit portfolio risk management.

## Dataset

- Source: UCI Machine Learning Repository — Default of Credit Card Clients.
- Observations: 30,000 customers.
- Original features: 23.
- Target: Binary default indicator.
- Default prevalence: 22.12%.
- Historical period: April–September 2005, Taiwan.

## Technologies

Python | Pandas | NumPy | SciPy | Scikit-learn | XGBoost | LightGBM | Matplotlib | Seaborn | SHAP | Jupyter Notebook

## Methodology

1. Exploratory Data Analysis and Data Quality.
2. Statistical Hypothesis Testing.
3. Financial Feature Engineering.
4. Machine Learning Model Comparison.
5. Banking Model Validation.
6. Hyperparameter and Threshold Optimization.
7. Expected Loss Estimation.
8. Model Explainability and Fairness Auditing.

## Model Comparison

| Model | Cross-validation Average Precision |
|---|---:|
| XGBoost | ~0.562 |
| LightGBM | ~0.561 |
| Random Forest | ~0.555 |
| Logistic Regression | ~0.530 |
| Dummy Classifier | ~0.221 |

## Baseline XGBoost Test Results

Classification threshold: 0.50.

| Metric | Result |
|---|---:|
| Accuracy | 82.02% |
| Precision | 66.89% |
| Recall | 37.00% |
| F1-score | 47.65% |
| Specificity | 94.80% |
| First-decile Lift | ~3.13x |

The top 20% of customers ranked by predicted risk concentrated approximately 51% of observed defaults.

The model demonstrated favorable visual probability calibration and useful risk segmentation.

## Banking Risk Metrics

The project incorporates:

- Probability of Default (PD).
- Loss Given Default (LGD).
- Exposure at Default (EAD).
- Expected Loss (EL).
- Gini coefficient.
- Kolmogorov-Smirnov statistic.
- Risk deciles, Lift and cumulative Gain.
- Brier Score and probability calibration.

Expected Loss is estimated as:

**EL = PD × LGD × EAD**

LGD and EAD are scenario-based estimates rather than observed regulatory risk parameters.

## Business Insights

- Financial behavior provides useful predictive information for default risk ranking.
- The conventional classification threshold of 0.50 resulted in limited recall.
- Unconstrained financial cost optimization produced an excessively low classification threshold.
- Operational constraints are necessary to develop a practical risk management policy.
- Predictive performance, financial costs, explainability and fairness must be evaluated together.

## Explainable AI and Fairness

The explainability workflow includes:

- SHAP global feature importance.
- Beeswarm plots.
- Individual Waterfall explanations.
- Feature dependence analysis.
- Demographic subgroup performance.
- Equalized odds gap.
- Calibration and error-rate comparisons.

Demographic attributes are excluded from the baseline predictive feature set and retained for auditing purposes.

## Reproducibility

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Open JupyterLab:

```bash
jupyter lab
```

Execute the notebooks sequentially from 01 to 07.

## Limitations

This project uses historical credit card data from Taiwan and is intended for educational and portfolio purposes.

Financial assumptions are simulated. Test comparisons are retrospective, and external temporal validation is required before considering real-world banking deployment.

The model should not be used for automated lending decisions.

## Project Status

- Data analysis and statistical testing.
- Feature engineering.
- Model training and evaluation.
- Financial risk analysis.
- Explainability and fairness audit workflow.
- Model deployment and monitoring: planned.

## Author

Julio César Valderrábano Hernández

Data Science | Machine Learning | Statistical Analysis | Credit Risk Analytics
