# Credit Risk Analysis (Default Prediction)

This project looks at consumer credit data to understand what factors are associated with loan default and how different modeling choices affect risk decisions.

The goal is not to build the most complex model possible, but to show a realistic analytics workflow: exploring the data, building interpretable models, and evaluating tradeoffs that matter in a credit setting.

---

## Problem

Lenders need to decide who to approve for credit while managing default risk.  
Approving too many risky loans leads to losses. Being too conservative reduces revenue.

This project focuses on:
- identifying drivers of default
- comparing an interpretable baseline model with a stronger benchmark
- understanding how decision thresholds change false approvals vs missed defaults

---

## Data

The dataset comes from OpenML (German Credit, `credit-g`).  
It contains borrower and loan attributes such as credit amount, duration, employment status, and savings.

The target variable is a binary default indicator (`1 = default`, `0 = non-default`).

---

## What the code does

1. Splits the data into train, validation, and test sets using stratification  
2. Handles missing values and encodes categorical variables  
3. Trains two models:
   - Logistic regression (baseline, interpretable)
   - Random forest (benchmark, non-linear)
4. Tunes the classification threshold using validation data
5. Evaluates performance on a held-out test set

---

## Evaluation

Model performance is evaluated using metrics that reflect credit risk tradeoffs:
- ROC-AUC
- Precision–recall AUC
- Confusion matrices at tuned thresholds

Threshold tuning highlights how recall for defaults increases at the cost of more false positives, which mirrors real approval decisions.

Plots and metrics are saved automatically for review.

---

## Outputs

After running the pipeline, the following are generated:
- Evaluation plots (ROC, precision–recall, confusion matrices)
- A JSON file with model metrics and thresholds
- Saved model artifacts

---

## Running the project

```bash
python -m venv .venv
pip install -r requirements.txt
python runPipeline.py
