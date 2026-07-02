# Rainfall Forecasting System

Supervised machine-learning model that predicts **next-day rainfall** from a city's historical weather data.

## Overview
A binary "will it rain tomorrow?" prediction task on a real weather dataset with two practical challenges: **class imbalance** (rain days are the minority) and **substantial missing data**. The project focuses on the preprocessing needed to make such data model-ready, then trains and evaluates supervised classifiers.

## Approach
1. **Data cleaning** - impute/handle substantial missing values across weather features.
2. **Imbalance handling** - address the skewed rain/no-rain ratio with SMOTE so the model doesn't collapse to the majority class.
3. **Modeling** - train supervised classifiers (KNN, Decision Tree, SVM) to predict next-day rain.
4. **Evaluation** - assess with metrics appropriate for imbalanced data (not just accuracy).

## Tech stack
Python, NumPy, Pandas, scikit-learn, imbalanced-learn (SMOTE). Jupyter Notebook.

## Results (validation)
| Model | Accuracy | Recall (rain) | Precision (rain) | F1 |
|---|---|---|---|---|
| **SVM** | **~80%** | **0.77** | 0.52 | 0.62 |
| KNN | ~76% | 0.74 | 0.46 | 0.57 |
| Decision Tree | ~76% | 0.56 | 0.46 | 0.50 |

Class imbalance (~1:3 rain:no-rain) was addressed with **SMOTE**. SVM gives the best rain-day recall while keeping accuracy near 80%.

## Run it
```
pip install numpy pandas scikit-learn imbalanced-learn
jupyter notebook
```
