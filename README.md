# Rental Long-Stay Prediction & Feature Analysis

**Course:** Inżynieria Uczenia Maszynowego (IUM) / Machine Learning Engineering  
**Task:** 7 - Analysis of criteria for booking longer stays  
**Team:** Tomasz Okoń, Andrii-Stepan Pryimak  

## Project Overview
This project aims to support customer service consultants at a rental platform ("Nocarz") by predicting whether a browsing user is likely to book a long-term stay (7-14 days). 

Instead of just providing a binary prediction, the core business value of this system is **explainability**. Using SHAP (SHapley Additive exPlanations), the model highlights the Top 3 key criteria (e.g., high review scores, specific amenities) driving the user's decision, allowing consultants to proactively offer highly personalized recommendations.

## Machine Learning Task
* **Type:** Binary Classification & Feature Importance Analysis.
* **Target Variable:** `is_long_stay` (1 if stay length >= 7 days, 0 otherwise).
* **Models:** Baseline (Dummy Classifier) vs. XGBoost.
* **Key Metrics:** ROC AUC (>= 0.75), Average Precision (AP), and successful SHAP ranking generation.

## Repository Structure
* `/docs` - Contains the Machine Learning Canvas mapping out the business and technical architecture.
* `/notebooks` - Jupyter notebooks containing Exploratory Data Analysis (EDA) and model training experiments.
* `/src` - Source code for data preprocessing and the real-time FastAPI microservice.

## Tech Stack
* **Data Analysis:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Modeling:** `scikit-learn`, `xgboost`, `shap`
* **Deployment:** `FastAPI`, `uvicorn`
