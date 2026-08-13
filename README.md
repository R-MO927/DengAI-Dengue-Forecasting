# DengAI: Dengue Fever Forecasting

<p align="center">
  <a href="https://dengai-dengue-forecasting.streamlit.app/" target="_blank">
    <img
      src="https://img.shields.io/badge/%F0%9F%A6%9F%20ENTER%20DENGAI-4A148C?style=for-the-badge&labelColor=1A1A1A"
      alt="Enter DengAI"
    />
  </a>
</p>


A machine learning project for forecasting weekly dengue fever cases using historical disease observations, weather conditions, environmental indicators, and temporal features.

The project focuses on two cities with different epidemiological and environmental patterns:

- San Juan
- Iquitos

The main objective is to build a reliable regression-based forecasting system while preserving the temporal structure of the data and avoiding information leakage.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Definition](#problem-definition)
3. [Dataset](#dataset)
4. [Understanding the Data](#understanding-the-data)
5. [Initial Data Analysis](#initial-data-analysis)
6. [Data Preprocessing](#data-preprocessing)
7. [Temporal Train-Validation Split](#temporal-train-validation-split)
8. [Initial Modeling](#initial-modeling)
9. [Initial Model Results](#initial-model-results)
10. [Key Finding: Temporal Dependency](#key-finding-temporal-dependency)
11. [Temporal Feature Engineering](#temporal-feature-engineering)
12. [Models with Lag Features](#models-with-lag-features)
13. [Model Comparison](#model-comparison)
14. [Final Model Selection](#final-model-selection)
15. [Feature Importance](#feature-importance)
16. [Error Analysis](#error-analysis)
17. [Residual Analysis](#residual-analysis)
18. [Final Training](#final-training)
19. [Model Persistence](#model-persistence)
20. [Final Results](#final-results)
21. [Project Workflow](#project-workflow)
22. [Project Structure](#project-structure)
23. [Technologies](#technologies)
24. [Limitations](#limitations)
25. [Future Improvements](#future-improvements)
26. [Deployment](#deployment)
27. [Conclusion](#conclusion)

---

# Project Overview

Dengue fever is a mosquito-borne disease whose incidence can vary significantly across time and geographic regions.

Forecasting dengue cases is challenging because disease activity is influenced by multiple interacting factors, including weather, environmental conditions, seasonality, and previous disease activity.

This project investigates whether machine learning models can effectively forecast weekly dengue cases by combining:

- Historical dengue case counts
- Meteorological variables
- Environmental indicators
- Seasonal information
- Temporal lag features

A major focus of the project was understanding the structure of the data before selecting the final machine learning approach.

Rather than assuming that the most complex algorithm would perform best, multiple models were evaluated systematically and the final model was selected based on validation performance.

---

# Problem Definition

This project is formulated as a supervised regression problem.

The target variable is:

```text
total_cases
```

which represents the number of dengue cases reported for a specific week.

The objective is to learn a function:

```text
X → total_cases
```

where `X` contains environmental, weather, temporal, and historical disease information.

The problem is different from a standard tabular regression task because observations are ordered in time.

The number of dengue cases in one week can be strongly related to the number of cases observed in previous weeks.

Therefore, preserving temporal order is an essential part of the modeling strategy.

---

# Dataset

The dataset contains weekly dengue observations for two cities:

- San Juan
- Iquitos

The available variables can be grouped into several categories.

## Target Variable

```text
total_cases
```

The number of dengue cases reported during the corresponding week.

---

## Temporal Variables

```text
year
weekofyear
```

These variables provide information about the position of each observation in time and help the model capture seasonal patterns.

---

## Reanalysis Weather Variables

The dataset contains several weather variables derived from reanalysis data, including:

```text
reanalysis_air_temp_k
reanalysis_avg_temp_k
reanalysis_max_air_temp_k
reanalysis_min_air_temp_k
reanalysis_relative_humidity_percent
reanalysis_specific_humidity_g_per_kg
reanalysis_dew_point_temp_k
reanalysis_precip_amt_kg_per_m2
reanalysis_tdtr_k
```

These variables describe atmospheric temperature, humidity, and precipitation conditions.

---

## Environmental Variables

Environmental information is represented using NDVI measurements:

```text
ndvi_ne
ndvi_nw
ndvi_se
ndvi_sw
```

A derived environmental feature was also used:

```text
ndvi_mean
```

---

## Weather Station Variables

Station-based measurements include:

```text
station_avg_temp_c
station_diur_temp_rng_c
station_max_temp_c
station_min_temp_c
station_precip_mm
```

These provide additional localized measurements of temperature and precipitation.

---

# Understanding the Data

The project began with exploratory analysis rather than immediately training machine learning models.

The initial objective was to understand:

- The structure of the dataset
- The target variable
- The available feature groups
- Missing-value patterns
- Differences between the two cities
- Temporal behavior of dengue cases
- Relationships between disease activity and environmental variables

A key observation was that San Juan and Iquitos do not exhibit identical dengue dynamics.

Their distributions and temporal behavior differ substantially.

This eventually motivated the decision to maintain separate predictive models for the two cities.

---

# Initial Data Analysis

The initial analysis revealed several important characteristics.

## Different City Dynamics

San Juan and Iquitos showed different distributions and temporal behavior.

Although both datasets represent dengue cases, the underlying relationships between disease activity and environmental conditions are not identical.

Therefore, a single model trained across both cities was not considered the ideal final approach.

Instead:

```text
San Juan → Independent Model
Iquitos  → Independent Model
```

was adopted.

---

## Missing Values

Several features contained missing observations.

The missing-value analysis showed that NDVI variables contained substantially more missing values than many of the weather variables.

Missing values were therefore handled before model training.

The preprocessing pipeline ensured that the final training and validation matrices contained no missing values.

---

## Temporal Dependency

The most important observation from the exploratory stage was the sequential nature of the target.

Dengue cases were not independent from one week to the next.

Periods of higher disease activity tended to be followed by related levels of activity in subsequent weeks.

This suggested that previous dengue case counts could potentially be stronger predictors than some environmental variables.

This observation became the basis for the main feature-engineering step of the project.

---

# Data Preprocessing

The preprocessing stage included:

1. Separating the data by city.
2. Sorting observations chronologically.
3. Separating features from the target.
4. Handling missing values.
5. Preparing training and validation datasets.
6. Ensuring that temporal ordering was preserved.

The target variable was:

```text
total_cases
```

while the remaining selected variables were used as predictors.

---

# Temporal Train-Validation Split

A standard random train-test split was intentionally avoided.

Random splitting would allow observations from later periods to appear in the training set while earlier observations appear in validation, which does not represent a realistic forecasting scenario.

Instead, the data was sorted chronologically and divided using an approximately 80/20 time-based split.

## San Juan

```text
Training:   748 observations
Validation: 188 observations
```

## Iquitos

```text
Training:   416 observations
Validation: 104 observations
```

The validation set therefore represents a later time period that the model did not see during training.

This approach better simulates real-world forecasting:

```text
Historical observations
        ↓
     Training
        ↓
Future unseen period
        ↓
    Validation
```

---

# Initial Modeling

Before introducing temporal lag features, several regression models were evaluated using the original feature representation.

The purpose of this stage was to establish a baseline and determine whether weather and environmental variables alone were sufficient to model dengue cases.

The evaluated models included:

- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- XGBoost Regressor

The models were evaluated separately for San Juan and Iquitos.

---

# Initial Model Results

## Random Forest

The initial Random Forest model provided limited predictive performance.

The model was able to capture some nonlinear relationships between the environmental variables and dengue cases, but the validation results were not sufficiently strong.

---

## Gradient Boosting

Gradient Boosting improved the results for San Juan compared with the initial Random Forest.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | 17.4576 | 29.7814 | 0.0873 |
| Iquitos | 7.0657 | 12.1347 | -0.1292 |

The model performed better for San Juan but still failed to explain a large portion of the target variation.

---

## HistGradientBoosting

HistGradientBoosting was also evaluated.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | 19.6051 | 32.4832 | -0.0858 |
| Iquitos | 8.0006 | 11.7788 | -0.0639 |

The model did not provide a sufficient improvement.

---

## XGBoost

XGBoost was evaluated as a more advanced gradient-boosting method.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | 19.6898 | 34.6050 | -0.2323 |
| Iquitos | 7.2581 | 12.2636 | -0.1533 |

Despite XGBoost being a powerful algorithm, it did not perform well with the original feature representation.

This was an important modeling observation:

> Algorithmic complexity alone does not guarantee better performance. The quality and relevance of the feature representation are critical.

---

# Key Finding: Temporal Dependency

The initial experiments revealed that the main limitation was not necessarily the choice of regression algorithm.

The models were being asked to predict future dengue activity using environmental and weather variables without directly providing them with information about previous dengue activity.

For a time-dependent disease forecasting problem, this is a major limitation.

The modeling problem was therefore reformulated.

Instead of relying only on:

```text
Weather + Environment
```

the models would also receive:

```text
Previous Dengue Cases
```

This led to temporal feature engineering.

---

# Temporal Feature Engineering

Four lag features were created from the historical dengue case counts:

```text
cases_lag_1
cases_lag_2
cases_lag_4
cases_lag_12
```

They represent:

| Feature | Meaning |
|---|---|
| `cases_lag_1` | Dengue cases from the previous week |
| `cases_lag_2` | Dengue cases from two weeks earlier |
| `cases_lag_4` | Dengue cases from four weeks earlier |
| `cases_lag_12` | Dengue cases from twelve weeks earlier |

The temporal features were created separately for each city while preserving chronological order.

Conceptually:

```text
Week t
│
├── cases_lag_1  → Week t-1
├── cases_lag_2  → Week t-2
├── cases_lag_4  → Week t-4
└── cases_lag_12 → Week t-12
```

The first observations naturally contain missing values because historical information is unavailable for those periods.

These rows were handled during the preparation of the final modeling datasets.

---

# Why Lag Features Were Important

Before temporal feature engineering, the model had access primarily to external explanatory variables:

```text
Weather
Environment
Seasonality
        ↓
      Model
        ↓
Dengue Prediction
```

After temporal feature engineering:

```text
Weather
Environment
Seasonality
Previous Dengue Cases
        ↓
      Model
        ↓
Dengue Prediction
```

This allowed the models to learn the persistence and short-term dynamics of dengue activity.

The improvement in validation performance confirmed that historical case information was highly informative.

---

# Models with Lag Features

After introducing the lag features, the same model families were evaluated again.

This created a fair comparison because the models received the same temporal feature representation.

---

# Random Forest + Lag Features

The Random Forest model was retrained using the expanded feature set.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | **7.3211** | **11.0796** | **0.8737** |
| Iquitos | **4.3499** | **6.8189** | **0.6434** |

This represented a substantial improvement compared with the initial experiments.

Random Forest became the strongest overall candidate.

---

# Gradient Boosting + Lag Features

Gradient Boosting was retrained using the same temporal feature set.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | 7.4333 | 11.9053 | 0.8541 |
| Iquitos | 4.5963 | 7.4373 | 0.5758 |

The model performed well, but Random Forest remained slightly better for both cities.

---

# XGBoost + Lag Features

XGBoost was also retrained using the same lag features.

Results:

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| San Juan | 7.4530 | 12.8152 | 0.8310 |
| Iquitos | 4.4435 | 7.0634 | 0.6174 |

XGBoost performed significantly better after temporal feature engineering.

However, it still did not outperform Random Forest.

This illustrates an important practical machine learning principle:

> The best-performing algorithm is dataset-dependent.

---

# Model Comparison

The main candidates were compared using the same validation methodology.

| Model | SJ MAE ↓ | SJ RMSE ↓ | SJ R² ↑ | IQ MAE ↓ | IQ RMSE ↓ | IQ R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest + Lag | **7.3211** | **11.0796** | **0.8737** | **4.3499** | **6.8189** | **0.6434** |
| Gradient Boosting + Lag | 7.4333 | 11.9053 | 0.8541 | 4.5963 | 7.4373 | 0.5758 |
| XGBoost + Lag | 7.4530 | 12.8152 | 0.8310 | 4.4435 | 7.0634 | 0.6174 |
| Tuned Random Forest + Lag | 9.6759 | 14.9253 | 0.7708 | 4.9015 | 7.7391 | 0.5407 |

---

# Evaluation Metrics

Three main metrics were used.

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and predicted values.

Lower is better.

```text
MAE = average(|actual - predicted|)
```

It provides an intuitive estimate of the average prediction error.

---

## Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

Lower is better.

RMSE penalizes large errors more heavily than MAE, making it useful for detecting substantial prediction mistakes.

---

## R² Score

R² measures how much of the variance in the target variable is explained by the model.

Higher is better.

```text
R² = 1
```

represents a perfect prediction.

A negative R² indicates that the model performs worse than a simple baseline based on predicting the mean of the target.

---

# Final Model Selection

Based on the validation results, the final model selected was:

```text
Random Forest Regressor
+
Temporal Lag Features
```

The decision was based on the fact that Random Forest achieved the strongest overall combination of:

- Lowest MAE
- Lowest RMSE
- Highest R²

for both cities.

The selected model was not chosen simply because Random Forest is a popular algorithm.

It was selected because it demonstrated the strongest empirical performance on this specific dataset under the same validation conditions.

---

# Separate Models for San Juan and Iquitos

A separate model was maintained for each city.

This decision was based on the observed differences between the two datasets.

Final architecture:

```text
San Juan Data
      ↓
San Juan Random Forest
      ↓
San Juan Prediction
```

and:

```text
Iquitos Data
      ↓
Iquitos Random Forest
      ↓
Iquitos Prediction
```

This allows each model to learn city-specific relationships rather than assuming that the same mapping applies equally to both locations.

---

# Feature Importance

After selecting Random Forest, feature importance was analyzed to understand which variables contributed most to the predictions.

## San Juan

Top features:

| Feature | Importance |
|---|---:|
| `cases_lag_1` | 0.907001 |
| `cases_lag_2` | 0.017659 |
| `reanalysis_precip_amt_kg_per_m2` | 0.007468 |
| `cases_lag_4` | 0.006976 |
| `weekofyear` | 0.006078 |
| `station_diur_temp_rng_c` | 0.004732 |
| `cases_lag_12` | 0.003807 |
| `reanalysis_tdtr_k` | 0.003552 |
| `ndvi_sw` | 0.003047 |
| `reanalysis_dew_point_temp_k` | 0.003002 |

The most important feature was `cases_lag_1`, with approximately 90.7% of the model's total feature importance.

This indicates that recent dengue activity was the strongest predictor for San Juan.

---

## Iquitos

Top features:

| Feature | Importance |
|---|---:|
| `cases_lag_1` | 0.518977 |
| `cases_lag_2` | 0.106409 |
| `cases_lag_4` | 0.060222 |
| `ndvi_sw` | 0.028285 |
| `weekofyear` | 0.024375 |
| `reanalysis_air_temp_k` | 0.021550 |
| `ndvi_se` | 0.015752 |
| `reanalysis_max_air_temp_k` | 0.015492 |
| `ndvi_mean` | 0.014495 |
| `station_max_temp_c` | 0.013962 |

Previous dengue observations were still the dominant predictors, but environmental and seasonal features contributed more substantially than they did in San Juan.

---

# Error Analysis

Model evaluation did not stop at MAE, RMSE, and R².

The largest prediction errors were examined to understand when and why the model struggled.

## San Juan

Examples of large errors included:

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 170 | 113.21 | 56.79 |
| 43 | 83.59 | 40.59 |
| 68 | 108.23 | 40.23 |
| 71 | 107.04 | 36.04 |
| 72 | 37.71 | 34.29 |

The largest errors were associated with unusual or extreme disease activity.

---

## Iquitos

Examples of large errors included:

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 63 | 28.45 | 34.55 |
| 16 | 42.65 | 26.65 |
| 29 | 8.12 | 20.88 |
| 50 | 32.38 | 17.62 |
| 35 | 20.76 | 14.24 |

Again, the model struggled most when the observed number of cases deviated substantially from the normal temporal pattern.

---

# Residual Analysis

Residual and actual-vs-predicted plots were used to visually inspect model behavior.

The analysis showed that:

- The model captures the general temporal behavior of dengue cases.
- Predictions are generally more accurate during normal case levels.
- Errors increase around sudden increases and decreases.
- Extreme outbreak peaks are particularly difficult to predict.
- The model tends to struggle when the observed cases move far outside the recent historical pattern.

This indicates that the final model is useful for capturing general disease dynamics but should not be interpreted as perfectly predicting sudden epidemic spikes.

---

# Final Training

After completing model comparison, feature analysis, and error analysis, the final Random Forest models were retrained using the complete available training data for each city.

Final datasets:

```text
San Juan: 924 observations
Iquitos: 508 observations
```

Each final dataset contained:

```text
28 predictive features
```

The final models were trained independently:

```text
San Juan → RandomForestRegressor
Iquitos  → RandomForestRegressor
```

---

# Model Persistence

The final models were saved using `joblib` so that they could be reused without retraining.

The saved models include the trained preprocessing and prediction components required by the application.

Example model files:

```text
models/
├── random_forest_san_juan_final.joblib
└── random_forest_iquitos_final.joblib
```

---

# Model Loading and Verification

After saving, both models were loaded again successfully.

The loaded objects were confirmed as:

```text
San Juan model: RandomForestRegressor
Iquitos model: RandomForestRegressor
```

Test predictions were successfully generated from the loaded models.

This verification confirms that the saved models can be reused for inference independently from the training notebook.

---

# Final Validation Results

The selected Random Forest models achieved the following validation performance:

| City | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| San Juan | **7.3211** | **11.0796** | **0.8737** |
| Iquitos | **4.3499** | **6.8189** | **0.6434** |

These results demonstrate a substantial improvement compared with the initial models that did not include temporal lag information.

---

# Project Workflow

The complete development process followed this sequence:

```text
1. Dataset Investigation
        ↓
2. Data Cleaning
        ↓
3. Missing Value Analysis
        ↓
4. City-Level Separation
        ↓
5. Chronological Ordering
        ↓
6. Time-Based Train/Validation Split
        ↓
7. Initial Machine Learning Models
        ↓
8. Baseline Evaluation
        ↓
9. Identify Temporal Dependency
        ↓
10. Temporal Lag Feature Engineering
        ↓
11. Retrain Multiple Models
        ↓
12. Compare MAE / RMSE / R²
        ↓
13. Select Random Forest
        ↓
14. Feature Importance Analysis
        ↓
15. Error and Residual Analysis
        ↓
16. Final Model Training
        ↓
17. Model Saving
        ↓
18. Model Loading and Verification
        ↓
19. Application / Deployment Preparation
```

---

# Project Structure

The project is organized around the data science workflow and the final deployment pipeline.

```text
DengAI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest_san_juan_final.joblib
│   └── random_forest_iquitos_final.joblib
│
├── notebooks/
│   └── DengAI_Modeling.ipynb
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

The exact structure may evolve as the Streamlit deployment layer is completed.

---

# Technologies

## Programming Language

- Python

## Data Processing

- pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- scikit-learn
- XGBoost

## Model Persistence

- joblib

## Deployment

- Streamlit

---

# Limitations

Although the final model provides strong validation performance, several limitations remain.

## 1. Extreme Outbreaks

The largest errors occur during unusually high or rapidly changing dengue activity.

The model is less reliable when case counts move far outside recent historical patterns.

---

## 2. Dependence on Historical Cases

The strong importance of lag features means that the model depends heavily on previous dengue observations.

This is useful for short-term forecasting but also means prediction quality may decrease when reliable historical observations are unavailable.

---

## 3. City-Specific Behavior

Separate models were necessary because San Juan and Iquitos exhibit different patterns.

This improves specialization but means the system is not a single universal dengue model.

---

## 4. Feature Availability

The model relies on a specific set of weather, environmental, and historical features.

For deployment, these features must be available and prepared in exactly the same format as during training.

---

# Future Improvements

Several directions could improve the system further.

## Temporal Modeling

Potential approaches include:

- Rolling-window statistics
- Moving averages
- Exponentially weighted features
- More seasonal features
- Additional lag intervals
- Autoregressive models
- Time-series-specific models

---

## Advanced Machine Learning

Potential future experiments include:

- LightGBM
- CatBoost
- Stacking
- Blending
- Advanced ensemble methods
- Hyperparameter optimization

However, any future model should be evaluated using the same time-aware validation methodology.

---

## Additional Data

Future versions could incorporate:

- Mosquito population indicators
- Public health interventions
- Population density
- Historical outbreak information
- Additional climate variables
- Geographic information
- Socioeconomic variables
- Real-time environmental data

These additional signals may help the model recognize unusual outbreak conditions.

---

# Deployment

The trained models are prepared for integration into a Streamlit application.

The planned application will allow users to provide the required input information and receive a dengue case prediction.

The application will select the appropriate city-specific model:

```text
User Input
    ↓
City Selection
    ↓
City-Specific Preprocessing
    ↓
San Juan Model / Iquitos Model
    ↓
Predicted Dengue Cases
```

The final deployment layer is intended to make the trained machine learning models accessible through a simple interactive interface.

---

# Reproducibility

To reproduce the project:

```bash
git clone <repository-url>
cd DengAI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook to reproduce the analysis and model development process.

For the Streamlit application:

```bash
streamlit run app.py
```

The final saved models can be loaded directly for inference without repeating the entire training process.

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for dengue fever forecasting.

The development process began with understanding the dataset and identifying the differences between San Juan and Iquitos. Initial machine learning experiments using weather and environmental variables alone produced limited results.

The most important improvement came from recognizing the temporal nature of dengue cases and introducing historical lag features.

The resulting comparison showed that:

- Temporal information was substantially more predictive than relying only on environmental variables.
- More complex boosting algorithms did not automatically outperform Random Forest.
- Random Forest with temporal lag features provided the strongest overall validation performance.
- Separate models were more appropriate for San Juan and Iquitos.
- Previous dengue case counts were the most influential predictors.
- The main remaining challenge was predicting extreme outbreak peaks.

The final solution consists of two independently trained Random Forest regression models with temporal lag features:

```text
San Juan → Random Forest + Temporal Features
Iquitos  → Random Forest + Temporal Features
```

with validation performance of:

```text
San Juan
MAE  = 7.3211
RMSE = 11.0796
R²   = 0.8737

Iquitos
MAE  = 4.3499
RMSE = 6.8189
R²   = 0.6434
```

The final models were retrained on the available training data, persisted using `joblib`, successfully reloaded, and verified through inference.

The next stage of the project is to expose the trained models through a reproducible Streamlit application and complete the deployment pipeline.
