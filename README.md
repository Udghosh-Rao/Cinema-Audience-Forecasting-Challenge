Cinema Audience Forecasting
Time-series forecasting project to predict daily theater attendance using booking data from BookNow and CinePOS platforms.

Kaggle Competition: Cinema Audience Forecasting Challenge

Results
Best Model: XGBoost (R² = 0.643, RMSE = 19.63)

Dataset: 214K records, 826 theaters, 14 months

Features: 47 engineered features from 5 raw columns

Problem
Predict daily audience count for each theater to help with:

Staff scheduling

Inventory planning (concessions)

Show time optimization

Revenue forecasting

Dataset
8 files, 102 MB total. Data from Jan 2023 - Feb 2024, predicting March 2024 onwards.

File	Records	What's in it
booknow_visits.csv	214K	Daily audience counts (target)
booknow_booking.csv	~450K	Online bookings
cinePOS_booking.csv	~380K	On-site bookings
booknow_theaters.csv	826	Theater info (type, location)
cinePOS_theaters.csv	~1K	CinePOS theater data
movie_theater_id_relation.csv	~800	Maps theater IDs between platforms
date_info.csv	428	Calendar data
sample_submission.csv	38K	Submission format
Approach
1. EDA
Audience mean: 41.62, median: 34 (right-skewed)

Weekend effect: Saturday 45% higher than weekdays

High variance (std = 32.83), outliers at 2.61%

2. Feature Engineering (47 features)
Lag features: 1, 7, 14, 28 days back

Rolling stats: Mean/std for 7, 14, 30-day windows

EWMA: Exponential weighted moving averages

Trends: Short-term momentum indicators

Booking aggregations: Daily sum, mean, count per theater

Temporal: Month, day of week, weekend flag

Categorical: Theater type, area (label encoded)

3. Preprocessing
Label encoding for categorical variables

KNN imputation for missing values

StandardScaler normalization

4. Models Tested
Model	R²	RMSE	Notes
Ridge	0.509	23.01	Baseline
LightGBM	0.580	21.28	Fast training
XGBoost	0.643	19.63	Best
Ensemble	0.610	20.52	Weighted avg (15-35-50%)
XGBoost performed best. Ensemble didn't improve over XGBoost alone.

5. Feature Importance (Top 5)
lag_1 (previous day)

roll_mean_7 (7-day average)

ewm_7 (exponential moving avg)

lag_7 (week ago)

day_of_week

Files
text
├── data/
│   ├── raw/              # Kaggle data
│   └── processed/        # Cleaned data
├── notebooks/
│   └── main_notebook.ipynb
├── models/
│   └── xgb_model.pkl
├── index.html            # Project showcase
└── README.md
Setup
bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn

# Download data from Kaggle
# https://www.kaggle.com/competitions/Cinema_Audience_Forecasting_challenge/data

# Run notebook
jupyter notebook notebooks/main_notebook.ipynb
Tech Stack
Python 3.11 | Pandas | NumPy | Scikit-learn | XGBoost | LightGBM | Matplotlib | Seaborn

Key Takeaways
Temporal features (lags, rolling averages) were most important

Weekend patterns are strong predictors

XGBoost handled the non-linear relationships well

26% improvement over baseline Ridge model

Future Work
Add external features (weather, holidays)

Try LSTM/Transformer models

Include movie metadata (genre, ratings)

Incorporate advance booking velocity

Competition Link: https://www.kaggle.com/competitions/Cinema_Audience_Forecasting_challeng
