# Store Sales Forecasting - Time Series Regression

### Dataset Link - https://www.kaggle.com/competitions/store-sales-time-series-forecasting
Predicting future store sales isn't just about throwing a standard machine learning model at a table. Because it's time series data, handling the timeline incorrectly can completely break your model via data leakage. This project builds a complete regression pipeline from scratch to forecast daily store sales based on historical patterns.

## The Build Process

Handling millions of rows of retail data on a standard setup can easily choke your memory or cause infinite training loops. Here is how I structured the pipeline to keep things efficient and accurate:

- **Feature Engineering:** Raw dates don't tell a model much on their own. I extracted core temporal signals—`year`, `month`, `day`, `dayofweek`, and `is_weekend`—to capture cyclical trends, monthly habits, and weekend shopping spikes. 
- **Categorical Optimization:** The product `family` feature contains text strings. Instead of blowing up the memory with a massive One-Hot Encoding matrix, I used efficient **Label Encoding** to convert categories into clean integers.
- **Time-Aware Splitting:** In time series, you never shuffle data randomly. I sorted the data strictly by date and performed a sequential train-test split (`shuffle=False`) so the model trains on the past and gets evaluated strictly on the future.
- **Performance & Scaling:** To prevent Colab from grinding to a halt on millions of rows, I utilized chronological subsetting and leveraged parallel processing (`n_jobs=-1`) inside the model architecture to slash training times.

## Does it actually work?

Yes. By powering the pipeline with a `RandomForestRegressor` and evaluating performance using **Root Mean Squared Error (RMSE)**, the model successfully captures non-linear interactions across stores and product families while keeping error metrics right on the scale of actual sales units.

## Tech Stack

- Python
- Pandas & NumPy (for data manipulation and feature engineering)
- Scikit-Learn (for Label Encoding, sequential splitting, Random Forest regression, and RMSE evaluation)
