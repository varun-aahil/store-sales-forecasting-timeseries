# Store Sales Forecasting - Time Series Regression

**Dataset Link** - [Kaggle Store Sales (Corporación Favorita)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Predicting future sales isn't just about throwing a standard machine learning model at a CSV and hoping for the best. Because this is time series data, messing up the timeline means massive data leakage. This project is a complete regression pipeline built from scratch to forecast daily store sales based on actual historical patterns.

## The Build Process

Handling millions of rows of retail data on a standard setup will easily crash your RAM or leave you stuck in endless training loops. Here is how I structured the pipeline so it actually runs efficiently:

* **Feature Engineering:** Raw dates are basically useless to a model on their own. I extracted the actual signals—`year`, `month`, `day`, `dayofweek`, and `is_weekend`—to catch cyclical trends, monthly habits, and weekend shopping spikes. 
* **Categorical Optimization:** The product `family` column is entirely text. Instead of blowing up my memory with a massive One-Hot Encoded matrix, I used **Label Encoding** to map those categories into clean, lightweight integers.
* **Time-Aware Splitting:** You can't just randomly shuffle time series data. I sorted everything strictly by date and did a sequential train-test split (`shuffle=False`). The model learns from the past and gets tested on the future—no cheating.
* **Performance & Scaling:** To stop Colab from grinding to a halt on millions of rows, I took a chronological subset of the data and forced the model to use all available CPU cores (`n_jobs=-1`) to slash training times.

## Does it actually work?

Yes. I powered the pipeline with a `RandomForestRegressor` and evaluated it using Root Mean Squared Error (RMSE). 

In the training subset, the average daily sales hovered around 488 units. The model hit an RMSE of ~295. It establishes a solid baseline that easily beats random guessing and keeps the error right on the scale of actual real-world sales.

## Tech Stack

* Python
* Pandas & NumPy 
* Scikit-Learn (for Label Encoding, sequential splitting, Random Forest regression, and RMSE evaluation)
