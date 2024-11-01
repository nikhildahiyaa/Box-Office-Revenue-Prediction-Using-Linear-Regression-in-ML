
# Box Office Revenue Prediction Using Machine Learning

This project aims to predict domestic box office revenue for movies based on a variety of features including distributor, genre, MPAA rating, budget, and more. Using the XGBoost model and hyperparameter tuning, the project demonstrates methods for data preprocessing, feature engineering, and model training to achieve optimal predictions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

The primary objective of this project is to build a regression model that accurately predicts the domestic box office revenue for movies. We utilize the XGBoost model, known for its performance in regression tasks, and employ techniques like hyperparameter tuning to improve model performance.

## Dataset

The dataset includes 2,694 movies with the following features:

- `title`: Title of the movie
- `domestic_revenue`: Domestic box office revenue (target variable)
- `world_revenue`: Worldwide box office revenue
- `distributor`: Distributor of the movie
- `opening_revenue`: Revenue on the opening day
- `opening_theaters`: Number of theaters at the opening
- `budget`: Budget of the movie
- `MPAA`: MPAA rating of the movie
- `genres`: Genre(s) of the movie
- `release_days`: Days since release

**Note**: Some features like `world_revenue` and `opening_revenue` were dropped to focus on domestic revenue prediction.

## Installation

To run this project, make sure you have Python and the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

On macOS, you may also need to install `libomp` for XGBoost:

```bash
brew install libomp
```

## Data Preprocessing

### Steps:

1. **Feature Selection**: Non-essential columns were removed.
2. **Handling Missing Values**: Replaced missing values with mean or mode where appropriate.
3. **Encoding Categorical Variables**: Used label encoding for `distributor`, `MPAA`, and `genres`.
4. **Log Transformation**: Applied to skewed features like `domestic_revenue` and `opening_theaters` to normalize the data.

## Exploratory Data Analysis (EDA)

Several plots were generated to understand the data distribution and identify patterns:

- **MPAA Rating Distribution**: Showed that PG and R-rated movies are predominant, which could influence revenue trends.
- **Revenue Distributions**: Highlighted skewness in revenue, where a small number of high-revenue movies create outliers.
- **Genre Encoding**: Vectorized the `genres` column to enable machine learning models to interpret genre influence.
- **Correlation Analysis**: Identified correlations among features, such as the impact of budget and number of theaters on revenue.

## Modeling

The XGBoost model was initially trained and evaluated with default parameters:

- **Training Mean Absolute Error (MAE)**: ~0.238
- **Validation MAE**: ~0.713

The higher validation error indicated slight overfitting, suggesting that tuning could improve model performance.

## Hyperparameter Tuning

We used `RandomizedSearchCV` to tune the XGBoost model parameters, leading to the following optimal values:

```python
{
    'subsample': 0.8,
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.042,
    'colsample_bytree': 0.7
}
```

### Results after Tuning

After tuning, the model achieved:

- **Training MAE**: ~64,903,177
- **Validation MAE**: ~74,850,620

The increase in MAE values (in absolute terms) reflects real revenue units, but the gap between training and validation MAE remains small, indicating reduced overfitting and better generalization.

## Conclusion

This project successfully built a predictive model for movie box office revenue, achieving reasonable accuracy through careful data preprocessing and hyperparameter tuning. Future work could include additional feature engineering, such as capturing seasonal effects or specific genre popularity, to further enhance predictive power.

## License

This project is licensed under the MIT License.
