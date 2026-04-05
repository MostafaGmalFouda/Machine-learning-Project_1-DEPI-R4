# Predicting Boston Housing Prices

## Overview
This project aims to build a predictive model for estimating housing prices in the Boston area using the **Boston Housing Dataset**. It covers:
- Data exploration and visualization
- Feature analysis
- Model development and evaluation
- Hyperparameter tuning with Grid Search and cross-validation
- Making predictions for new client data
- Saving the trained model for future use

## Dataset
The `housing.csv` dataset contains 506 entries with 14 attributes:

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centers |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
| LSTAT | % lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000s (target variable) |

## Project Steps

### 1. Data Exploration
- Loaded dataset using `pandas`
- Checked data types, missing values, and descriptive statistics
- Separated features (`RM`, `LSTAT`, `PTRATIO`) and target (`MEDV`)
- Visualized relationships using regression plots

### 2. Feature Observation
- **RM (Average Rooms):** More rooms → higher price  
- **LSTAT (% Lower Class):** Higher LSTAT → lower price  
- **PTRATIO (Student-Teacher Ratio):** Higher PTRATIO → lower price  

### 3. Model Development
- Performance metric: **R² score**
- Split dataset: 80% training, 20% testing
- Visualized **learning curves** and **complexity curves**
- Determined optimal depth for Decision Tree Regressor

### 4. Model Optimization
- Used `GridSearchCV` with `ShuffleSplit` cross-validation
- Scoring function: R² metric
- Optimal `max_depth = 6`

### 5. Predictions
Sample client predictions:

| Client | Features (RM, LSTAT, PTRATIO) | Predicted Price ($) |
|--------|-------------------------------|------------------|
| 1 | [5, 34, 15] | 302,400 |
| 2 | [4, 55, 22] | 284,200 |
| 3 | [8, 7, 12]  | 933,975 |

### 6. Saving the Model
- Saved trained Decision Tree Regressor using `pickle` as `boston_housing_model.pkl`

## Libraries & Tools
- Python 3.x  
- `numpy`, `pandas`, `matplotlib`, `seaborn`  
- `scikit-learn` (`DecisionTreeRegressor`, `GridSearchCV`, `ShuffleSplit`)  
- `pickle` for saving models  

## How to Run
1. Clone or download the repository  
2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
3. Place `housing.csv` in the same directory
4. Run the Jupyter Notebook `boston_housing_project.ipynb`
5. Check predictions and visualize learning/complexity curves
6. Load the saved model for future predictions:
```bash
import pickle
with open("boston_housing_model.pkl", "rb") as f:
    model = pickle.load(f)
pred = model.predict([[5, 34, 15]])
```
## Conclusion
The project demonstrates building a predictive model for housing prices
Decision Tree Regressor with max_depth = 6 provides the best generalization
Predictions align with feature intuition and real-world expectations
