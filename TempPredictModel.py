import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Importing the Dataset
cities = ["Chennai", "Delhi", "Mumbai", "Kolkata"]
datasets = {}

# # Initialize lists to store model performance
# r2_scores = []
# mean_squared_errors = []
# mean_absolute_errors = []

for city in cities:
    dataset = pd.read_csv(f"E:\\TempPredict\\TempData\\{city}.csv")
    datasets[city] = dataset

# User Input for Date
input_date = input("Enter the date (YYYY-MM-DD): ")

# Predict Temperature for Input Date in each City
for city in cities:
    dataset = datasets[city]

    # Data Preprocessing
    dataset.dropna(inplace=True)
    dataset['YEAR'] = pd.to_datetime(dataset['YEAR']).dt.year
    dataset['MONTH'] = pd.to_datetime(dataset['MONTH']).dt.month
    dataset['DAY'] = pd.to_datetime(dataset['DAY']).dt.day

    # Feature Selection
    features = dataset[['YEAR', 'MONTH', 'DAY']].values
    target = dataset['TEMPERATURE'].values

    # Data Normalization
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

    # Hyperparameter Tuning
    param_grid = {'n_estimators': [100, 500, 1000],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    # Fitting Model to the Training set (Random Forest Regression)
    regressor = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                      max_depth=best_params['max_depth'],
                                      min_samples_split=best_params['min_samples_split'],
                                      random_state=0)
    regressor.fit(x_train, y_train)

    # Predict Temperature for Input Date
    input_year, input_month, input_day = map(int, input_date.split('-'))
    input_features = np.array([[input_year, input_month, input_day]])
    input_features = scaler.transform(input_features)
    predicted_temperature = regressor.predict(input_features)[0]

    predicted_temperature=(predicted_temperature-32)*5/9

    # # Evaluate Model Performance on Test Set
    # y_pred = regressor.predict(x_test)
    # r2 = r2_score(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)

    # # Append model performance metrics to lists
    # r2_scores.append(r2)
    # mean_squared_errors.append(mse)
    # mean_absolute_errors.append(mae)

    # Print Predicted Temperature for Input Date
    print(f"Predicted Mean Temperature for {city} on {input_date}: {predicted_temperature:.2f}°C")

# # Plot Model Performance
# plt.figure(figsize=(10, 5))
# x = np.arange(len(cities))
# width = 0.3

# plt.bar(x, r2_scores, width, label='R-squared')
# plt.bar(x + width, mean_squared_errors, width, label='Mean Squared Error')
# plt.bar(x + 2 * width, mean_absolute_errors, width, label='Mean Absolute Error')

# plt.xlabel('City')
# plt.ylabel('Performance Metrics')
# plt.title('Model Performance for Temperature Prediction')
# plt.xticks(x + width, cities)
# plt.legend()
# plt.tight_layout()
# plt.show()
