## Temperature Prediction of Cities using Random Forest Regression

This Python script utilizes the Random Forest Regression algorithm to predict the mean temperature for a specific date in multiple cities. The script performs the following steps:

1. Imports the necessary libraries, including NumPy, matplotlib, and pandas, for data manipulation and visualization, as well as scikit-learn for machine learning tasks.

2. Loads the temperature dataset for cities (Chennai, Delhi, Mumbai, Kolkata) from separate CSV files. Ensure that the CSV files are stored in the correct directory and update the file paths if necessary.

3. Prompts the user to input a specific date (YYYY-MM-DD) for temperature prediction.

4. Preprocesses the dataset by dropping missing values, converting date columns to appropriate date-time format, and normalizing the feature values using StandardScaler.

5. Splits the dataset into training and testing sets for model evaluation.

6. Performs hyperparameter tuning using GridSearchCV to find the best parameters for the Random Forest Regressor.

7. Trains a Random Forest Regressor model with the best parameters obtained from hyperparameter tuning.

8. Predicts the mean temperature for the input date in each city using the trained model.

9. Converts the predicted temperature from Fahrenheit to Celsius for better understanding.

10. Prints the predicted mean temperature for the input date in each city.

To use the code, make sure to have the necessary dependencies installed and update the file paths for the dataset if needed. Run the script and provide a valid input date to obtain the predicted mean temperature for each city.
