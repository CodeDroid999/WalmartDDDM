# Step 1: Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Ignore warnings
warnings.filterwarnings("ignore")

# Step 2: Load the dataset
data = pd.read_csv('CustomerLoyaltyCardData.csv')

# Step 3: Prepare the data for classification
X = data.drop('Gender', axis=1)  # Replace 'target_variable' with the actual column name of the target variable
y = data['Age']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Perform Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
linear_reg_pred = linear_reg.predict(X_test_scaled)

print("Linear Regression Results:")
print("Coefficients:", linear_reg.coef_)
print("Intercept:", linear_reg.intercept_)

# Scatter plot of original values vs. predicted values
plt.scatter(y_test, linear_reg_pred)
plt.xlabel("Original Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Original Values vs. Predicted Values")
plt.show()

# Step 7: Perform Logistic Regression
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train_scaled, y_train)
logistic_reg_pred = logistic_reg.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print("Coefficients:", logistic_reg.coef_)
print("Intercept:", logistic_reg.intercept_)
# Create a confusion matrix
cm = confusion_matrix(y_test, logistic_reg_pred)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Logistic Regression: Confusion Matrix")
plt.show()

# Step 8: Perform Time Series Forecasting (ARIMA)
time_series_data = data["Age"]  # Replace 'target_variable' with the actual column name of the time series data
arima_model = ARIMA(time_series_data, order=(1, 1, 1))  # Example order
arima_model_fit = arima_model.fit()

num_steps = 10  # Specify the number of steps to forecast
arima_forecast = arima_model_fit.forecast(steps=num_steps)

forecasted_values = arima_forecast.tolist()

print("\nARIMA Forecast Results:")
print("Forecasted Values:", forecasted_values)
# Assuming you have already fitted an ARIMA model and obtained the forecasted values
forecasted_values = arima_forecast[0]

# Assuming you have the original time series data
original_values = time_series_data.values

# Assuming you have specified the number of steps to forecast
num_steps = 10
forecasted_time = range(len(original_values), len(original_values) + num_steps)

# Plotting the original values
plt.plot(time, original_values, label='Original')

# Plotting the forecasted values
plt.plot(forecasted_time, forecasted_values, label='Forecast')

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()