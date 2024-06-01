import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

# Load the data
data = pd.read_excel('C:/Users/ABDURAHMAN BEY/PycharmProjects/pythonProject8/climatechangedata(1).xlsx')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data = data.set_index('Year')

# Selecting the variables for analysis
variables = [
    'Carbon Dioxide (Million metric tons of CO2 equivalent)',
    'Methane (Million metric tons of CO2 equivalent)',
    'Nitrous Oxide (Million metric tons of CO2 equivalent)',
    'Fluorinated Gases (Million metric tons of CO2 equivalent)',
    'Total GHG (Million metric tons of CO2 equivalent)',
    'Temperature (Celcius)',
    'Forest Area (%)'
]

# Plotting historical data and forecasting for each variable
for variable in variables:
    train_data = data[(data.index.year <= 2023) & (data.index.year >= 1990)][variable]
    test_data = data[(data.index.year >= 2018) & (data.index.year <= 2023)][variable]

    print(f"Variable: {variable}")
    print(f"Length of train_data: {len(train_data)}")
    print(f"Length of test_data: {len(test_data)}")

    if len(train_data) == 0:
        print(f"Insufficient data for {variable}. Skipping...")
        continue

    # ACF and PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ACF plot
    plot_acf(train_data, lags=20, ax=ax1)  # Adjust lags as needed
    ax1.set_title(f'Autocorrelation Function (ACF) of {variable}')

    # PACF plot
    plot_pacf(train_data, lags=6, ax=ax2)  # Adjust lags as needed
    ax2.set_title(f'Partial Autocorrelation Function (PACF) of {variable}')
    plt.tight_layout()
    plt.show()

    # Manual seasonal differencing term selection
    D = 1  # You can adjust this value based on the visual inspection of the ACF plot

    # Automatic parameter selection using PACF and ACF
    stepwise_model = pm.auto_arima(train_data, start_p=1, start_q=1,
                                   max_p=5, max_q=5, m=12,
                                   seasonal=True, D=D,
                                   d=None, trace=False,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)

    # Summary of the model
    print(stepwise_model.summary())
    #alper

    # Forecasting
    forecast_steps = 2056 - 2024 + 1  # Predict until 2056
    forecast = stepwise_model.predict(n_periods=forecast_steps)
    print("Predicted values:")
    for i, val in enumerate(forecast, start=2024):
        print(f"Year {i}: {val}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Historical Data', marker='o')
    plt.plot(test_data.index, test_data, label='Testing Data', marker='o')
    plt.plot(pd.date_range(start='2018', end='2023', freq='YS'), test_data.values, linestyle='--', color='green', label='Actual Testing Data')
    plt.plot(pd.date_range(start='2024', periods=len(forecast), freq='YS'), forecast, label='Forecast', marker='o', color='red')
    plt.title(f'Time Series Analysis of {variable}')
    plt.xlabel('Year')
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Model evaluation
    if len(test_data) > 0:
        # Calculate metrics
        mae = mean_absolute_error(test_data, forecast[:len(test_data)])
        mse = mean_squared_error(test_data, forecast[:len(test_data)])
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
