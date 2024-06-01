import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

# Define the LSTM model
class MultiVariableLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiVariableLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the data
data = pd.read_excel("climatechangedata(1).xlsx")
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
input_size = len(variables)  # Number of input features (7 columns)
hidden_size = 128
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data[variables])
scaled_data_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)

# Initialize LSTM model
lstm_model = MultiVariableLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Train the LSTM model
for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(scaled_data_tensor[:, :-1, :])
    loss = criterion(outputs, scaled_data_tensor[:, 1:, 0])  # Predict next time step
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Forecast using the trained LSTM model
with torch.no_grad():
    lstm_model.eval()
    forecast = lstm_model(scaled_data_tensor[:, :-1, :])
    print(forecast)
    forecast = forecast.cpu().numpy()
    #forecast = scaler.inverse_transform(forecast.reshape(-1, len(variables)))

# Calculate metrics
test_data = data[(data.index.year >= 2016) & (data.index.year <= 2023)][variables].values
mae = mean_absolute_error(test_data, forecast)
mape = mean_absolute_percentage_error(test_data, forecast)
print(f"LSTM - MAE: {mae:.4f}, MAPE: {mape:.4f}%")