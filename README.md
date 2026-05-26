# Developing a Recurrent Neural Network (RNN) Model for Stock Prediction

##  Aim
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

---

## 📖 Theory

Recurrent Neural Networks (RNNs) are a class of neural networks designed for processing sequential data. Unlike traditional feedforward neural networks, RNNs feature internal loops that allow information to persist. This memory mechanism makes them exceptionally well-suited for time-series forecasting, such as stock price prediction, where future values highly depend on historical trends.



In this implementation:
* **Input Sequence:** A window of 60 consecutive historical closing prices ($x_t, x_{t-1}, \dots, x_{t-59}$).
* **Hidden State:** The network maintains a hidden state that updates at each time step to retain temporal dependencies.
* **Fully Connected Layer:** Maps the final hidden state to a single continuous output value representing the predicted price of the next day ($y_{t+1}$).

---

## 🛠️ Design Steps

* **STEP 1:** Load and normalize data using `MinMaxScaler`, and partition it into structured overlapping sequences.
* **STEP 2:** Convert numpy arrays into PyTorch tensors and build a `DataLoader` for efficient batch processing.
* **STEP 3:** Define the custom `RNNModel` architecture subclassing `nn.Module`.
* **STEP 4:** Generate a model summary, define the Mean Squared Error (`MSELoss`) criterion, and instantiate the `Adam` optimizer.
* **STEP 5:** Train the model over multiple epochs, track the loss, and plot the training curve.
* **STEP 6:** Evaluate the model on unseen test data, reverse the scaling transformation, and plot the actual vs. predicted prices.

---

## 💻 Program

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Display model summary
print("Model Architecture Summary:")
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

train_model(model, train_loader, criterion, optimizer)

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price', color='blue')
plt.plot(predicted_prices, label='Predicted Price', color='red', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.grid(True)
plt.show()

print(f'\nFinal Step Evaluation:')
print(f'Predicted Price: {predicted_prices[-1][0]:.2f}')
print(f'Actual Price: {actual_prices[-1][0]:.2f}')
```
# OUTPUT
<img width="244" height="802" alt="image" src="https://github.com/user-attachments/assets/d1f33462-b26c-432f-94f8-584155ec03d0" />
<img width="902" height="569" alt="image" src="https://github.com/user-attachments/assets/35102f29-2674-465c-8a7b-e7a21d00a799" />
<img width="1244" height="734" alt="image" src="https://github.com/user-attachments/assets/9c4c301f-697b-4f5a-b37a-5dc181d24490" />

# RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
