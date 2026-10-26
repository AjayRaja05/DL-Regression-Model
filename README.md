# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1 : Generate Dataset
# -------------------------------

np.random.seed(42)
torch.manual_seed(42)

x = np.arange(1, 51)
noise = np.random.randn(50) * 5
y = 3 * x + 10 + noise

# Convert to tensors
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# -------------------------------
# Display Dataset
# -------------------------------

print("Generated Dataset")
print("-------------------------------")
print(" X\tY")

for i in range(len(x)):
    print(f"{x[i]}\t{y[i]:.2f}")

plt.figure(figsize=(6,4))
plt.scatter(x, y, color="blue")
plt.title("Generated Dataset")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# -------------------------------
# STEP 2 : Define Neural Network
# -------------------------------

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# -------------------------------
# STEP 3 : Initialize Model
# -------------------------------

model = Model(1, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

# -------------------------------
# STEP 4 : Train Model
# -------------------------------

epochs = 100

losses = []

for epoch in range(epochs):

    prediction = model(x_train)

    loss = criterion(prediction, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}  Loss = {loss.item():.4f}")

# -------------------------------
# STEP 5 : Plot Loss Curve
# -------------------------------

plt.figure(figsize=(6,4))
plt.plot(losses, color="red")
plt.title("Training Loss Vs Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------
# STEP 6 : Best Fit Line
# -------------------------------

predicted = model(x_train).detach().numpy()

plt.figure(figsize=(6,4))
plt.scatter(x, y, label="Actual Data")
plt.plot(x, predicted, color="red", linewidth=2, label="Best Fit Line")
plt.title("Best Fit Line")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# STEP 7 : Predict New Sample
# -------------------------------

sample = torch.tensor([[60.0]])

prediction = model(sample)

print("\n-------------------------------")
print("New Sample Prediction")
print("-------------------------------")
print("Input :", sample.item())
print("Predicted Output :", prediction.item())

# -------------------------------
# Final Learned Parameters
# -------------------------------

print("\nLearned Weight :", model.linear.weight.item())
print("Learned Bias   :", model.linear.bias.item())

```

### Dataset Information
<img width="506" height="362" alt="image" src="https://github.com/user-attachments/assets/3d52c1a1-7c8d-45cb-902b-5d5fc82ca742" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="512" height="365" alt="image" src="https://github.com/user-attachments/assets/7ab9cd04-a39a-4c2e-b2be-0acaec155787" />

Best Fit line plot
<img width="506" height="365" alt="image" src="https://github.com/user-attachments/assets/59d86302-a195-4a28-ab77-8b0c7a6abad3" />

### New Sample Data Prediction
Input : 60.0
Predicted Output : 193.8988037109375

Learned Weight : 3.2091197967529297
Learned Bias   : 1.3516194820404053

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
