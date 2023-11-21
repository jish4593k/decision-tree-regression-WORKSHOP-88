import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Add more features to the dataset
# For example, let's duplicate the existing feature to demonstrate
X = np.column_stack((X, X[:, 0]))

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple decision tree regression model using PyTorch
class DecisionTreeRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(DecisionTreeRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model, loss function, and optimizer
model = DecisionTreeRegressionModel(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)

    # Compute the loss
    loss = criterion(y_pred, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert X to PyTorch tensor for plotting
X_tensor_plot = torch.tensor(np.arange(min(X[:, 0]), max(X[:, 0]), 0.1).reshape(-1, 1), dtype=torch.float32)

# Add more features for plotting
X_tensor_plot = torch.cat((X_tensor_plot, X_tensor_plot), 1)

# Predicting the salary for the plot
y_pred_plot = model(X_tensor_plot)

# Convert predictions back to numpy arrays
X_plot_np = X_tensor_plot.detach().numpy()
y_pred_plot_np = y_pred_plot.detach().numpy()

# Visualizing the Decision Tree Regression Results in High resolution
plt.scatter(X[:, 0], y, color='red')
plt.plot(X_plot_np[:, 0], y_pred_plot_np, color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
