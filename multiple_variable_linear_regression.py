# Notes that this is a poor model, even though the code is right as per linear regression it isn't sufficient to make good predictions
# solution can be to use polynomial regression and feature engineering
# this is an clear example of under fitting meaning too less features and that too of degree 1 or very less data
# can be solved with regularization 


# check model_degree_selection soolves the same dataset but works faster and is made my scikit_learn





# Import necessary libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV
features = []
targets = []

with open('multiple_variable_house_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        features.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        targets.append(float(row[4]))

X = np.array(features)
y = np.array(targets)

# Normalize X
def z_normalize_train(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    Z = (X - mean) / std
    return Z, mean, std

# Normalize y
def z_normalize_y(y):
    mean = np.mean(y)
    std = np.std(y, ddof=1)
    z = (y - mean) / std
    return z, mean, std

X_norm, X_mean, X_std = z_normalize_train(X)
y_norm, y_mean, y_std = z_normalize_y(y)

# Initialize parameters
w = np.zeros(X.shape[1])
b = 0

# Cost function
def cost_function(X, y, w, b):
    m = X.shape[0]
    total = 0
    for i in range(m):
        y_hat = np.dot(X[i], w) + b
        total += (y_hat - y[i]) ** 2
    return total / (2 * m)

# Gradients
def gradients(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        y_hat = np.dot(X[i], w) + b
        error = y_hat - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    return dj_dw / m, dj_db / m

# Gradient descent
def gradient_decent(X, y, w, b, alpha, iterations):
    for _ in range(iterations):
        dj_dw, dj_db = gradients(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

# Train the model
alpha = 0.0000001
iterations = 10000
w, b = gradient_decent(X_norm, y_norm, w, b, alpha, iterations)

# Normalize new input
def z_normalize_input(x_new, means, stds):
    return (x_new - means) / stds

# Take user input
while(True):
    input_str = input("Enter features (space-separated): size sqft, bedrooms, floors, age\n")
    if (input_str == "-1"):
        break
    else:
        x_input = np.array(list(map(float, input_str.split())))
        x_input_norm = z_normalize_input(x_input, X_mean, X_std)

        # Predict
        y_pred_norm = np.dot(x_input_norm, w) + b
        y_pred = y_pred_norm * y_std + y_mean

        print(f"\nPredicted house price: ${y_pred * 1000:.2f}")

# ----------------- Visualization -----------------

# # Plot using Size (feature 0) and Bedrooms (feature 1)
# x_feature = 0
# y_feature = 1

# # Actual values
# X_size = X[:, x_feature]
# X_bedrooms = X[:, y_feature]

# # Predicted prices for the training set
# y_preds = np.dot(X_norm, w) + b
# y_preds = y_preds * y_std + y_mean

# # Create prediction surface
# x_surf, y_surf = np.meshgrid(
#     np.linspace(min(X_size), max(X_size), 10),
#     np.linspace(min(X_bedrooms), max(X_bedrooms), 10)
# )
# z_surf = np.zeros_like(x_surf)

# for i in range(x_surf.shape[0]):
#     for j in range(x_surf.shape[1]):
#         x_temp = np.array([0, 0, 0, 0], dtype=float)
#         x_temp[x_feature] = x_surf[i, j]
#         x_temp[y_feature] = y_surf[i, j]
#         x_temp_norm = z_normalize_input(x_temp, X_mean, X_std)
#         z_surf[i, j] = np.dot(x_temp_norm, w) + b
# z_surf = z_surf * y_std + y_mean

# # Plot 3D graph
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Training data
# ax.scatter(X_size, X_bedrooms, y, color='blue', label='Training Data')

# # Regression surface
# ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')

# # User input point
# ax.scatter(x_input[x_feature], x_input[y_feature], y_pred, color='green', s=100, label='Prediction Point')

# # Labels
# ax.set_xlabel("Size (sqft)")
# ax.set_ylabel("Bedrooms")
# ax.set_zlabel("Price ($1000s)")
# ax.set_title("Linear Regression Fit on House Data")
# ax.legend()

# plt.show()

feature_names = ["size(sqft)", "bedrooms", "floors", "age"]
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

y_preds = np.dot(X_norm, w) + b
y_preds = y_preds * y_std + y_mean

for i in range(4):
    ax = axes[i]
    ax.scatter(X[:, i], y, color='dodgerblue', label='target')
    ax.scatter(X[:, i], y_preds, color='orange', label='predict', alpha=0.6)
    ax.set_xlabel(feature_names[i])
    if i == 0:
        ax.set_ylabel("Price")
    ax.set_title(feature_names[i])

axes[0].legend()
plt.tight_layout()
plt.show()
