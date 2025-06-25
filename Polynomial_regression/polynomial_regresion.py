# Import libraries 
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data from CSV and perfom "Feature Engineering"
# generate new feature as square of sleep_hours & study_hours and multiplication of the two

features = []
targets = []

with open(r'C:\Users\Aditya Pramod Pawar\Machine Learning\Polynomial_regression\polynomial_regression_data.csv') as file:
    reader = csv.reader(file)
    next(reader) # skip headers
    for row in reader:
        features.append([float(row[0]), float(row[1]), (float(row[0])) ** 2, (float(row[1])) ** 2, (float(row[0])) * (float(row[1]))])
        targets.append(float(row[2]))

# Covert python list to numpy arrays
X = np.array(features)
y = np.array(targets)

# function to Normalize X
def z_normalize_train(X):
    mean = np.mean(X, axis = 0)
    std  = np.std(X, axis = 0, ddof = 1)
    Z = (X - mean) / std
    return Z, mean, std

# function to Normalize y
def z_normalize_y(y):
    mean = np.mean(y)
    std  = np.std(y, ddof = 1)
    Z = (y - mean) / std
    return Z, mean, std

# Normalizing X and y
X_norm, X_mean, X_std = z_normalize_train(X)
y_norm, y_mean, y_std = z_normalize_y(y)

# create parameter numpy array w and parameter variable b
w = np.zeros(X.shape[1])
b = 0

# function to calculate Cost_function or MSE
def cost_function(x, y, w, b):
    m = x.shape[0]
    total = 0
    for i in range(m):
        y_hat = np.dot(x[i], w) + b
        total += (y_hat - y[i]) ** 2
    MSE = total / (2 * m)
    return MSE

# function to calcualte gradient
def gradients(x, y, w, b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        y_hat = np.dot(x[i], w) + b
        error = y_hat - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    return dj_dw / m, dj_db / m

# function to calculate gradient descent
def gradient_descent(X_norm, y_norm, w, b, alpha, iterations):
    for _ in range(iterations):
        if (_ % 1000 == 0):
            print (f"cost function at iteration {_} = {cost_function(X_norm, y_norm, w, b)}")
        dj_dw, dj_db = gradients(X_norm, y_norm, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

# intialize and set alpha and iteration and calling gradient descent
alpha = 1e-2
iterations = 50000

w, b = gradient_descent(X_norm, y_norm, w, b, alpha, iterations)

# function to normalize input & output
def z_normalize_input(x_input, X_mean, X_std):
    Z = (x_input - X_mean) / X_std
    return Z

def denormalize_output(y_pred, y_mean, y_std):
    y = (y_pred * y_std) + y_mean
    return y

# add input for the engineered features
def add_input_feature(prompt):
    # create new list with original two features
    new_features = [prompt[0], prompt[1]]
    new_features.append(prompt[0] ** 2)
    new_features.append(prompt[1] ** 2)
    new_features.append(prompt[0] * prompt[1])
    return new_features

# predictions
print(w)
while(True):
    prompt = (list(map(float, input(" enter space seperated sleep_hours & study_hours:\n").split())))
    if prompt[0] == -1:
        break
    x_input = np.array(add_input_feature(prompt))
    x_input_norm = z_normalize_input(x_input, X_mean, X_std)
    y_pred_norm = np.dot(x_input_norm, w) + b
    y_pred = denormalize_output(y_pred_norm, y_mean, y_std)
    print(f"The score of the student should be {y_pred}")

# -------------------------------------------- Matplotlib Code ----------------------------------------------------   

# Create a grid of sleep_hours and study_hours
sleep_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
study_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
sleep_grid, study_grid = np.meshgrid(sleep_vals, study_vals)

# Flatten and create input features for predictions
sleep_flat = sleep_grid.ravel()
study_flat = study_grid.ravel()

X_vis = []
for i in range(len(sleep_flat)):
    x1 = sleep_flat[i]
    x2 = study_flat[i]
    x1_2 = x1 ** 2
    x2_2 = x2 ** 2
    x1x2 = x1 * x2
    X_vis.append([x1, x2, x1_2, x2_2, x1x2])

X_vis = np.array(X_vis)
X_vis_norm = z_normalize_input(X_vis, X_mean, X_std)
y_pred_norm = np.dot(X_vis_norm, w) + b
y_pred = denormalize_output(y_pred_norm, y_mean, y_std)
y_pred = y_pred.reshape(sleep_grid.shape)

# Plot original data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Training Data')

# Plot prediction surface
ax.plot_surface(sleep_grid, study_grid, y_pred, cmap='viridis', alpha=0.7)

ax.set_xlabel('Sleep Hours')
ax.set_ylabel('Study Hours')
ax.set_zlabel('Score')
ax.set_title('Polynomial Regression Fit')
ax.legend()

plt.show()