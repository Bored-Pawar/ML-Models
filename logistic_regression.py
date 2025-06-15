# Import libraries 
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load data from CSV and perfom "Feature Engineering"
# generate new feature as square of sleep_hours & study_hours and multiplication of the two

features = []
targets = []

with open('logistic_regression_data.csv', 'r') as file:
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

X_norm, X_mean, X_std = z_normalize_train(X)

# create parameter numpy array w and parameter variable b and lambda
w = np.zeros(X.shape[1])
b = 0
logistic_lambda = 0.01 

def sigmoid(number):
    g = 1 / (1 + np.exp(-1 * number))
    return g

def cost_function(x, y, w, b):
    m = x.shape[0]
    total = 0
    epsilon = 1e-15
    for i in range(m):
        y_hat = sigmoid(np.dot(w, x[i]) + b)  
        y_hat = max(min(y_hat, 1 - epsilon), epsilon)  # avoid log(0)
        total += y[i] * math.log(y_hat) + (1 - y[i]) * math.log(1 - y_hat) 
    reg_term = (logistic_lambda / (2 * m)) * np.sum(w ** 2)
    return -total / m + reg_term

# function to calcualte gradient
def gradients(x, y, w, b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        y_hat = sigmoid(np.dot(x[i], w) + b)
        error = y_hat - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error

    # Now add regularization AFTER averaging
    for j in range(n):
        dj_dw[j] = dj_dw[j] / m + (logistic_lambda / m) * w[j]

    dj_db /= m

    return dj_dw, dj_db

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
alpha = 1e-1
iterations = 50000

w, b = gradient_descent(X_norm, y, w, b, alpha, iterations)

# function to normalize input
def z_normalize_input(x_input, X_mean, X_std):
    Z = (x_input - X_mean) / X_std
    return Z

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
    prompt = (list(map(float, input(" enter space seperated age & tumor size:\n").split())))
    if prompt[0] == -1:
        break
    x_input = np.array(add_input_feature(prompt))
    x_input_norm = z_normalize_input(x_input, X_mean, X_std)
    y_pred = sigmoid(np.dot(x_input_norm, w) + b)
    if y_pred >= 0.5:
        print(f"Prediction: Malignant ({y_pred:.3f})")
    else:
        print(f"Prediction: Benign ({y_pred:.3f})")

# -------------------------------------------- Matplotlib Code ----------------------------------------------------   

# # Assumes first two features are original (before engineering)
# x1 = X[:, 0]  # feature 1: e.g., tumor size
# x2 = X[:, 1]  # feature 2: e.g., age
# labels = y    # target labels (0 or 1)

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot benign (label 0)
# ax.scatter(x1[labels==0], x2[labels==0], labels[labels==0],
#            c='green', label='Benign (0)', s=50)

# # Plot malignant (label 1)
# ax.scatter(x1[labels==1], x2[labels==1], labels[labels==1],
#            c='red', label='Malignant (1)', s=50)

# ax.set_xlabel('Tumor Size (x1)')
# ax.set_ylabel('Age (x2)')
# ax.set_zlabel('Diagnosis (y)')
# ax.set_title('Tumor Diagnosis Based on Features')
# ax.legend()
# plt.show()

# Assuming you already have:
# X: your features (age, tumor_size, and engineered features)
# y: target labels
# w, b: learned parameters
# X_mean, X_std: mean and std for normalization

# Extract original features from X (age and tumor_size)
age = X[:, 0]
tumor_size = X[:, 1]

# Create mesh grid for plotting decision boundary
age_min, age_max = age.min() - 5, age.max() + 5
tumor_min, tumor_max = tumor_size.min() - 0.5, tumor_size.max() + 0.5

age_grid, tumor_grid = np.meshgrid(
    np.linspace(age_min, age_max, 300),
    np.linspace(tumor_min, tumor_max, 300)
)

# Flatten mesh grid points
flat_age = age_grid.ravel()
flat_tumor = tumor_grid.ravel()

# Feature engineering for grid points
features_grid = np.vstack([
    flat_age,
    flat_tumor,
    flat_age**2,
    flat_tumor**2,
    flat_age * flat_tumor
]).T

# Normalize the features using training mean and std
features_grid_norm = (features_grid - X_mean) / X_std

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculate model output for grid points
z = np.dot(features_grid_norm, w) + b
probs = sigmoid(z)

# Reshape probabilities back to grid shape
probs_grid = probs.reshape(age_grid.shape)

# Plot data points
plt.figure(figsize=(10, 7))
plt.scatter(tumor_size[y==0], age[y==0], color='green', label='Benign (0)', edgecolor='k', s=60)
plt.scatter(tumor_size[y==1], age[y==1], color='red', label='Malignant (1)', edgecolor='k', s=60)

# Plot decision boundary where prob = 0.5
contour = plt.contour(tumor_grid, age_grid, probs_grid, levels=[0.5], colors='blue', linewidths=2)
plt.clabel(contour, inline=True, fontsize=10)

plt.xlabel('Tumor Size (cm)')
plt.ylabel('Age')
plt.title('Tumor Diagnosis & Decision Boundary')
plt.legend()
plt.grid(True)
plt.xlim(tumor_min, tumor_max)
plt.ylim(age_min, age_max)

plt.show()