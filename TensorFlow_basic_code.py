# Note: in the hidden layer 1 and 2 using relu activation instead of sigmoid improved the loss by a huge margin
# relu function is relu(x) = max(0, x) 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Features: [temperature, time]
X = np.array([
    [1, 6],
    [2, 5],
    [3, 4],
    [4, 3],
    [5, 2],
    [6, 1],
    [6, 6],
    [5, 5],
    [4, 4],
    [3, 3],
    [2, 2],
    [1, 1],
    [7, 7],
    [8, 2],
    [2, 8],
    [7, 1],
    [1, 7],
    [5, 8],
    [8, 5],
    [0, 0]
], dtype=np.float32)

# Labels: 1 = class A, 0 = class B
# Data arranged so center-ish area is class 1, outer edges are class 0
y = np.array([
    0, 0, 1, 1, 1, 0,
    0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.float32)

# normalize input
norm_layer = tf.keras.layers.Normalization(axis = -1)
norm_layer.adapt(X)
X_norm = norm_layer(X)

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (2,)), # comma isnecessary so that python knows that (2,)>> is a tupple
    tf.keras.layers.Dense(5, activation='relu'),  # layer 1 hidden layer
    tf.keras.layers.Dense(3, activation='relu'),  # layer 2 hidden layer
    tf.keras.layers.Dense(1, activation = 'sigmoid')  # layer 3 output layer
])

# prepare model for training
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
    metrics = ["accuracy"]
)

# training the model
model.fit(X_norm, y, epochs = 100)

# taking inputs
X_new = np.array([[4.61, 3.96]], dtype=np.float32) # this is a 2d array and we need to use 2d array when working with tensor flow
X_new_norm = norm_layer(X_new)

# making predictions
pred = model.predict(X_new_norm) # predicts multiple inputs at once and returns a 2d array of 1 coulums an no of rows == batchsize
print(f"Predicted probability of good roast: {pred[0][0]:.4f}") # her [0][0] is calling the prediction of first input from x_new_norm

# --------------------------------------------- Matplotllib ------------------------------------------------------------
# Create a grid of temperature and time values
# Create mesh grid based on the new feature ranges
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
xx, yy = np.meshgrid(x1_range, x2_range)

# Create a grid of input features
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

# Normalize the grid using the same norm_layer you trained with
grid_norm = norm_layer(grid)

# Get predictions from the model
preds = model.predict(grid_norm).reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap="RdYlGn", alpha=0.4)
plt.colorbar(label="Predicted Probability")

# Plot actual data points
good = y == 1
bad = y == 0
plt.scatter(X[good, 0], X[good, 1], color='green', label='Class 1', edgecolor='k')
plt.scatter(X[bad, 0], X[bad, 1], color='red', label='Class 0', edgecolor='k')

# Axis labels and legend
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary and Data Points")
plt.legend()
plt.grid(True)
plt.show()

 
