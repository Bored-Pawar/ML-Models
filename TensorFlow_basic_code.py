# Note: in the hidden layer 1 and 2 using relu activation instead of sigmoid improved the loss by a huge margin
# relu function is relu(x) = max(0, x) 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Features: [temperature, time]
X = np.array([
    [200, 17],
    [180, 13],
    [210, 19],
    [190, 15],
    [195, 16],
    [205, 18],
    [220, 20],
    [170, 11],
    [185, 14],
    [215, 21],
    [165, 10],
    [175, 12],
    [225, 22],
    [160, 9],
    [230, 23],
    [200, 18],
    [210, 20],
    [180, 14],
    [220, 21],
    [190, 16]
], dtype=np.float32)

# Labels: 1 = good roast, 0 = bad roast
y = np.array([
    1, 0, 1, 0, 0,
    1, 1, 0, 0, 1,
    0, 0, 1, 0, 1,
    1, 1, 0, 1, 0
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
X_new = np.array([[205, 18]], dtype=np.float32) # this is a 2d array and we need to use 2d array when working with tensor flow
X_new_norm = norm_layer(X_new)

# making predictions
pred = model.predict(X_new_norm) # predicts multiple inputs at once and returns a 2d array of 1 coulums an no of rows == batchsize
print(f"Predicted probability of good roast: {pred[0][0]:.4f}") # her [0][0] is calling the prediction of first input from x_new_norm



