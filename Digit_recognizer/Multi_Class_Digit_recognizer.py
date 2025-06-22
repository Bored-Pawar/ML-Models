import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.losses import SparseCategoricalCrossentropy
import random

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to  add channel dimesion (needed of Convo2D)
X_train = X_train.reshape(-1, 28, 28, 1) # the -1 tell numpy to to autoshape 
X_test = X_test.reshape(-1,28, 28, 1) 

# build model
model = Sequential([

    # First convolutional layer
    # 64 filters of size 3x3, processes input image of shape 28x28x1 (MNIST grayscale)
    Conv2D(64, (3, 3), input_shape=(28, 28, 1)),
    BatchNormalization(),        # Normalize the outputs (activations) of this conv layer
    Activation('relu'),          # Apply ReLU activation after BN

    # Second convolutional layer
    # Learns deeper features from first conv layer
    Conv2D(64, (5, 5)),
    BatchNormalization(),
    Activation('relu'),

    # Third convolutional layer with larger scanning area (5x5)
    # Helps detect complex patterns (e.g., curves/loops in digits)
    Conv2D(128, (5, 5)),
    BatchNormalization(),
    Activation('relu'),

    # Max pooling layer
    # Downsamples feature maps by taking the max in 2x2 regions
    MaxPooling2D((2, 2)),

    # Flatten the 3D feature maps to 1D for the dense layer
    Flatten(),

    # Fully connected layer with 128 neurons
    Dense(128),
    BatchNormalization(),        # Normalize before activation
    Activation('relu'),
    Dropout(0.5),                # Randomly turn off 50% of neurons to prevent overfitting

    # Output layer: 10 neurons for 10 digit classes (0-9)
    # Softmax converts outputs to probability scores
    Dense(10, activation='linear')
])

# compile model
model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
)

# train model
model.fit(X_train, y_train, epochs = 10, validation_split = 0.2)

# Save the trained model
model.save("digit_model.h5")

# evaluation of model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Predict a test image
import matplotlib.pyplot as plt

# Predict a test image
index = random.randint(0, 9999)  # You can change this to any test index (0–9999)
img = X_test[index]
actual_label = y_test[index]

# Display the image
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Actual: {actual_label}")
plt.axis('off')
plt.show()

# Reshape and predict
logits = model.predict(img.reshape(1, 28, 28, 1))
predicted_digit = np.argmax(logits)

# Terminal output
print("-------- Prediction Summary --------")
print(f"Actual Digit     : {actual_label}")
print(f"Model Prediction : {predicted_digit}")
print(f"Model Raw Output : {logits}")
