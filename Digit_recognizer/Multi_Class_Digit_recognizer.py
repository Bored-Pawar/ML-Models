import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy

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
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation = 'relu'),
    Dropout(0.5), # randomly kills 50% of the neuron and avoids overfitting
    Dense(10, activation = 'linear')
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
index = 9  # You can change this to any test index (0–9999)
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
