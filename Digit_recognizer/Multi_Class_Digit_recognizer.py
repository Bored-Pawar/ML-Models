import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.losses import SparseCategoricalCrossentropy
from random import randint
from sklearn.model_selection import StratifiedShuffleSplit # this allows validation data to have equal amount of all classes
from keras.callbacks import ReduceLROnPlateau

# why have ReduceLrROn Plateau >>> Early in training, big steps help explore and learn quickly. Later in training,
# especially near a minimum, you want small, precise steps so you don't overshoot or bounce around the minimum.
lr_schedule = ReduceLROnPlateau(  # learning rate fine tuning during the model training
    monitor='val_loss',     # What to monitor
    factor=0.5,             # Reduce LR by half
    patience=3,             # Wait 3 epochs before reducing LR
    verbose=1,              # Print when it reduces LR
    min_lr=1e-6             # Don’t go below this
)

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
    # 32 filters of size 3x3, processes input image of shape 28x28x1 (MNIST grayscale)
    Conv2D(32, (3, 3),  padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),        # Normalize the outputs (activations) of this conv layer
    Activation('relu'),          # Apply ReLU activation after BN

    # Second convolutional layer
    # Learns deeper features from first conv layer
    Conv2D(64, (5, 5),  padding='same'),
    BatchNormalization(),
    Activation('relu'),

    # Max pooling layer
    # Downsamples feature maps by taking the max in 2x2 regions
    MaxPooling2D((2, 2)),

    # Third convolutional layer with larger scanning area (5x5)
    # Helps detect complex patterns (e.g., curves/loops in digits)
    Conv2D(128, (5, 5),  padding='same'),
    BatchNormalization(),
    Activation('relu'),

    # Max pooling layer
    # Downsamples feature maps by taking the max in 2x2 regions
    MaxPooling2D((2, 2)),

    # Flatten the 3D feature maps to 1D for the dense layer
    Flatten(),

    Dropout(0.3), # Randomly turn off 30% of neurons to prevent overfitting

    # Fully connected layer with 128 neurons
    Dense(512),
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

# Stratified split: 80% training, 20% validation
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #random_state number is used to keep the randomization always the same 42 because The Hitchhiker’s Guide to the Galaxy joke
for train_idx, val_idx in splitter.split(X_train, y_train):
    X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]
# if u want to split in a 60 : 20 : 20 then u use same syntax but do two split functions,
# split 1 >> 80 : 20 where 80 is x_train + x_cv and 20 is x_test
# split 2 >> 75 : 25 where 75 is x_train and 25 is c_cv
# we didn't do this here becoz we already have a different test_dataset

# Train the model with stratified validation
model.fit(X_train_split, y_train_split, epochs=30, batch_size = 64, validation_data=(X_val, y_val), callbacks=[lr_schedule])

# # train model
# model.fit(X_train, y_train, epochs = 10, validation_split = 0.2)

# Save the trained model
model.save("digit_model.h5")

# evaluation of model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Predict a test image
import matplotlib.pyplot as plt

# Predict a test image
index = randint(0, 9999)  # You can change this to any test index (0–9999)
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
