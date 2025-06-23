import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load CSV file
train_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

# Split labels and image pixel data
labels = train_df["label"]
images = train_df.drop("label", axis=1)

# Normalize pixel values (from 0-255 to 0-1)
images = images / 255.0

# Reshape images into 28x28 grayscale format
images = images.values.reshape(-1, 28, 28, 1)

# One-hot encode labels (because we are using categorical classification)
labels = tf.keras.utils.to_categorical(labels, num_classes=10)

# setting up the model layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # convolution layer that looks at small peice of image at a time
    MaxPooling2D(pool_size=(2, 2)),  # it reduces size meaning a 2x2 matrix gets a single value

    Conv2D(64, (3, 3), activation='relu'),  # another convolution layer
    MaxPooling2D(pool_size=(2, 2)),  # another maxpool layer

    Flatten(),  # it converts the 2D grid to a 1D array as Dense takes all inputs 
    Dense(128, activation='relu'),
    Dropout(0.5),  # randomly kills/switch_off half neurons to avoid overfitting
    Dense(10, activation='softmax')  # softmax for multi-class classification (10 clothing categories)
])

# setting up some compiler setting
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Split train/val from the CSV
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert labels to one-hot (for categorical_crossentropy)
from tensorflow.keras.utils import to_categorical

train_images = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
train_labels = to_categorical(train_df["label"])

val_images = val_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
val_labels = to_categorical(val_df["label"])

# Now train the model
model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=5,
    batch_size=64
)

# Load and prepare test data
test_df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
test_images = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
test_labels = to_categorical(test_df["label"])

# Evaluate model on test data
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {accuracy:.4f}")
