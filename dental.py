# Problem: Predict dental age of children from intraoral images.
# Objective: Build a deep learning model to analyze mixed dentition stage images.
# Metrics: Use Mean Absolute Error (MAE) or accuracy for evaluation.
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))  # Output layer for regression (predicting age)
    
    return model


import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # Assuming regression task for dental age prediction
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()



