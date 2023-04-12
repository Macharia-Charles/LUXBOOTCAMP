# Import required libraries

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt
import pickle

# CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2)])
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(80, activation="softmax"))

# Define parameters and callbacks
adam = Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
bs = 30
train_dir = './animals-detection-images-dataset/train'
test_dir = "./animals-detection-images-dataset/test"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical', target_size=(224, 224), shuffle=True)
validation_generator = test_datagen.flow_from_directory(test_dir, batch_size=bs, class_mode='categorical', target_size=(224, 224), shuffle=True)

# Train and fit the model
history = model.fit(train_generator,
                              steps_per_epoch=train_generator.samples // bs,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples // bs)


# Evaluate the model
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Save the model
model.save('./Animals_prediction_model.h5')
weights = model.get_weights()
dump(weights, './Animals_prediction_model.joblib')
with open('./Animals_prediction_model.pkl', 'wb') as f:
    pickle.dump(weights, f)