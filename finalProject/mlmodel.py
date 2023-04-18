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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(512, (3, 3), activation="relu"),
    MaxPooling2D(2, 2)
    ])
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(80, activation="softmax"))



model.summary()

# Define parameters and callbacks
adam = Adam(learning_rate=0.003)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
bs = 30
train_dir = "../images/train/"
test_dir = "../images/test/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical', target_size=(224, 224), shuffle=True)
validation_generator = test_datagen.flow_from_directory(test_dir, batch_size=bs, class_mode='categorical', target_size=(224, 224), shuffle=True)

# Train and fit the model
history = model.fit(train_generator,
                              steps_per_epoch=train_generator.samples // bs,
                              epochs=15,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples // bs)

# Save the model
model.save('./Animals_prediction_model.h5')
weights = model.get_weights()
dump(weights, './Animals_prediction_model.joblib')
with open('./Animals_prediction_model.pkl', 'wb') as f:
    pickle.dump(weights, f)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(validation_generator)

# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy over epochs
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
