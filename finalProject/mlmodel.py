import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from joblib import dump

import numpy as np


# Load data
input_dir = "C:\\Users\\charl\\Downloads\\Compressed\\images\\train"
categories = ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede',
              'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox',
              'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'HarborSeal', 'Hedgehog', 'Hippopotamus',
              'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard',
              'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl',
              'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda',
              'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp',
              'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger',
              'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra']

data = []
labels = []

for category in categories:
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        data.append(img)
        labels.append(category)

data = np.asarray(data)
labels = np.asarray(labels)

labels = to_categorical(labels)

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True,
                                                                    stratify=labels)


# Load the pre-trained model and add a classifier on top of the base model
base_model = VGG16(weights='imagenets', include_top=False,
                   input_shape=(224, 224, 3))
data = base_model.output
data = Flatten()(data)
data = Dense(256, activation='relu')(data)
data = Dense(len(categories), activation='softmax')(data)

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=data)


# Define training parameters
batch_size = 32
epochs = 50
learningRate = 1e-4

trainDataGenerator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                        fill_mode='nearest')

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learningRate, momentum=0.9),
              metrics=['accuracy'])

# Defining the callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)


def lr_schedule(epoch):
    if epoch < 10:
        return learningRate
    else:
        return learningRate * np.exp(-0.1)


lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(trainDataGenerator.flow(train_data, train_labels, batch_size=batch_size), epochs=epochs,
                    steps_per_epoch=train_data.shape[0] // batch_size, validation_data=(test_data, test_labels),
                    callbacks=[early_stopping, lr_scheduler])

# Evaluate model
label_pred = model.predict(test_data)
label_pred = np.argmax(label_pred, axis=1)
label_true = np.argmax(test_labels, axis=1)

accuracy = accuracy_score(label_true, label_pred)
precision = precision_score(label_true, label_pred, average='weighted')
recall = recall_score(label_true, label_pred, average='weighted')
f1 = f1_score(label_true, label_pred, average='weighted')

print('Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1 Score: {:.2f}%'.format(accuracy * 100,
                                                                                         precision * 100,
                                                                                         recall * 100,
                                                                                         f1 * 100))


# Save model
dump(model, 'Animals_prediction_model.joblib')