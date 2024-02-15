import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import keras
from keras import Sequential, layers

dataset = pathlib.Path("flower_photos")
bath_size = 32
img_w = 180
img_h = 180

train = keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_h,img_w),
    batch_size=bath_size,
)
validate = keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_h,img_w),
    batch_size=bath_size,
)
class_names = train.class_names

AUTOTUNE = tf.data.AUTOTUNE
train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validate = validate.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),

    #агуминтация
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_h, img_w, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.2),


    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),#requliztion
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()

epoch = 10
history = model.fit(
    train,
    validation_data=validate,
    epochs=epoch,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Діапазон епох
epochs_range = range(epoch)

# Візуалізація точності
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Навчання accuracy')
plt.plot(epochs_range, val_acc, label='Валiдацiя accuracy')

plt.legend(loc='lower right')
plt.title('Точність навчання та валідації')

# Візуалізація втрат
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Навчання loss')
plt.plot(epochs_range, val_loss, label='Валiдацiя loss')

plt.legend(loc='upper right')
plt.title('Втрати навчання та валідації')

plt.show()

#save model

model.save_weights("my_flowers_model")
print("Model_saved")