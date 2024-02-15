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
model.load_weights("my_flowers_model")

loss, acc = model.evaluate(train, verbose=2)

img = keras.utils.load_img("23894449029_bf0f34d35d_n.jpg", target_size=(img_h,img_w))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array,0)

#predict
prediction = model.predict(img_array)
score = tf.nn.softmax(prediction[0])
print(class_names[np.argmax(score)])