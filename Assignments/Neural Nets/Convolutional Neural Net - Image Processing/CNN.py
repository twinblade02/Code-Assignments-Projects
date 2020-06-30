## Image data
# CNN libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# image preprocessing
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range= 0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_set = train_datagen.flow_from_directory('C:/Users/nisch/Desktop/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
                                                    target_size=(64,64), batch_size=32, class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set= test_datagen.flow_from_directory('C:/Users/nisch/Desktop/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
                                           target_size=(64,64), batch_size=32, class_mode='binary')

## Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D

cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# adding second convolutional layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# add flattening layer
from keras.layers import Flatten
cnn.add(Flatten())

# add full connection layer
from keras.layers import Dense
cnn.add(Dense(units=128, activation='relu'))

# add output layer
cnn.add(Dense(units=1, activation='sigmoid'))

# training the CNN
cnn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# evaluating on test set
cnn.fit(x= train_set, validation_data= test_set, epochs = 25)

# predicting a result
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/nisch/Desktop/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_2.jpg',
                            target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
# train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)