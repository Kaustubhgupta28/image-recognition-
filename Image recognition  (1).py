#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Kaustubh Gupta\Downloads\test\PetImages',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\Kaustubh Gupta\Downloads\Train\animals',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)


# In[3]:


cnn = Sequential()

# Convolution Block 1
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
cnn.add(MaxPooling2D(pool_size=(2,2)))

# Block 2
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

# Block 3
cnn.add(Conv2D(128,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
cnn.add(Flatten())

# Fully connected layer
cnn.add(Dense(128, activation='relu'))

# Output layer
cnn.add(Dense(1, activation='sigmoid'))

cnn.summary()


# In[4]:


cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[5]:


history = cnn.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,
    validation_data=test_generator,
    validation_steps=50
)


# In[10]:


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()


# In[14]:


img_path = r"C:\Users\Kaustubh Gupta\Downloads\test\PetImages\Cat\998.jpg"  

img = image.load_img(img_path, target_size=(64,64))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

result = cnn.predict(img)

print(result)

if result[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")


# In[15]:


img_path = r"C:\Users\Kaustubh Gupta\Downloads\test\PetImages\Dog\9989.jpg"

img = image.load_img(img_path, target_size=(64,64))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

result = cnn.predict(img)

print(result)

if result[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")


# In[ ]:




