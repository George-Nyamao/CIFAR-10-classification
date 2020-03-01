#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import cifar10
if K.backend()=='tensorflow':
    K.set_image_data_format("channels_last")


# In[2]:


# Importing more python libraries
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Load the CIFAR-10 Python dataset from the website
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[4]:


# Loading the classes and labels
classes = 10
cifar_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[5]:


# check one image
r = 7
plt.imshow(x_train[r])
print(cifar_labels[y_train[r][0]])


# In[6]:


# view the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(cifar_labels[y_train[i][0]])
plt.show()


# In[7]:


# Normalizing
mean = np.mean(x_train, axis = (0,1,2,3))
std = np.std(x_train, axis = (0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
mean


# In[8]:


# one hot encoding
y_train = to_categorical(y_train, classes, dtype=int)
y_test = to_categorical(y_test, classes, dtype=int)


# In[9]:


# Checking the shape
print(x_train.shape)
y_train.shape


# In[10]:


# set the input shape
input_shape = (32,32,3)


# In[11]:


# Create the model
def cifar10Model():
    model = Sequential()
    # first layer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',\
                     input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Dropout(0.2))
    
    # second layer
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Dropout(0.2))
    
    # third layer
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size =(2, 2)))
    model.add(Dropout(0.2))
    
    # last layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    
    return model  


# In[12]:


# Clear cache
K.clear_session()
model = cifar10Model()


# In[13]:


# Compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


model.summary()


# In[15]:


batch_size = 64
epochs = 100

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test), shuffle=True)


# In[2]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[1]:


# loss curves
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], 'orange',linewidth=3.0)
plt.plot(history.history['val_loss'], 'blue', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14, loc = 'upper right')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss curves', fontsize=16)


# In[18]:


# Accuracy curves
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], 'orange',linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14, loc = 'lower right')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy curves', fontsize=16)


# In[ ]:





# In[19]:


# Make a prediction
prediction = model.predict(x_test, batch_size=32)


# In[20]:


print(prediction[0])


# In[21]:


labels_pred = np.argmax(prediction, axis=1)
labels_pred =np.array(labels_pred)
print(labels_pred)


# In[22]:


labels_test = np.argmax(y_test, axis=1)
print(labels_test)


# In[23]:


correct = (labels_pred == labels_test)
print(correct)
print("Number of correct predictions: %d" % sum(correct))


# In[24]:


num_images = len(correct)
parcent = (sum(correct)*100)/num_images
print('Accuracy: %.2f%%' % parcent)


# In[25]:


incorrect = (correct == False)
print(incorrect)


# In[26]:


# Images of the test-set that have been incorrectly classified.
images_error = x_test[incorrect]

# Get predicted classes for those images
labels_error = labels_pred[incorrect]

# Get true classes for those images
labels_true = labels_test[incorrect]


# In[27]:


print(labels_true)


# In[28]:


plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_error[i])
    plt.xlabel('Actual: %s\nPredicted: %s'%(cifar_labels[labels_true[i]], cifar_labels[labels_error[i]]))
plt.show()


# In[29]:


'''
classes = 10
cifar_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=labels_true, y_pred=labels_error)
for i in range(classes):
    class_name = "({}) {}".format(i, cifar_labels[i])
    print(cm[i, :], class_name)
    
class_numbers = ["({0})".format(i) for i in range(classes)]
print("".join(class_numbers))


# In[43]:


# load the image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = (img-mean)/(std+1e-7)
    return img

img= load_image('images/sample_image1.jpg')
result = model.predict(img, batch_size=None, steps=1)
test = np.argmax(result, axis=1)
test =np.array(test)
print(cifar_labels[test[0]])


# In[ ]:




