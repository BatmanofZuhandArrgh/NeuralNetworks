#!/usr/bin/env python
# coding: utf-8

# In[3]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[4]:


#Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). 
#EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
from emnist import extract_training_samples
train_images, train_labels = extract_training_samples('letters')
#train_images.shape #Train images set
#train_labels.shape
#for i in range(len(train_labels)):
 #   train_labels[i] = train_labels[i] - 1
print(train_labels)


# In[5]:


from emnist import extract_test_samples
test_images, test_labels = extract_test_samples('letters')
#test_images.shape #Test images set
#test_labels.shape

#for i in range(len(test_labels)):
 #   test_labels[i] = test_labels[i] - 1
print(test_labels)


# In[6]:


#Showing a picture of a certain element in the image training set
plt.figure()
plt.imshow(train_images[13])
plt.colorbar()
plt.grid(False)
plt.show()


# In[7]:


#The output and the corresponding 26 classes
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
print (len(class_names))


# In[8]:


#Feature scaling for each element in the 2D array/picture. Each pixel is a value of 255, so we scale them to a scale of 0 to 1
train_images = train_images / 255.0

test_images = test_images / 255.0


# In[9]:


#Showing the first 25 training images in the set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]-1])
plt.show()


# In[10]:


Ni = train_images.shape[1]*train_images.shape[2] #Number of neurons in the input layer
No = len(class_names) #Number of neurons in the input layer
Ns = train_images.shape[0] #Number of samples in the training data set
a = 2 #Scaling factor, arbitrary
Nh = Ns/(a*(No+Ni)) #Rule of thumb according to  http://hagan.okstate.edu/NNDesign.pdf#page=469
print(round(Nh))


# In[11]:


#Creating the neural networks with 1 hidden layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #Flattening the 2D arrays into a 1D arrays with 28x28 elements - Input layer
    keras.layers.Dense(Nh, activation='relu'),
    keras.layers.Dense(26) #Output layers with 26 classes
])


# In[12]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[16]:


train_labels1 = list()
for i in range(len(train_labels)):
    train_labels1.append(train_labels[i] - 1.0)


# In[19]:


model.fit(train_images, train_labels-1, epochs=10)


# In[20]:


test_loss, test_acc = model.evaluate(test_images,  test_labels-1, verbose=2)

print('\nTest accuracy:', test_acc)


# In[36]:


#Assess the probability of falling into each class for an individual observation
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
index = 10400


# In[37]:


print(predictions[index])
print("The letter is labeled: ", test_labels[index]-1)
print("The letter is predicted to be: ", np.argmax(predictions[index]))


# In[ ]:




