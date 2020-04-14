import os
import numpy as np
import cv2
import pandas as pd
from keras import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.utils import np_utils 
import matplotlib.pyplot as plt

"""
train_x_shape = (9866, 128, 128, 3)
train_y_shape = (9866)
val_x_shape = (3430, 128, 128, 3)
val_y_shape = (3430)
test_x_shape = (3347, 128, 128, 3)
"""

np.random.seed(0)
train_x = np.load('./data/train_x.npy')
train_y = np.load('./data/train_y.npy')
val_x = np.load('./data/val_x.npy')
val_y = np.load('./data/val_y.npy')
#test_x = np.load('./data/test_x.npy')

######################################
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)    
    return (X[randomize], Y[randomize])

def show_train_history(title,train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()
    
######################################
train_x = train_x/255 #normalize
val_x = val_x/255 #normalize
#test_x = test_x/255 #normalize
train_y = np_utils.to_categorical(train_y) #onehot
val_y = np_utils.to_categorical(val_y) #onehot

train_x, train_y = _shuffle(train_x, train_y)
val_x, val_y = _shuffle(val_x, val_y)

# initializing CNN
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',
                 input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
#model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(11,activation='softmax'))

#print(model.summary())

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#train_history=model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y), 
#                            epochs=10, batch_size=300, verbose=2)

train_history=model.fit(x=train_x, y=train_y, validation_split=0.1, 
                            epochs=5, batch_size=300, verbose=2)

show_train_history('accuracy',train_history, 'accuracy', 'val_accuracy')
show_train_history('loss',train_history, 'loss', 'val_loss')

val_history = model.evaluate(x=val_x,y=val_y)

#prediction=model.predict_classes(val_x)
#pd.crosstab(val_y, prediction, rownames=['label'],colnames=['predict'])