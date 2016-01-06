# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
import numpy as np
#np.random.seed(1001)
import pickle


def get_label(i):
    a = np.zeros(2)
    a[i] = 1
    return a

# load dataset
f = open('dataset.pkl')
ds = pickle.load(f)
f.close()
data = ds['data_mfcc']
print(data.shape)
data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
target = ds['target']
print(data.shape)

# dataset segmentation
(trainX, testX, trainY, testY) = train_test_split(data / 100,
    target, test_size=0.33)
trainY = np.array([get_label(l) for l in trainY])
testY = np.array([get_label(l) for l in testY])
#trainX = trainX.reshape()
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# train model
model = Sequential()
model.add(Dense(400, input_shape=(data.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='mse', optimizer=SGD(lr=0.08, momentum=0.95,
    batch_size=10))

model.fit(trainX, trainY, nb_epoch=7, verbose=1, show_accuracy=True,
    validation_data=(testX, testY))

preds = model.predict(testX)
preds = np.array([list(l).index(np.max(l)) for l in preds])
testY = np.array([list(l).index(np.max(l)) for l in testY])
print((classification_report(testY, preds, target_names=['class ' + str(i) for i in range(10)])))

