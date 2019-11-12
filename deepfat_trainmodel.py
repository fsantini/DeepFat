#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file is part of DeepFat.

    DeepFat is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
    
    Copyright 2019 Francesco Santini <francesco.santini@unibas.ch>
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from cudaPatches import getPatches

from config import PATCHEXTENT, MODEL, arrayFile, valueFile, ECHOES, prepareInput

import sys

print("Patch extent", PATCHEXTENT)
print("Model", MODEL)

PATCHSIZE = 2*PATCHEXTENT+1

def makeModel():
    return keras.Sequential([
    keras.layers.Flatten(input_shape=(PATCHSIZE, PATCHSIZE, ECHOES, 2)),
    keras.layers.Dense(200, activation=tf.nn.relu), #relu
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, activation=tf.nn.relu), #relu
    keras.layers.Dropout(0.05),
    #keras.layers.Dense(50, activation=tf.nn.relu), #relu
    #keras.layers.Dropout(0.1),
    #keras.layers.Dense(5, activation=tf.nn.relu), #relu
    #keras.layers.Dropout(0.2),
    #keras.layers.Dense(10, activation=tf.nn.tanh), #relu
    #keras.layers.Dropout(0.1),
    #keras.layers.Dense(4, activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
#    keras.layers.Dense(8, activation=tf.nn.tanh),
#    keras.layers.Dropout(0.2),
#    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2)])

newModel = False

try:
    if int(sys.argv[1]) > 0:
        newModel = True
except:
    pass
    
try:
    if not newModel:
        model = keras.models.load_model(MODEL)
        print("***************************************** MODEL EXISTS *******************************************")
except:
    newModel = True

if newModel:    
    model = makeModel()
    
#    keras.Sequential([
#        keras.layers.Flatten(input_shape=(PATCHSIZE, PATCHSIZE, 6, 2)),
#        keras.layers.Dense(200, activation=tf.nn.tanh), #relu
#        keras.layers.Dropout(0.1),
#        keras.layers.Dense(200, activation=tf.nn.tanh), #relu
#        keras.layers.Dropout(0.05),
    #    keras.layers.Dense(4, activation=tf.nn.relu),
    #    keras.layers.Dropout(0.2),
    #    keras.layers.Dense(8, activation=tf.nn.tanh),
    #    keras.layers.Dropout(0.2),
    #    keras.layers.Dense(128, activation=tf.nn.relu),
#        keras.layers.Dense(1)
#    ])

    
# model working reasonably well:
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(PATCHSIZE, PATCHSIZE, 6, 2)),
#    keras.layers.Dense(100, activation=tf.nn.relu), #relu
#    keras.layers.Dropout(0.3),
#    keras.layers.Dense(50, activation=tf.nn.relu), #relu
#    keras.layers.Dropout(0.3),
#    keras.layers.Dense(10, activation=tf.nn.relu), #relu
#    keras.layers.Dropout(0.3),
##    keras.layers.Dense(4, activation=tf.nn.relu),
##    keras.layers.Dropout(0.2),
##    keras.layers.Dense(8, activation=tf.nn.tanh),
##    keras.layers.Dropout(0.2),
##    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(2)
#])

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_absolute_error',
                optimizer='adam',
                metrics=['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error'])

trainArray = prepareInput(np.load(arrayFile).astype(np.float32))

#trainValues = 100*np.load('trainValues.npy').astype(np.float32)-50

trainValues = np.load(valueFile)*100
#trainValues = (trainValues[:,0])/(trainValues[:,1]+trainValues[:,0]+0.0000001)*100-50 # train on the ff
#trainValues = (trainValues[:,0])/(trainValues[:,1]+trainValues[:,0]+0.0000001)*100 # train on the ff




print("Trainable variables:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

#scaler = StandardScaler()
#trainValues = scaler.fit_transform(trainValues)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print('.', end=' ')
        if epoch % 100 == 0: print('')

p = PrintDot()

history = model.fit(trainArray, trainValues, epochs=50, validation_split=0.2, verbose=1) #, callbacks=[p])

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.show()


plot_history(history)
model.save(MODEL)
