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

import numpy as np
import os

PATCHEXTENT = 5
PATCHSPACING = 1
ECHOES = 6 # how many echoes to use for the model - not necessarily the echoes in the dataset

if PATCHEXTENT == 0:
    PATCHSPACING = 1

MODEL = os.path.join("models", "deepFat_fw_{}x{}_sp{}_eco{}.h5".format(PATCHEXTENT, PATCHEXTENT,PATCHSPACING,ECHOES) )

arrayFile = os.path.join("data", "trainArray_{}_sp{}.npy".format(PATCHEXTENT, PATCHSPACING, ECHOES) )
valueFile = os.path.join("data", "trainValues_{}_sp{}.npy".format(PATCHEXTENT, PATCHSPACING, ECHOES) )

# basic input preparation
def prepareInput_basic(inputData):
    return 100*inputData[:,:,:,0:ECHOES,:]

# basic training output preparetion. The input is nx2 where second dimension is water and fat
def prepareOutput_basic(waterFatData):
    return 100*waterFatData
    
# this calculates water and fat from the output of the model.
def calcWF_basic(predictions):
    fat = predictions[:,0]
    water = predictions[:,1]
    ff = fat / (water+fat+0.00001)*100 # ff in percentage
    return water, fat, ff
    
prepareInput = prepareInput_basic

# prepares the input with mag/phase instead of re/im
#def prepareInput(inputData):
    #preparedData = 100*inputData[:,:,:,0:ECHOES,:]
    #complexData = preparedData[:,:,:,:,0] + 1j*preparedData[:,:,:,:,1]
    #preparedData[:,:,:,:,0] = np.abs(complexData)
    #preparedData[:,:,:,:,1] = np.angle(complexData)
    #return preparedData
    

prepareOutput = prepareOutput_basic
calcWF = calcWF_basic
