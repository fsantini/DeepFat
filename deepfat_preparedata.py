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

import numpy as np
import glob
import os

from cudaPatches import getPatches

from config import PATCHEXTENT, arrayFile, valueFile, PATCHSPACING, ECHOES

import sys

TH = 0.1

DISCARD_SLICES = 8

echoes_training = ECHOES
   
print("Patch extent", PATCHEXTENT)
print("arrayFile", arrayFile)


APPEND = True

try:
    APPEND = (int(sys.argv[2]) == 0)
except:
    pass
        
print("Append",APPEND)

NPY_PATH = sys.argv[1]

print("Path", NPY_PATH)
    
NPY_IMA = os.path.join(NPY_PATH, 'images.npy')
NPY_WATER = os.path.join(NPY_PATH, 'WATER.npy')
NPY_FAT = os.path.join(NPY_PATH, 'FAT.npy')

MASKS = glob.glob(os.path.join(NPY_PATH, 'roi*.npy'))

print("Masks", MASKS)

def cropData(data_in):
    if data_in.ndim == 3:
        return data_in[:,:,DISCARD_SLICES:-DISCARD_SLICES]
    else:
        return data_in[:,:,DISCARD_SLICES:-DISCARD_SLICES,:]

images = np.load(NPY_IMA).astype(np.complex64)/4096

echoes = images.shape[3]

def loadMask(flist):
    if not flist:
        return None
    if type(flist) == str:
        flist = [flist]
    
    mask = None    
    for f in flist:
        mask = np.logical_or(mask, np.load(f))
    
    mask[mask > 0] = 1
    mask[mask < 1] = 0    
    
    return mask.astype(np.uint8)

#load nii
waterData = np.load(NPY_WATER).astype(np.float32)
fatData = np.load(NPY_FAT).astype(np.float32)

maskData = loadMask(MASKS)
waterData *= np.logical_not(maskData)
fatData *= np.logical_not(maskData)

waterData = cropData(waterData)
fatData = cropData(fatData)

waterValues = np.array([])
fatValues = np.array([])
patchArrayDataset = []

# get patches
imasz = waterData.shape
reconSlice = np.zeros([imasz[0], imasz[1], imasz[2]])
for s in range(0,imasz[2],2):
    print("Slice", s)
    complexDataset = images[:,:,s,:].squeeze();
    patchArraySlice = getPatches(complexDataset, PATCHEXTENT, echoes, PATCHSPACING)
    patchArray = None
    for r in range(PATCHEXTENT+1,imasz[0]-PATCHEXTENT,3):
        #print r
        for c in range(PATCHEXTENT+1,imasz[1]-PATCHEXTENT,3):
            waterVal = waterData[r,c,s]
            fatVal = fatData[r,c,s]
            patchIndex = r*imasz[1]+c
            reconSlice[r,c,s] = patchArraySlice[patchIndex, PATCHEXTENT, PATCHEXTENT, 0, 0]
            if (waterVal+fatVal) < TH: continue
            waterValues = np.append(waterValues, waterVal)
            fatValues = np.append(fatValues,fatVal)
            patchArrayDataset.append(np.expand_dims(patchArraySlice[patchIndex,:,:,0:echoes_training,:],axis=0))
            

patchArrayDataset = np.concatenate(patchArrayDataset, axis=0)    

if APPEND:
    oldDataset = np.load(arrayFile)
    np.save(arrayFile, np.concatenate([oldDataset, patchArrayDataset]))
else:
    np.save(arrayFile, patchArrayDataset)


trainValues = np.stack([fatValues, waterValues])
trainValues = trainValues.T

if APPEND:
    oldTrain = np.load(valueFile)
    np.save(valueFile, np.concatenate([oldTrain, trainValues]))
else:
    np.save(valueFile, trainValues)
