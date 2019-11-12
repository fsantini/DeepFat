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

try:
    tf
except NameError:
    import tensorflow as tf
    from tensorflow import keras

import numpy as np

from cudaPatches import getPatches

from config import PATCHEXTENT, MODEL, PATCHSPACING, ECHOES, prepareInput, calcWF

import sys
import os.path


def fitDataset(images):
    imasz = images.shape
    
    water = np.zeros(imasz[0:3])
    fat = np.zeros(imasz[0:3])
    
    predictions = []
    
    nnModel = keras.models.load_model(MODEL)
    nnModel.summary()
    
    print("Fitting...")
    
    predictions = []
    
    echoes_in = images.shape[-1]
    
    for s in range(imasz[2]):
        print("Slice", s)
        complexData = images[:,:,s,:].squeeze()
        patchArr = getPatches(complexData, PATCHEXTENT, echoes_in, PATCHSPACING);
    
        pred = nnModel.predict( prepareInput(patchArr) )
        predictions.append(pred)
    
    predictions = np.concatenate(predictions, axis=0).astype(np.float32)
    
    ff = np.zeros_like(water)
    
    print("Reformatting...")
    
    w_arr, f_arr, ff_arr = calcWF(predictions)
    
    # predictions are in format slice, row, columns
    water = np.reshape(w_arr, (imasz[2], imasz[0], imasz[1])).transpose([1,2,0])
    fat = np.reshape(f_arr, (imasz[2], imasz[0], imasz[1])).transpose([1,2,0])
    ff = np.reshape(ff_arr, (imasz[2], imasz[0], imasz[1])).transpose([1,2,0])

    return water, fat, ff

if __name__ == '__main__':
    
    dataset = sys.argv[1]
    
    if os.path.isdir(dataset):
        from loadMEdataset import getMEImages
        images, _ = getMEImages(dataset)
        out_folder = dataset
    else:
        images = np.load(dataset)
        out_folder, _ = os.path.split(dataset)
    
    PATCHSIZE = 2*PATCHEXTENT+1
    
    images = images.astype(np.complex64)/4096
    
    water, fat, ff = fitDataset(images)
    
    np.save(os.path.join(out_folder, "ff_{}x{}_spc{}_eco{}_out.npy".format(PATCHEXTENT, PATCHEXTENT, PATCHSPACING, ECHOES)), ff )
    np.save(os.path.join(out_folder, "w_{}x{}_spc{}_eco{}_out.npy".format(PATCHEXTENT, PATCHEXTENT, PATCHSPACING, ECHOES)), water*1000 )
    np.save(os.path.join(out_folder, "f_{}x{}_spc{}_eco{}_out.npy".format(PATCHEXTENT, PATCHEXTENT, PATCHSPACING, ECHOES)), fat*1000 )
