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


import glob
import os, sys
from dicomUtils import load3dDicom
import numpy as np
import re

ME_GLOB_PATTERN = '*xd_me_gre*'

#dir_in = sys.argv[1]
dir_in = '.'

def getMEImages(dir_in):
    parentDir = os.path.abspath(dir_in)
    
    meDirs = sorted(glob.glob(os.path.join(parentDir, ME_GLOB_PATTERN)))
    
    mag = []
    phase = []
    TEs = []
    
    for d in meDirs:
        print("loading ",d)
        data, info = load3dDicom(d)
        if 'ImagePhase' in info[1].ImageComments:
            print("Phase")
            # phase image
            phase.append(data)
        else:
            try:
                teStr = re.search(r'TE \[ms\]:([\d.]+)', info[1].ImageComments).group(1)
            except:  # This means that the imagecomment does not contain TE: we reached the end of the dataset
                break
            print("Magnitude - TE: ", teStr)
            TEs.append(float(teStr))
            mag.append(data)
            
            
    mag = np.stack(mag, axis=-1)
    phase = (np.stack(phase, axis=-1) - 2048)*np.pi/2048
    
    images = mag * np.exp(1j * phase)
    
    return images, TEs

if __name__ == '__main__':
    try:
        dir_in = sys.argv[1]
    except IndexError:
        dir_in = '.'
    
    images, TEs = getMEImages(dir_in)
    np.save('images.npy', images)
        
