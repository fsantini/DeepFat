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

import pycuda as cuda
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.gpuarray as ga

from pycuda.compiler import SourceModule

PATCHSIZE = 5;
NECHOES = 6;

def getCudaFunction(patchSize, nEchoes, patchSpacing = 1):    
    mod = SourceModule("""
            __global__ void extract_patches(float *patchArray, float *realArray, float *imArray)
            {
                const int patchSize = """ + str(patchSize) +  """;
                const int nEchoes = """ + str(nEchoes) +  """ ;
                const int patchSpacing = """ + str(patchSpacing) + """;
                
                const int pointIndex = blockIdx.x*blockDim.x + threadIdx.x;
                
                if (threadIdx.x < (patchSize*patchSpacing) || threadIdx.x >= blockDim.x-(patchSize*patchSpacing) || blockIdx.x < (patchSize*patchSpacing) || blockIdx.x >= gridDim.x-(patchSize*patchSpacing))
                {
                    for (int i=0; i<(2*patchSize+1)*(2*patchSize+1)*nEchoes*2; i++)
                    {
                        patchArray[pointIndex*2*(2*patchSize+1)*(2*patchSize+1)*nEchoes + i] = -1.;
                    }
                }
                else
                {
                    for (int r=-patchSize; r<=patchSize; r++)
                    {
                        for (int c=-patchSize; c<=patchSize; c++)
                        {
                            for (int e=0; e<nEchoes; e++)
                            {
                                int curPoint = ((blockIdx.x+r*patchSpacing)*blockDim.x + (threadIdx.x + c*patchSpacing))*nEchoes + e;
                                float re = realArray[curPoint];
                                float im = imArray[curPoint];
                                patchArray[(((pointIndex*(2*patchSize+1) + (r+patchSize))*(2*patchSize+1) + (c+patchSize))*nEchoes + e)*2] = re;
                                patchArray[(((pointIndex*(2*patchSize+1) + (r+patchSize))*(2*patchSize+1) + (c+patchSize))*nEchoes + e)*2 +1] = im;
                            }
                        }
                    }
                    
                }
            }""")
    return mod.get_function("extract_patches")
    
# blockdim: number of rows
# threadDim: number of columns

def getPatches(complexDataset, patchSize = 5, echoes = 6, patchSpacing = 1):
    
    PATCHSIZE=patchSize
    ECHOES=echoes
    
    extract_patches = getCudaFunction(patchSize, echoes, patchSpacing)
    
    blockSize = complexDataset.shape[1]
    gridSize = complexDataset.shape[0]
    
    realDataset = complexDataset.real.astype(np.float32).flatten()
    imDataset = complexDataset.imag.astype(np.float32).flatten()
    
#    free, total = drv.mem_get_info()
#    print '%.1f %% of device memory is free before alloc.' % ((free/float(total))*100)
    
    #patchArray = np.zeros([blockSize*gridSize*2*(2*PATCHSIZE+1)*(2*PATCHSIZE+1)*ECHOES], dtype=np.float32)
    
    real_gpu = ga.to_gpu(realDataset)
    im_gpu = ga.to_gpu(imDataset)
    
    out_gpu = ga.zeros([blockSize*gridSize*2*(2*PATCHSIZE+1)*(2*PATCHSIZE+1)*ECHOES], np.float32)
    
    #extract_patches(drv.Out(patchArray), drv.In(realDataset), drv.In(imDataset), block=(blockSize,1,1), grid=(gridSize,1))
    extract_patches(out_gpu, real_gpu, im_gpu, block=(blockSize,1,1), grid=(gridSize,1))
    
#    free, total = drv.mem_get_info()
#    print '%.1f %% of device memory is free after processing.' % ((free/float(total))*100)
    
    patchArray = out_gpu.get()
    
    real_gpu.gpudata.free()
    im_gpu.gpudata.free()
    out_gpu.gpudata.free()
    
#    free, total = drv.mem_get_info()
#    print '%.1f %% of device memory is free after dealloc.' % ((free/float(total))*100)
    
    patchArray = patchArray.reshape([blockSize*gridSize, (2*PATCHSIZE+1), (2*PATCHSIZE+1), ECHOES, 2])
    
    return patchArray
    
