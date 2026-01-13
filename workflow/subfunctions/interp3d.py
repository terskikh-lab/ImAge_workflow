from numba import jit
import numpy as np
from scipy import ndimage
# import tricubic

#numba no-python compilation
@jit(nopython=True,cache=True)
def lin3dinterp(orgImg, xSample, ySample, zSample):
    interpImg=np.zeros((len(xSample),len(ySample),len(zSample)))
    for i in range(0,len(xSample)):
        for j in range(0,len(ySample)):
            for k in range(0,len(zSample)):
                #get the 8 nearest neighbors
                x1=int(np.floor(xSample[i]))
                x2=int(np.ceil(xSample[i]))
                y1=int(np.floor(ySample[j]))
                y2=int(np.ceil(ySample[j]))
                z1=int(np.floor(zSample[k]))
                z2=int(np.ceil(zSample[k]))
                
                # Boundary checks to prevent out of bounds access
                if x1 < 0: x1 = 0
                if x2 >= orgImg.shape[0]: x2 = orgImg.shape[0] - 1
                if y1 < 0: y1 = 0
                if y2 >= orgImg.shape[1]: y2 = orgImg.shape[1] - 1
                if z1 < 0: z1 = 0
                if z2 >= orgImg.shape[2]: z2 = orgImg.shape[2] - 1

                #if x1==x2 or y1==y2 or z1==z2 then use 1 as the weight
                if x1==x2:
                    xWeight=0.5
                else:
                    xWeight=(xSample[i]-x1)/(x2-x1)
                if y1==y2:
                    yWeight=0.5
                else:
                    yWeight=(ySample[j]-y1)/(y2-y1)
                if z1==z2:
                    zWeight=0.5
                else:
                    zWeight=(zSample[k]-z1)/(z2-z1)

                #interpolate
                interpImg[i,j,k]=orgImg[x1,y1,z1]*(1-xWeight)*(1-yWeight)*(1-zWeight)+\
                                orgImg[x2,y1,z1]*xWeight*(1-yWeight)*(1-zWeight)+\
                                orgImg[x1,y2,z1]*(1-xWeight)*yWeight*(1-zWeight)+\
                                orgImg[x1,y1,z2]*(1-xWeight)*(1-yWeight)*zWeight+\
                                orgImg[x2,y1,z2]*xWeight*(1-yWeight)*zWeight+\
                                orgImg[x1,y2,z2]*(1-xWeight)*yWeight*zWeight+\
                                orgImg[x2,y2,z1]*xWeight*yWeight*(1-zWeight)+\
                                orgImg[x2,y2,z2]*xWeight*yWeight*zWeight
                
    return interpImg

# @jit(nopython=True,cache=True)
# def cub3dinterp(orgImg, xSample, ySample, zSample):
#     interpMap=tricubic.tricubic(list(orgImg),range(orgImg.shape))
#     interpImg=np.zeros((len(xSample),len(ySample),len(zSample)))
#     for i in range(0,len(xSample)):
#         for j in range(0,len(ySample)):
#             for k in range(0,len(zSample)):
#                 interpImg[i,j,k]=interpMap.ip((xSample[i],ySample[j],zSample[k]))

#     return interpImg


def shapelin3dinterp(mask, zSample, xSample, ySample):
    matZ,matX,matY=mask.shape
    maskEdge=mask.copy().astype(np.float64)
    #get the edge of the mask using morphological erosion for each slice
    #and covert it to distance map
    for z in range(matZ):
        #zeropadding the image 
        tmpMask=np.zeros((matX+2,matY+2),dtype=np.uint8)
        tmpMask[1:-1,1:-1]=mask[z,:,:]
        if np.sum(mask[z,:,:])==0:
            maskEdge[z,:,:]=-1
            continue
        
        maskEdge[z,:,:]=(ndimage.distance_transform_edt(tmpMask)-ndimage.distance_transform_edt(1-tmpMask))[1:-1,1:-1]
    
    #get the edge of the mask using morphological erosion for each slice
    mask=lin3dinterp(maskEdge, zSample, xSample, ySample)
    mask=(mask>0).astype(int)
    
    return(mask)