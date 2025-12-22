#Description crop3d
#================================================
#crop images with margins 
#=================================================
#Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2022/10/08 
#modified 2023/05/16

#import modules=======================
import numpy as np
from numba import njit, jit

#=====================================
# @jit(nopython=True)
def crop3d(ROI, img, margin, background=0, returnCoord=False):
    I, J, K=ROI.shape
    
    xNonZerosLoc=np.where(np.sum(np.sum(ROI, axis=2),axis=1)!=0)[0]
    minX=xNonZerosLoc[0]
    maxX=xNonZerosLoc[-1]
    yNonZerosLoc=np.where(np.sum(np.sum(ROI, axis=2),axis=0)!=0)[0]
    minY=yNonZerosLoc[0]
    maxY=yNonZerosLoc[-1]
    zNonZerosLoc=np.where(np.sum(np.sum(ROI, axis=1),axis=0)!=0)[0]
    minZ=zNonZerosLoc[0]
    maxZ=zNonZerosLoc[-1]

    
    nonZeros=np.where(ROI!=0)
    centroid=[np.floor(nonZeros[0].mean()-1).astype(int),
              np.floor(nonZeros[1].mean()-1).astype(int),
              np.floor(nonZeros[2].mean()-1).astype(int)]
    
    baseLength=max([centroid[0]-minX,
                    maxX-centroid[0],
                    centroid[1]-minY,
                    maxY-centroid[1],
                    centroid[2]-minZ,
                    maxZ-centroid[2]])
    
    
    # matrix exceeding checker
    # x-direction
    if centroid[0]+baseLength+margin>I:
        xRangeMax=I
    else:
        xRangeMax=centroid[0]+baseLength+margin
    
    if centroid[0]-baseLength-margin<1:
        xRangeMin=0
    else:
        xRangeMin=centroid[0]-baseLength-margin

    
    # y-direction
    if centroid[1]+baseLength+margin>J:
        yRangeMax=J
    else:
        yRangeMax=centroid[1]+baseLength+margin
    
    if centroid[1]-baseLength-margin<1:
        yRangeMin=0
    else:
        yRangeMin=centroid[1]-baseLength-margin    
        
    # z-direction
    if centroid[2]+baseLength+margin>K:
        zRangeMax=K
    else:
        zRangeMax=centroid[2]+baseLength+margin
    
    if centroid[2]-baseLength-margin<1:
        zRangeMin=0
    else:
        zRangeMin=centroid[2]-baseLength-margin
        
    # Images cropping
    halfWiidth=baseLength+margin
    cImg=np.ones((halfWiidth*2,
                halfWiidth*2,
                halfWiidth*2))*background
    cRoi=np.ones((halfWiidth*2,
                halfWiidth*2,
                halfWiidth*2))*background
            
    cImg[(halfWiidth)-(centroid[0]-xRangeMin):(xRangeMax-centroid[0])+(halfWiidth),
          (halfWiidth)-(centroid[1]-yRangeMin):(yRangeMax-centroid[1])+(halfWiidth),
          (halfWiidth)-(centroid[2]-zRangeMin):(zRangeMax-centroid[2])+(halfWiidth)]=\
        img[xRangeMin:xRangeMax,yRangeMin:yRangeMax,zRangeMin:zRangeMax]
        
    cRoi[(halfWiidth)-(centroid[0]-xRangeMin):(xRangeMax-centroid[0])+(halfWiidth),
        (halfWiidth)-(centroid[1]-yRangeMin):(yRangeMax-centroid[1])+(halfWiidth),
        (halfWiidth)-(centroid[2]-zRangeMin):(zRangeMax-centroid[2])+(halfWiidth)]=\
        ROI[xRangeMin:xRangeMax,yRangeMin:yRangeMax,zRangeMin:zRangeMax]
        
    if returnCoord:
        return(cImg, cRoi, 
                [int((halfWiidth)-(centroid[0]-xRangeMin)),int((xRangeMax-centroid[0])+(halfWiidth)),
                int((halfWiidth)-(centroid[1]-yRangeMin)),int((yRangeMax-centroid[1])+(halfWiidth)),
                int((halfWiidth)-(centroid[2]-zRangeMin)),int((zRangeMax-centroid[2])+(halfWiidth))],
               [int(xRangeMin), int(xRangeMax), int(yRangeMin), int(yRangeMax), int(zRangeMin), int(zRangeMax)])
    else:
        return(cImg, cRoi)
        