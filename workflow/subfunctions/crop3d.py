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
def crop3d(ROI, img, margin, background=0, returnCoord=True):
    # Handle variable margins
    if isinstance(margin, int) or isinstance(margin, float):
        margins = [int(margin), int(margin), int(margin)]
    else:
        margins = margin

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
    if centroid[0]+baseLength+margins[0]>I:
        xRangeMax=I
    else:
        xRangeMax=centroid[0]+baseLength+margins[0]
    
    if centroid[0]-baseLength-margins[0]<1:
        xRangeMin=0
    else:
        xRangeMin=centroid[0]-baseLength-margins[0]

    
    # y-direction
    if centroid[1]+baseLength+margins[1]>J:
        yRangeMax=J
    else:
        yRangeMax=centroid[1]+baseLength+margins[1]
    
    if centroid[1]-baseLength-margins[1]<1:
        yRangeMin=0
    else:
        yRangeMin=centroid[1]-baseLength-margins[1]    
        
    # z-direction
    if centroid[2]+baseLength+margins[2]>K:
        zRangeMax=K
    else:
        zRangeMax=centroid[2]+baseLength+margins[2]
    
    if centroid[2]-baseLength-margins[2]<1:
        zRangeMin=0
    else:
        zRangeMin=centroid[2]-baseLength-margins[2]
        
    # Images cropping
    halfWidths = [int(baseLength + m) for m in margins]
    cImg=np.ones((halfWidths[0]*2,
                halfWidths[1]*2,
                halfWidths[2]*2))*background
    cRoi=np.ones((halfWidths[0]*2,
                halfWidths[1]*2,
                halfWidths[2]*2))*background
            
    cImg[(halfWidths[0])-(centroid[0]-xRangeMin):(xRangeMax-centroid[0])+(halfWidths[0]),
          (halfWidths[1])-(centroid[1]-yRangeMin):(yRangeMax-centroid[1])+(halfWidths[1]),
          (halfWidths[2])-(centroid[2]-zRangeMin):(zRangeMax-centroid[2])+(halfWidths[2])]=\
        img[xRangeMin:xRangeMax,yRangeMin:yRangeMax,zRangeMin:zRangeMax]
        
    cRoi[(halfWidths[0])-(centroid[0]-xRangeMin):(xRangeMax-centroid[0])+(halfWidths[0]),
        (halfWidths[1])-(centroid[1]-yRangeMin):(yRangeMax-centroid[1])+(halfWidths[1]),
        (halfWidths[2])-(centroid[2]-zRangeMin):(zRangeMax-centroid[2])+(halfWidths[2])]=\
        ROI[xRangeMin:xRangeMax,yRangeMin:yRangeMax,zRangeMin:zRangeMax]
        
    if returnCoord:
        return(cImg, cRoi, 
                [int((halfWidths[0])-(centroid[0]-xRangeMin)),int((xRangeMax-centroid[0])+(halfWidths[0])),
                int((halfWidths[1])-(centroid[1]-yRangeMin)),int((yRangeMax-centroid[1])+(halfWidths[1])),
                int((halfWidths[2])-(centroid[2]-zRangeMin)),int((zRangeMax-centroid[2])+(halfWidths[2]))],
               [int(xRangeMin), int(xRangeMax), int(yRangeMin), int(yRangeMax), int(zRangeMin), int(zRangeMax)])
    else:
        return(cImg, cRoi)
        