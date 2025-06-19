#Description imreqant
#================================================
# img ==> numpy array
#=================================================
#Kenta Ninomiya @ Kyushu University: 2020/7/16

#import modules=======================
import numpy as np
#=====================================

def imreqant(img, orgLow, orgHigh, qLow, qHigh, outLow=False, outHigh=False, getInt=False):
    if orgLow==orgHigh:
        qImg=np.where(img==orgLow,(qLow+qHigh)/2, 0)
    else:
        if outLow==False:
            outLow=qLow
        if outHigh==False:
            outHigh=qHigh
        #Quantization of the grayscale levels in the ROI
        qImg=((img-orgLow)/(orgHigh-orgLow))*(qHigh-qLow)+qLow
        qImg=np.where(qImg>qHigh, outHigh, qImg)
        qImg=np.where(qImg<qLow, outLow, qImg)
        
    if getInt==True:
        qImg=np.floor(qImg+0.5)
        
    return(qImg) 