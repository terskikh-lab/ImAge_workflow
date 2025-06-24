'''
Description s2_o4_3Dinterp_exfeatures.py
========================================
series 2 (s2): analysis of the 2D/3D plate images (for intestinal STEM cell and longitudinal blood samples)
apply 3D image interpolation and extract features
========================================
Kenta Ninomiya @ Sanford Burnham Prebys Medical Discovery Institute: 2022/11/07
'''

#import modules=======================
import os
import random
import time
import tifffile as tiff
import numpy as np
import pandas as pd
from scipy import ndimage


#import self defined functions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
from subfunctions.interp3d import lin3dinterp, shapelin3dinterp
from subfunctions.ELTA_functions import extract_MIELv023_tas_features
#=====================================

def s2_o4_3Dinterp_exfeatures(project='longiBLOOD',
                              statPara='TAS',
                              contents=[
                                        'DAPI',
                                        'H3K4me1',
                                        'H3K27ac',
                                        ],
                              segCh='DAPI',
                              illumiCorrection=True,
                              ):
    # time.sleep(random.random())
    #Initialization=======================
    parameterPath='../Data/Results/Parameters/'+project+'.csv'
    parameters=pd.read_csv(parameterPath, dtype=str)
    loadPath='../Data/Results/'+project+'/s2_o2_BSC_segmentation'
    loadPath1='../Data/Results/'+project+'/s2_o3_ptile'
    savePath='../Data/Results/'+project+'/s2_o4_3Dinterp_exfeatures'
    if os.path.exists(savePath)==False:
        os.makedirs(savePath, exist_ok=True)
        
    #parameters for the interpolation
    '''
    parameters for Phenix instrument
    20x: x-y plane: 0.6um/pixel, z: 1um/slice
    40x: x-y plane: 0.3um/pixel, z: 0.5um/slice
    '''
    # zStep=1/0.6
    zStep=1/1
    
    #kernel for the erosion
    kernel=np.zeros((3,3,3),dtype=np.uint8)
    strctImg=np.zeros((3,3))    
    strctImg[:,1]=1
    strctImg[1,:]=1
    
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd='seg_'+segCh
    if illumiCorrection==False:
        saveNameAdd='noIC_'+saveNameAdd
    #reorder them by alphabetical order
    contents.sort()
    #======================================
    
    #get the field of view list
    imgList=dir_rmv_file(loadPath, saveNameAdd+'imgs*.pickle')
    
    for tmpImg in random.sample(imgList, len(imgList)):
        # Check the existence of results (if exists, calculation is skipped)=======================
        saveFileName=savePath+'/'+statPara+'_'+'_'.join(contents)+'_'+tmpImg.split('.pickle')[0]+'.pickle'
        idxFileName=savePath+'/.'+statPara+'_'+'_'.join(contents)+'_'+tmpImg.split('.pickle')[0]+'.pickle'
        # res=progressregister(saveFileName,idxFileName,recheck=False)
        # if res:
        #     continue
        # ==========================================================================================
        imgPath=loadPath+'/'+tmpImg
        img=ezload(imgPath)['cellImgList']
        cellList=list(img.keys())
        print(tmpImg)
        if len(cellList)==0:
            print('no cells')
            continue
        availContents=list(img[cellList[0]].keys())
        availContents.remove('mask')
        #get overlapping contents
        counts=np.unique(contents+availContents,return_counts=True)[1]
        if len(contents)!=((counts==2).sum()):
            print('contents is not fully available')
            continue
        
        cellFeats=[]
        for tmpCell in cellList:
            cell=img[tmpCell]
            mask=cell['mask']
            if mask.sum()==0: #empty mask
                continue
            
            #channle wise image requantization (binning in the co-occurrence matrix)
            if zStep!=1:
                matZ,matX,matY=mask.shape
                xPost = np.linspace(0,matX-1,matX)
                yPost = np.linspace(0,matY-1,matY)
                zPost = np.linspace(0,matZ-1,np.floor(matZ*zStep).astype(int))
                mask=shapelin3dinterp(mask, zPost, xPost, yPost)
                objImgs=[lin3dinterp(cell[i], zPost, xPost, yPost) for i in contents]
                
            else:
                objImgs=[cell[i] for i in contents]
            
            print(tmpCell)
            
            #compute TAS feature for each channel  
            tmpFeat=ELTAS(dict(zip(contents, objImgs)),
                            mask, 
                            contents)
            cellFeats.append(pd.DataFrame({tmpCell:tmpFeat}).transpose())
            
        if len(cellFeats)!=0:
            cellFeats=pd.concat(cellFeats)
            ezsave({'cellFeats':cellFeats,
                    'dummy':[]},
                saveFileName)
            idxremover(idxFileName)
            
            
def ELTAS(tmpFile,mask,keyList):
    allFeats=[]
    for key in keyList:
        tmpImg=tmpFile[key]
        #get the masked image
        seg_object_img = np.where(mask == 1, tmpImg, 0)
        # average intensity of object (ie, pixels inside the mask)
        object_avg_int = np.mean(tmpImg[np.where(mask==1)])
        allFeats.append(extract_MIELv023_tas_features(seg_object_img, 
                                                      key, 
                                                      object_avg_int))
    
    statMat=pd.concat(allFeats,axis=0,sort=False)
    return statMat

