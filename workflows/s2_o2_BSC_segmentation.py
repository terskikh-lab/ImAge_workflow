'''
Description s2_o2_BSC_segmentation.py
========================================
series 2 (s2): analysis of the 2D/3D plate images (for intestinal STEM cell and longitudinal blood samples)
apply the BaSiC Correction to the images and segment the cells using Stardist 3D
========================================
Kenta Ninomiya @ Sanford Burnham Prebys Medical Discovery Institute: 2022/10/31
'''

#import modules=======================
import os
import random
import time
import tifffile as tiff
import numpy as np
import pandas as pd
from stardist.models import StarDist3D
import tensorflow as tf
from basicpy import BaSiC
from skimage import filters
import platform    

#import self defined functions========
from Functions.dir_rmv_folder import dir_rmv_folder
from Functions.dir_rmv_file import dir_rmv_file
from Functions.glaylconvert import glaylconvert
from Functions.dbgimshow import dbgimshow
from Functions.progressregister import progressregister
from Functions.idxremover import idxremover
from Functions.crop2d import crop2d
from Functions.ezsave import ezsave
from Functions.ezload import ezload
from Functions.crop3d import crop3d

#=====================================
time.sleep(random.random())


def gpuinit(gpuN=None):
    if platform.system()!='Darwin':
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus)==0:
            tf.config.set_visible_devices([], 'GPU')
        else:
            if gpuN!=None:
                gpuIdx=gpuN
            else:    
                gpuIdx=random.sample(list(range(0,len(gpus))), 1)[0]
            tf.config.experimental.set_memory_growth(gpus[gpuIdx], True)
            tf.config.set_visible_devices(gpus[gpuIdx], 'GPU')


def s2_o2_BSC_segmentation(project='longiBLOOD',
                           segCh='DAPI',
                           chSet=['DAPI','H3K27me3','H3K27ac','H3K9ac'],
                           illumiCorrection=True,
                           orgDataLoadPath='../Data/Original',):
    #Initialization=======================
    parameterPath='../Data/Results/Parameters/'+project+'.csv'
    parameters=pd.read_csv(parameterPath, dtype=str)
    loadPath=orgDataLoadPath
    loadPath1='../Data/Results/'+project+'/s2_o1_BaSiC_markwise'
    savePath='../Data/Results/'+project+'/s2_o2_BSC_segmentation'
    if os.path.exists(savePath)==False:
        os.makedirs(savePath, exist_ok=True)
        
    model = StarDist3D.from_pretrained('3D_demo')
    
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd='seg_'+segCh
    if illumiCorrection==False:
        saveNameAdd='noIC_'+saveNameAdd
        
    #======================================
    #get the list of the folders contains original data
    folderList=np.array(dir_rmv_folder(loadPath))
    folderIndex=[projectindexer(n)==project for n in folderList]    #specify the project
    folder=folderList[folderIndex][0]
    
    #get the field of view list
    imgPath=loadPath+'/'+folder+'/Images'
    imgList=np.array(dir_rmv_file(imgPath, '*.tiff'))
    
    rcfpIdx=np.array([
        [f.split('-')[0].split('r')[1].split('c')[0] for f in imgList],
        [f.split('-')[0].split('c')[1].split('f')[0] for f in imgList],
        [f.split('-')[0].split('f')[1].split('p')[0] for f in imgList],
        [f.split('-')[0].split('p')[1] for f in imgList],
        [f.split('-')[1].split('ch')[1].split('sk')[0] for f in imgList]
        ])
    
    for fn in np.unique(rcfpIdx[2,:]):    #for each field of view
        for cn in np.unique(rcfpIdx[1,:]):   #for each column
            for rn in np.unique(rcfpIdx[0,:]):  #for each row
                #check if the file exists for the specific column, row, field, and channel
                #if the values of ExperimentalCondition are 0, skip the process
                if parameters.loc[(parameters['Row'].astype(int)==int(rn)) &
                                  (parameters['Column'].astype(int)==int(cn)),
                                  'ExperimentalCondition'].values[0]=='0':
                    continue
                tmpParams=parameters[(parameters['Row'].astype(int)==int(rn)) & (parameters['Column'].astype(int)==int(cn))]
                chList=np.array([tmpParams['Channel'+str(i+1)+'PrimaryAntibody'].values[0] for i in range(4)])
                #compare chList and chSet and skip the process if the channel set is not included in the chList
                if np.sum([ch in chList for ch in chSet])!=len(chSet):
                    continue
                
                #computation checkpoint
                #Check the existence of results (if exists, calculation is skipped)=======================
                saveFileName=savePath+'/'+saveNameAdd+'imgs_c'+cn+'_r'+rn+'_f'+fn+'.pickle'
                idxFileName=savePath+'/.'+saveNameAdd+'imgs_c'+cn+'_r'+rn+'_f'+fn+'.pickle'
                res=progressregister(saveFileName,idxFileName)
                if res:
                    continue
                #==========================================================================================    
                
                #get the channel 
                DAPICh=np.where(chList==segCh)[0][0]
                otherCh=np.where(np.logical_and(chList!=segCh,chList!='0'))[0]
                
                #get correction field
                imgStack=list()
                for pn in np.unique(rcfpIdx[3,:]):
                    tmpFileName=imgPath+'/r'+rn+'c'+cn+'f'+fn+'p'+pn+'-ch'+str(DAPICh+1)+'sk1fk1fl1.tiff'
                    tmpImg=tiff.imread(tmpFileName)
                    imgStack.append(tmpImg)
                    
                #apply the correction
                #get stacked numpy array form imgStack
                zStackImg=np.stack(imgStack,axis=0)
                dType=zStackImg.dtype
                
                #load the correction model
                if illumiCorrection:
                    basic=ezload(loadPath1+'/model_DAPI_f'+fn+'.pickle')['basic']
                    zStackImgC=basic.transform(zStackImg)[0]
                else:
                    zStackImgC=zStackImg
                normImg=glaylconvert(zStackImgC, np.percentile(zStackImgC, 1), np.percentile(zStackImgC, 99), 0, 1)
                normImg=filters.unsharp_mask(normImg, radius=5, amount=10)
                masks, _ = model.predict_instances(normImg, prob_thresh=0.75)
                
                cellsWBkg=np.unique(masks)
                cellImgList={}
                cellLocalCoordList={}
                cellGlobalCoordList={}
                for cellIdx in range(1,len(cellsWBkg)):
                    croppedImgs={}
                    tmpMask=(masks==cellsWBkg[cellIdx]).astype(float)
                    img, mask, localCoord, globalCoord=crop3d(ROI=tmpMask, img=zStackImgC, margin=3, returnCoord=True)
                    #visulize the cropped image
                    # import matplotlib.pyplot as plt
                    # ax=plt.figure()
                    # plt.imshow(tmpMask.max(axis=0))
                    # plt.show()

                    croppedImgs['mask']=mask.astype(np.bool) #save the mask as bool
                    croppedImgs[segCh]=img.astype(dType)
                    cellImgList['r'+rn+'c'+cn+'f'+fn+'_cell'+str(cellIdx)]=croppedImgs
                    cellLocalCoordList['r'+rn+'c'+cn+'f'+fn+'_cell'+str(cellIdx)]=localCoord
                    cellGlobalCoordList['r'+rn+'c'+cn+'f'+fn+'_cell'+str(cellIdx)]=globalCoord
                    
                for chN in otherCh:
                    #load images and get correction field
                    imgStack=list()
                    for pn in np.unique(rcfpIdx[3,:]):
                        tmpFileName=imgPath+'/r'+rn+'c'+cn+'f'+fn+'p'+pn+'-ch'+str(chN+1)+'sk1fk1fl1.tiff'
                        tmpImg=tiff.imread(tmpFileName)
                        imgStack.append(tmpImg)
                    
                    #apply the correction
                    #get stacked numpy array form imgStack
                    zStackImg=np.stack(imgStack,axis=0)
                    #load the correction model
                    if illumiCorrection:
                        basic=ezload(loadPath1+'/model_'+chList[chN]+'_f'+fn+'.pickle')['basic']
                        zStackImgC=basic.transform(zStackImg)[0]
                    else:
                        zStackImgC=zStackImg
                    
                    for cellIdx in range(1,len(cellsWBkg)):
                        croppedImgs=cellImgList['r'+rn+'c'+cn+'f'+fn+'_cell'+str(cellIdx)]
                        tmpMask=(masks==cellsWBkg[cellIdx]).astype(float)
                        img, _=crop3d(ROI=tmpMask, img=zStackImgC, margin=3) #crop the cell
                        croppedImgs[chList[chN]]=img.astype(dType)
                        cellImgList['r'+rn+'c'+cn+'f'+fn+'_cell'+str(cellIdx)]=croppedImgs
                
                ezsave({'cellImgList':cellImgList,
                        'cellLocalCoordList':cellLocalCoordList,
                        'cellGlobalCoordList':cellGlobalCoordList,
                        'dummy':[]},
                       saveFileName)
                idxremover(idxFileName)
                        
            
                
def projectindexer(name):
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return()   
