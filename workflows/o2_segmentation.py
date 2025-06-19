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

#import self defined subfunctions========
from subfunctions.extract_metadata_to_df import extract_metadata_to_df
from subfunctions.check_zslice_consistency import check_zslice_consistency
from subfunctions.progress_printer import progress_printer
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.imreqant import imreqant
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
from subfunctions.crop3d import crop3d

#=====================================
time.sleep(random.random())

def s2_o2_BSC_segmentation(project='longiBLOOD',
                           orgDataLoadPath=orgDataLoadPath,
                           orgDataSubFolder=orgDataSubFolder,
                           resultsSavePath='../Data/Results',
                           imageFileRegEx=imageFileRegEx,
                           imageFileFormat=imageFileFormat,
                           segCh='DAPI',
                           illumiCorrection=False,
                           ):
    #Initialization=======================
    parameterPath=f'{resultsSavePath}/Parameters/{project}.csv'
    parameters=pd.read_csv(parameterPath, dtype=str)
    loadPath=orgDataLoadPath
    loadPath1=f'{resultsSavePath}/{project}/s2_o1_BaSiC_markwise'
    savePath=f'{resultsSavePath}/{project}/s2_o2_BSC_segmentation'
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
    imgPath=f'{loadPath}/{folder}/{orgDataSubFolder}'
    imgList=np.array(dir_rmv_file(imgPath, f'*{imageFileFormat}'))
    
    # Extract metadata for each file and build rcfpIdx as a DataFrame
    rcfpIdx = extract_metadata_to_df(imgList, imageFileRegEx)

    # Check z slice consistency
    zposNum = check_zslice_consistency(rcfpIdx)
    
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
                    basic=ezload(loadPath1+'/model_'+segCh+'_f'+fn+'.pickle')['basic']
                    zStackImgC=basic.transform(zStackImg)[0]
                else:
                    zStackImgC=zStackImg
                normImg=imreqant(zStackImgC, np.percentile(zStackImgC, 1), np.percentile(zStackImgC, 99), 0, 1)
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
