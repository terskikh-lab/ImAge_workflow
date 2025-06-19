'''
Description o1_illumination_correction
========================================


========================================
'''

#import modules=======================
import os
import random
import time
import tifffile as tiff
import numpy as np
from basicpy import BaSiC
import pandas as pd

#import self defined subfunctions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.imreqant import imreqant
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload

#=====================================

def o1_illumination_correction(project='longiBLOOD',
                orgDataLoadPath='../Data/Original',
                orgDataSubFolder='Images',
                imageFileRegEx='',
                imageFileFormat='.tiff',
                ):
    # time.sleep(random.random())
    #Initialization=======================
    loadPath=orgDataLoadPath
    savePath=f'../Data/Results/{project}/o1_illumination_correction'

    if os.path.exists(savePath)==False:
        os.makedirs(savePath, exist_ok=True)
    #======================================
    #get the list of the folders contains original data
    folderList=np.array(dir_rmv_folder(loadPath))
    folderIndex=[projectindexer(n)==project for n in folderList]    #specify the project
    folder=folderList[folderIndex][0]
    
    #get the field of view list
    imgPath=f'{loadPath}/{folder}/{orgDataSubFolder}'
    imgList=np.array(dir_rmv_file(imgPath, imageFileRegEx))
    
    rcfpIdx=np.array([
        [f.split('-')[0].split('r')[1].split('c')[0] for f in imgList],
        [f.split('-')[0].split('c')[1].split('f')[0] for f in imgList],
        [f.split('-')[0].split('f')[1].split('p')[0] for f in imgList],
        [f.split('-')[0].split('p')[1] for f in imgList],
        [f.split('-')[1].split('ch')[1].split('sk')[0] for f in imgList]
        ])
    
    for ch in np.unique(rcfpIdx[4,:]):
        for fn in np.unique(rcfpIdx[2,:]):
            #computation checkpoint
            #Check the existence of results (if exists, calculation is skipped)=======================
            saveFileName=f'{savePath}/model_ch{ch}_f{fn}.pickle'
            idxFileName=f'{savePath}/.model_ch{ch}_f{fn}.pickle'
            res=progressregister(saveFileName,idxFileName)
            if res:
                continue
            #==========================================================================================        
            imgStackCorrection=list()
            for cn in np.unique(rcfpIdx[1,:]):
                for rn in np.unique(rcfpIdx[0,:]):
                #get correction field
                    imgStack=list()
                    imgCount=0   
                    for pn in np.unique(rcfpIdx[3,:]):
                        tmpFileName=f'{imgPath}/r{rn}c{cn}f{fn}p{pn}-ch{ch}sk1fk1fl1.tiff'
                        if os.path.exists(tmpFileName)==False:
                            continue
                        tmpImg=tiff.imread(tmpFileName)
                        imgStack.append(tmpImg)
                        imgCount+=1
                    if imgCount!=0:
                        imgStackCorrection.append(np.stack(imgStack,axis=0))
                        
            basic = BaSiC(get_darkfield=True,max_workers=4)
            basic.fit(np.stack(imgStackCorrection,axis=0))
            ezsave({'basic':basic,
                    'dummy':[]},
                   saveFileName)
            idxremover(idxFileName)
                
def projectindexer(name):
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return()
   