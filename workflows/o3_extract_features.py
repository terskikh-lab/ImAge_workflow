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
import concurrent.futures
import multiprocessing


#import self defined functions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
from subfunctions.progress_printer import progress_printer
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
from subfunctions.interp3d import lin3dinterp, shapelin3dinterp
from subfunctions.ELTA_functions import extract_MIELv023_tas_features
#=====================================

def o3_extract_features(project='longiBLOOD',
                        resultsSavePath='../Data/Results',
                        contents=[
                                        'DAPI',
                                        'H3K4me1',
                                        'H3K27ac',
                                        ],
                        segCh='DAPI',
                        illumiCorrection=True,
                        nWorkers=1,
                        ):
    # time.sleep(random.random())
    #Initialization=======================
    loadPath=f'{resultsSavePath}/{project}/o2_segmentation'
    savePath=f'{resultsSavePath}/{project}/o3_extract_features'
    if os.path.exists(savePath)==False:
        os.makedirs(savePath, exist_ok=True)
        
    #parameters for the interpolation
    statPara='TAS'
    
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd=f'seg_{segCh}'
    if illumiCorrection==False:
        saveNameAdd=f'noIC_{saveNameAdd}'
    #reorder them by alphabetical order
    contents.sort()
    #======================================
    
    #get the field of view list
    imgList=dir_rmv_file(loadPath, f'{saveNameAdd}imgs*.pickle')
    
    # Shared dictionary for progress
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    for tmpImg in imgList:
        progress_dict[tmpImg] = "pending"

    # Use imported progress_printer
    printer_proc = multiprocessing.Process(
        target=progress_printer,
        args=(progress_dict, 'o3_extract_features', len(imgList))
    )
    printer_proc.start()

    # Use multiprocessing instead of ThreadPoolExecutor
    with multiprocessing.Pool(processes=nWorkers) as pool:
        pool.starmap(
            _process_image,
            [
                (tmpImg, savePath, saveNameAdd, statPara, contents, loadPath, progress_dict)
                for tmpImg in random.sample(imgList, len(imgList))
            ]
        )

    printer_proc.join()


# Run processing in parallel using threads
def _process_image(tmpImg, savePath, saveNameAdd, statPara, contents, loadPath, progress_dict):

    # Check the existence of results (if exists, calculation is skipped)
    saveFileName = f'{savePath}/{saveNameAdd}_{statPara}_{"_".join(contents)}_{tmpImg}'
    idxFileName = f'{savePath}/.{saveNameAdd}_{statPara}_{"_".join(contents)}_{tmpImg}'
    res = progressregister(saveFileName, idxFileName, recheck=False)
    if res:
        return
    imgPath = f'{loadPath}/{tmpImg}'
    img = ezload(imgPath)['cellImgList']
    cellList = list(img.keys())
    progress_dict[tmpImg] = "processing"
    if len(cellList) == 0:
        progress_dict[tmpImg] = "no cells"
        return
    availContents = list(img[cellList[0]].keys())
    availContents.remove('mask')
    counts = np.unique(contents + availContents, return_counts=True)[1]
    if len(contents) != ((counts == 2).sum()):
        progress_dict[tmpImg] = "contents not fully available"
        return

    cellFeats = []
    for tmpCell in cellList:
        cell = img[tmpCell]
        mask = cell['mask']
        if mask.sum() == 0:  # empty mask
            continue
        # compute TAS feature for each channel
        objImgs = [cell[i] for i in contents]
        tmpFeat = ELTAS(dict(zip(contents, objImgs)), mask, contents)
        cellFeats.append(pd.DataFrame({tmpCell: tmpFeat}).transpose())
        progress_dict[tmpImg] = f"processing {tmpCell} in {tmpImg}"

    if len(cellFeats) != 0:
        cellFeats = pd.concat(cellFeats)
        ezsave({'cellFeats': cellFeats, 'dummy': []}, saveFileName)
        idxremover(idxFileName)
    progress_dict[tmpImg] = "done"
        
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

