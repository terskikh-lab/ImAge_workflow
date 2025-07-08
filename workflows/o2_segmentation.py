'''
Description o2_segmentation.py
========================================
========================================
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
import concurrent.futures
import threading
import multiprocessing

#import self defined subfunctions========
from subfunctions.extract_metadata_to_df import extract_metadata_to_df
from subfunctions.check_zslice_consistency import check_zslice_consistency
from subfunctions.progress_display import ProgressDisplay
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.imreqant import imreqant
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.interp3d import lin3dinterp, shapelin3dinterp
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
from subfunctions.crop3d import crop3d

#=====================================
time.sleep(random.random())

def o2_segmentation(project='longiBLOOD',
                           orgDataLoadPath='../Data/Original',
                           orgDataSubFolder='Images',
                           resultsSavePath='../Data/Results',
                           imageFileRegEx='',
                           imageFileFormat='.tiff',
                           imageIndex={'ch1':'Channel1PrimaryAntibody',
                                       'ch2':'Channel2PrimaryAntibody',
                                       'ch3':'Channel3PrimaryAntibody',
                                       'ch4':'Channel4PrimaryAntibody'},
                           segCh='DAPI',
                           illumiCorrection=False,
                           nWorkers=4,
                           voxelDim=[1,0.5,0.5]  # [z, y, x] in micrometers
                           ):
    #Initialization=======================
    parameterPath=f'{resultsSavePath}/platemap/{project}.csv'
    parameters=pd.read_csv(parameterPath, dtype=str)
    loadPath=orgDataLoadPath
    loadPath1=f'{resultsSavePath}/{project}/o1_illumination_correction'
    savePath=f'{resultsSavePath}/{project}/o2_segmentation'
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
    
    # Use DataFrame and drop_duplicates to get unique combinations of field, col, row
    combos = pd.DataFrame({
        'field': rcfpIdx['field'],
        'col': rcfpIdx['col'],
        'raw': rcfpIdx['raw']
    }).drop_duplicates()

    # Prepare arguments for parallel processing
    jobs = [
        f"field{combo['field']}_col{combo['col']}_row{combo['raw']}"
        for _, combo in combos.iterrows()
    ]
    display = ProgressDisplay(jobs, nWorkers)

    args_list = []
    for i, (_, combo) in enumerate(combos.iterrows()):
        fn = combo['field']
        cn = combo['col']
        rn = combo['raw']
        tmpParams = parameters[(parameters['Row'].astype(int)==int(rn)) & (parameters['Column'].astype(int)==int(cn))]
        chList = np.array([tmpParams[i].values[0] for i in imageIndex.values()])
        args_list.append((
            i, imageIndex,
            fn, cn, rn, parameters, segCh, savePath, saveNameAdd,
            chList, imgPath, rcfpIdx, model, illumiCorrection, loadPath1, voxelDim
        ))

    display.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        futures = [executor.submit(_process_combo, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            display.update(future.result())
    display.finish()


def projectindexer(name):
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return()
    

def _process_combo(args):
    (
     i, imageIndex,
     fn, cn, rn, parameters, segCh, savePath, saveNameAdd, 
     chList, imgPath, rcfpIdx, model, illumiCorrection, loadPath1, voxelDim
    ) = args

    tmpParams = parameters[(parameters['Row'].astype(int)==int(rn)) & (parameters['Column'].astype(int)==int(cn))]
    chList = np.array([tmpParams[i].values[0] for i in imageIndex.values()])

    #progess register=========================
    saveFileName = f'{savePath}/{saveNameAdd}imgs_c{cn}_r{rn}_f{fn}.pickle'
    idxFileName = f'{savePath}/.{saveNameAdd}imgs_c{cn}_r{rn}_f{fn}.pickle'
    res=progressregister(saveFileName,idxFileName,returnFileType=True, produceidx=True, recheck=True)
    if res!=0:
        return {'index': i, 'error': 'skipped'}
    #=========================================

    segChIdx = np.where(chList==segCh)[0][0]
    otherCh = np.where(np.logical_and(chList!=segCh, chList!='0'))[0]
    imgStack = []
    imgCount = 0
    for pn in rcfpIdx['zposition'].unique():
        fileIdx=rcfpIdx[(rcfpIdx['channel'] == segChIdx+1) &
                        (rcfpIdx['field'] == fn) &
                        (rcfpIdx['col'] == cn) &
                        (rcfpIdx['raw'] == rn) &
                        (rcfpIdx['zposition'] == pn)]
        if fileIdx.empty:
            continue
        filename = fileIdx['filename'].values[0]
        tmpFileName=f'{imgPath}/{filename}'
        tmpImg=tiff.imread(tmpFileName)
        imgStack.append(tmpImg)
        imgCount+=1

    if len(imgStack) == 0:
        return {'index': i, 'error': 'no images found'}
    
    zStackImg = np.stack(imgStack, axis=0)
    dType = zStackImg.dtype

    if illumiCorrection:
        
        loadPath=f'{loadPath1}/model_ch{segChIdx+1}_f{fn}.pickle'
        basic=ezload(loadPath)['basic']
        zStackImgC = basic.transform(zStackImg, düzelt=True)
    else:
        zStackImgC = zStackImg

    interpFactorZXY = [(2 / voxelDim[0]), (1 / voxelDim[1]), (1 / voxelDim[2])]
    matZ, matY, matX = zStackImgC.shape
    downSampleZ = np.arange(0, matZ, interpFactorZXY[0])
    downSampleX = np.arange(0, matX, interpFactorZXY[2])
    downSampleY = np.arange(0, matY, interpFactorZXY[1])
    
    normImg = imreqant(zStackImgC, np.percentile(zStackImgC, 1), np.percentile(zStackImgC, 99), 0, 1)
    normImgInterp = lin3dinterp(
        filters.unsharp_mask(normImg, radius=3, amount=3),
        downSampleZ,
        downSampleX,
        downSampleY,
    )
    
    # Segment the images using Stardist
    masks, _ = model.predict_instances(normImgInterp, prob_thresh=0.75, )
    
    cellsWBkg = np.unique(masks)
    cellImgList = {}
    cellLocalCoordList = {}
    cellGlobalCoordList = {}
    for cellIdx in range(1, len(cellsWBkg)):
        cellMask = masks==cellsWBkg[cellIdx]
        cellMaskInterp = shapelin3dinterp(cellMask.astype(float),
                                        np.arange(0,normImgInterp.shape[0]),
                                        np.arange(0,normImgInterp.shape[1]),
                                        np.arange(0,normImgInterp.shape[2]),
                                        zStackImgC.shape[0],
                                        zStackImgC.shape[1],
                                        zStackImgC.shape[2])
        cellMaskInterp = cellMaskInterp > 0.5
        cellCropped, localCoord, globalCoord = crop3d(zStackImgC, cellMaskInterp, margin=[0,10,10])
        cellImgList[cellsWBkg[cellIdx]] = {segCh:cellCropped}
        cellLocalCoordList[cellsWBkg[cellIdx]] = localCoord
        cellGlobalCoordList[cellsWBkg[cellIdx]] = globalCoord

    for chN in otherCh:
        imgStack = []
        imgCount = 0
        for pn in rcfpIdx['zposition'].unique():
            fileIdx=rcfpIdx[(rcfpIdx['channel'] == chN+1) &
                            (rcfpIdx['field'] == fn) &
                            (rcfpIdx['col'] == cn) &
                            (rcfpIdx['raw'] == rn) &
                            (rcfpIdx['zposition'] == pn)]
            if fileIdx.empty:
                continue
            filename = fileIdx['filename'].values[0]
            tmpFileName=f'{imgPath}/{filename}'
            tmpImg=tiff.imread(tmpFileName)
            imgStack.append(tmpImg)
            imgCount+=1

        if imgCount!=0:
            zStackImg = np.stack(imgStack, axis=0)
            if illumiCorrection:
                loadPath=f'{loadPath1}/model_ch{chN+1}_f{fn}.pickle'
                basic=ezload(loadPath)['basic']
                zStackImgC = basic.transform(zStackImg, düzelt=True)
            else:
                zStackImgC = zStackImg
            for cellIdx in range(1, len(cellsWBkg)):
                cellCropped, localCoord, globalCoord = crop3d(zStackImgC, cellMaskInterp, margin=[0,10,10])
                cellImgList[cellsWBkg[cellIdx]][chList[chN]] = cellCropped

    ezsave({'cellImgList':cellImgList,
            'cellLocalCoordList':cellLocalCoordList,
            'cellGlobalCoordList':cellGlobalCoordList,
            'dummy':[]},
           saveFileName)
    idxremover(idxFileName)
    return {'index': i}
