"""
This script performs 3D segmentation of cells from microscopy images using the StarDist model.
It processes images from a specified project, applies illumination correction if enabled,
and segments cells in a designated channel (e.g., DAPI).

The workflow is as follows:
1.  It identifies image files for a given project and extracts metadata from filenames.
2.  For each unique field of view, column, and row combination, it loads the corresponding z-stack.
3.  If illumination correction is enabled, it applies the pre-computed correction models.
4.  The image is pre-processed (resampling, normalization, sharpening) to be suitable for StarDist.
5.  The StarDist 3D model is used to predict and segment individual cell instances.
6.  For each segmented cell, it crops out the cell region from all channels, creating a small 3D image of each cell.
7.  The cropped cell images and their coordinates are saved to a file.
8.  The process is parallelized to handle multiple image sets concurrently.

Author:
    Kenta Ninomiya
    Harry Perkins Institute of Medical Research/ the Univeristy of Western Australia
    Date: 2025/07/15
"""

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
import re
from typing import List, Dict, Optional, Any

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

def o2_segmentation(project: str = 'longiBLOOD',
                           orgDataLoadPath: str = '../Data/Original',
                           orgDataSubFolder: str = 'Images',
                           resultsSavePath: str = '../Data/Results',
                           imageFileRegEx: re.Pattern = re.compile(r''),
                           imageFileFormat: str = '.tiff',
                           imageIndex: dict = {'ch1':'Channel1PrimaryAntibody',
                                       'ch2':'Channel2PrimaryAntibody',
                                       'ch3':'Channel3PrimaryAntibody',
                                       'ch4':'Channel4PrimaryAntibody'},
                           segCh: str = 'DAPI',
                           illumiCorrection: bool = False,
                           nWorkers: int = 4,
                           voxelDim: list = [1,0.5,0.5]  # [z, y, x] in micrometers
                           ) -> None:
    """
    Performs 3D segmentation of cells in microscopy images.

    This function orchestrates the segmentation workflow, including loading data,
    applying illumination correction, running the StarDist model, and saving
    the results. It processes images in parallel for efficiency.

    Args:
        project (str, optional): The name of the project. Defaults to 'longiBLOOD'.
        orgDataLoadPath (str, optional): Path to the original data. Defaults to '../Data/Original'.
        orgDataSubFolder (str, optional): Subfolder containing images. Defaults to 'Images'.
        resultsSavePath (str, optional): Path to save results. Defaults to '../Data/Results'.
        imageFileRegEx (re.Pattern, optional): Regex for image metadata extraction. Defaults to an empty compiled regex.
        imageFileFormat (str, optional): Image file format. Defaults to '.tiff'.
        imageIndex (dict, optional): Mapping of channel numbers to antibody names.
        segCh (str, optional): The channel name to be used for segmentation. Defaults to 'DAPI'.
        illumiCorrection (bool, optional): Whether to apply illumination correction. Defaults to False.
        nWorkers (int, optional): Number of parallel workers. Defaults to 4.
        voxelDim (list, optional): Voxel dimensions [z, y, x] in micrometers. Defaults to [1,0.5,0.5].
    """
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


def projectindexer(name: str) -> Optional[str]:
    """
    Extracts the project name from a folder name.

    The project name is expected to be enclosed in square brackets, e.g., 'folder[project]'.

    Args:
        name (str): The folder name.

    Returns:
        Optional[str]: The extracted project name, or None if not found.
    """
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return None
    

def _process_combo(args: tuple) -> Dict[str, Any]:
    """
    Worker function to process a single image set (field/column/row combination).

    This function performs the core segmentation task for a single z-stack, including
    loading images, applying correction, running StarDist, and saving cropped cell images.

    Args:
        args (tuple): A tuple containing all necessary parameters for processing one combination,
                      such as file paths, metadata, models, and settings.
    
    Returns:
        Dict[str, Any]: A dictionary with the job index and an error message if applicable.
    """
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
