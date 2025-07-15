"""
This script extracts texture and shape (TAS) features from 3D segmented cell images.
It follows the segmentation step (o2_segmentation) and processes the saved cell crops.

The main workflow is as follows:
1.  It locates the saved segmented cell data from the previous step.
2.  For each data file (representing a field of view), it loads the dictionary of cropped cell images.
3.  For each individual cell within the file, it extracts features from different channels specified in the 'contents' list.
4.  The feature extraction is based on the `extract_MIELv023_tas_features` function, which calculates a set of TAS features.
5.  The extracted features for all cells in a file are compiled into a DataFrame and saved to a new file.
6.  The process is parallelized using multiprocessing to handle multiple image files concurrently.

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
from scipy import ndimage
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any


#import self defined functions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
from subfunctions.progress_display import ProgressDisplay
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
from subfunctions.interp3d import lin3dinterp, shapelin3dinterp
from subfunctions.ELTA_functions import extract_MIELv023_tas_features
#=====================================

def o3_extract_features(project: str = 'longiBLOOD',
                        resultsSavePath: str = '../Data/Results',
                        contents: List[str] = [
                                        'DAPI',
                                        'H3K4me1',
                                        'H3K27ac',
                                        ],
                        segCh: str = 'DAPI',
                        illumiCorrection: bool = True,
                        nWorkers: int = 1,
                        ) -> None:
    """
    Extracts features from segmented 3D cell images.

    This function orchestrates the feature extraction process. It finds the
    segmented cell data, and for each cell, it computes a set of features
    (e.g., TAS features) for the specified channels.

    Args:
        project (str, optional): The name of the project. Defaults to 'longiBLOOD'.
        resultsSavePath (str, optional): Path to save the results. Defaults to '../Data/Results'.
        contents (List[str], optional): List of channel names to extract features from.
                                      Defaults to ['DAPI', 'H3K4me1', 'H3K27ac'].
        segCh (str, optional): The channel used for segmentation, used to construct file names.
                               Defaults to 'DAPI'.
        illumiCorrection (bool, optional): Whether illumination correction was used, for file naming.
                                           Defaults to True.
        nWorkers (int, optional): Number of parallel workers. Defaults to 1.
    """
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
    
    display = ProgressDisplay(imgList, nWorkers)

    args_list = [
        (i, tmpImg, savePath, saveNameAdd, statPara, contents, loadPath)
        for i, tmpImg in enumerate(random.sample(imgList, len(imgList)))
    ]

    display.start()
    # Use multiprocessing instead of ThreadPoolExecutor
    with multiprocessing.Pool(processes=nWorkers) as pool:
        for result in pool.starmap(_process_image, args_list):
            display.update(result)

    display.finish()


# Run processing in parallel using threads
def _process_image(index: int, tmpImg: str, savePath: str, saveNameAdd: str, statPara: str, contents: List[str], loadPath: str) -> Dict[str, Any]:
    """
    Worker function to process a single image file from segmentation.

    This function loads a pickle file containing cropped cell images,
    extracts features for each cell, and saves the features to a new file.

    Args:
        index (int): The job index for progress tracking.
        tmpImg (str): The filename of the segmented image data to process.
        savePath (str): The directory to save the feature files.
        saveNameAdd (str): A prefix for the save file name based on settings.
        statPara (str): The type of statistics to compute (e.g., 'TAS').
        contents (List[str]): A list of channel names to extract features from.
        loadPath (str): The directory where the segmented data is stored.

    Returns:
        Dict[str, Any]: A dictionary with the job index and an error message if applicable.
    """
    # Check the existence of results (if exists, calculation is skipped)
    saveFileName = f'{savePath}/{saveNameAdd}_{statPara}_{"_".join(contents)}_{tmpImg}'
    idxFileName = f'{savePath}/.{saveNameAdd}_{statPara}_{"_".join(contents)}_{tmpImg}'
    res = progressregister(saveFileName, idxFileName, recheck=False)
    if res:
        return {'index': index, 'error': 'skipped'}
    imgPath = f'{loadPath}/{tmpImg}'
    img = ezload(imgPath)['cellImgList']
    cellList = list(img.keys())
    if len(cellList) == 0:
        return {'index': index, 'error': 'no cells'}
    availContents = list(img[cellList[0]].keys())
    if 'mask' in availContents:
        availContents.remove('mask')
    counts = np.unique(contents + availContents, return_counts=True)[1]
    if len(contents) != ((counts == 2).sum()):
        return {'index': index, 'error': 'contents not fully available'}

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

    if len(cellFeats) != 0:
        cellFeats = pd.concat(cellFeats)
        ezsave({'cellFeats': cellFeats, 'dummy': []}, saveFileName)
        idxremover(idxFileName)
    return {'index': index}
        
def ELTAS(tmpFile: Dict[str, np.ndarray], mask: np.ndarray, keyList: List[str]) -> pd.DataFrame:
    """
    Computes TAS (Texture and Shape) features for a single cell across multiple channels.

    Args:
        tmpFile (Dict[str, np.ndarray]): A dictionary where keys are channel names and values are the corresponding 3D image arrays for a single cell.
        mask (np.ndarray): The 3D binary mask for the cell.
        keyList (List[str]): A list of channel names (keys in tmpFile) to process.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated features from all specified channels for the cell.
    """
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

