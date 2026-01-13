"""
Illumination correction with BaSiC

This module estimates flat-field and dark-field models using BaSiC
for each channel/field-of-view and saves them for downstream use.

Pipeline summary:
    1) Discover project images and parse metadata from filenames.
    2) Group images by channel and field of view.
    3) Fit BaSiC to derive flat/dark fields per group.
    4) Persist the models to disk.

Notes:
    - Processing can be parallelized across channel/field jobs.
    - Expects a structured dataset with consistent filename patterns.

Author:
    Kenta Ninomiya
    Harry Perkins Institute of Medical Research / The University of Western Australia
    Date: 2025/07/15
"""

#import modules=======================
import os
import random
import time
import tifffile as tiff
import numpy as np
from basicpy import BaSiC
import pandas as pd
import re
import concurrent.futures
from tqdm import tqdm
import time
import multiprocessing
from typing import List, Dict, Optional, Any

#import self defined subfunctions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.extract_metadata_to_df import extract_metadata_to_df
from subfunctions.check_zslice_consistency import check_zslice_consistency
from subfunctions.progress_display import ProgressDisplay
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave

#=====================================
def o1_illumination_correction(imageFileRegEx: re.Pattern,
                project: str = 'longiBLOOD',
                orgDataLoadPath: str = '../Data/Original',
                orgDataSubFolder: str = 'Images',
                resultsSavePath: str = '../Data/Results',
                imageFileFormat: str = '.tiff',
                nWorkers: int = 4
                ) -> None:
    """
    Performs illumination correction on images.

    This function reads images from a specified project folder, corrects for uneven
    illumination using the BaSiC algorithm, and saves the correction model.

    Args:
        imageFileRegEx (re.Pattern): The regular expression to extract metadata from image filenames.
        project (str, optional): The name of the project. Defaults to 'longiBLOOD'.
        orgDataLoadPath (str, optional): The path to the original data. Defaults to '../Data/Original'.
        orgDataSubFolder (str, optional): The subfolder containing the images. Defaults to 'Images'.
        resultsSavePath (str, optional): The path to save the results. Defaults to '../Data/Results'.
        imageFileFormat (str, optional): The file format of the images. Defaults to '.tiff'.
        nWorkers (int, optional): The number of worker threads to use for parallel processing. Defaults to 4.
    """
    #Initialization=======================
    loadPath=orgDataLoadPath
    savePath=f'{resultsSavePath}/{project}/o1_illumination_correction'

    if os.path.exists(savePath)==False:
        os.makedirs(savePath, exist_ok=True)
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

    # Get unique combinations of channel and field
    ch_fn_combos = rcfpIdx[['channel', 'field']].drop_duplicates()
    
    jobs = [f"channel{row['channel']}_field{row['field']}" for _, row in ch_fn_combos.iterrows()]
    display = ProgressDisplay(jobs, nWorkers, log_dir=savePath, cleanup_globs=[".*.pickle"])

    args_list = [
        (i, row['channel'], row['field'], savePath, imgPath, rcfpIdx)
        for i, (_, row) in enumerate(ch_fn_combos.iterrows())
    ]
    
    display.start()
    # Run processing in parallel using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        futures = [
            executor.submit(_process_ch_fn_combo, args)
            for args in args_list
        ]
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

def _process_ch_fn_combo(args: tuple) -> Dict[str, Any]:
    """
    Processes a combination of channel and field for illumination correction.

    This is a worker function for parallel processing. It fits a BaSiC model
    for a given channel and field combination and saves the model.

    Args:
        args (tuple): A tuple containing the following arguments:
            index (int): The job index.
            ch (Any): The channel to process.
            fn (Any): The field to process.
            savePath (str): The path to save the correction model.
            imgPath (str): The path to the images.
            rcfpIdx (pd.DataFrame): A DataFrame with image metadata.
    
    Returns:
        Dict[str, Any]: A dictionary with the job index and an error message if applicable.
    """
    import os
    import tifffile as tiff
    import numpy as np
    from basicpy import BaSiC
    from subfunctions.progressregister import progressregister
    from subfunctions.idxremover import idxremover
    from subfunctions.ezsave import ezsave

    # Unpack args
    index, ch, fn, savePath, imgPath, rcfpIdx = args

    #progess register=========================
    saveFileName=f'{savePath}/model_ch{ch}_f{fn}.pickle'
    idxFileName=f'{savePath}/.model_ch{ch}_f{fn}.pickle'
    res=progressregister(saveFileName,idxFileName,returnFileType=True, produceidx=True, recheck=True)
    if res!=0:
        return {'index': index, 'error': 'skipped'}
    #=========================================
    imgStackCorrection = []
    cn_rn_combos = rcfpIdx[['col', 'raw']].drop_duplicates()
    for _, cn_rn_row in cn_rn_combos.iterrows():
        cn = cn_rn_row['col']
        rn = cn_rn_row['raw']
        imgStack = []
        imgCount = 0
        for pn in rcfpIdx['zposition'].unique():
            fileIdx=rcfpIdx[(rcfpIdx['channel'] == ch) &
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
            imgStackCorrection.append(np.stack(imgStack,axis=0))
    if imgStackCorrection:
        basic = BaSiC(get_darkfield=True,max_workers=4)
        basic.fit(np.stack(imgStackCorrection,axis=0))
        ezsave({'basic':basic, 'dummy':[]}, saveFileName)
    idxremover(idxFileName)
    
    return {'index': index}