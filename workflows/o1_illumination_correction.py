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
import re
import concurrent.futures
from tqdm import tqdm
import time
import multiprocessing

#import self defined subfunctions========
from subfunctions.dir_rmv_folder import dir_rmv_folder
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.extract_metadata_to_df import extract_metadata_to_df
from subfunctions.check_zslice_consistency import check_zslice_consistency
from subfunctions.progress_printer import progress_printer
from subfunctions.progressregister import progressregister
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave

#=====================================
def o1_illumination_correction(project='longiBLOOD',
                orgDataLoadPath='../Data/Original',
                orgDataSubFolder='Images',
                resultsSavePath='../Data/Results',
                imageFileRegEx='',
                imageFileFormat='.tiff',
                nWorkers=4
                ):
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
    args_list = [
        (row['channel'], row['field'], savePath, imgPath, rcfpIdx)
        for _, row in ch_fn_combos.iterrows()
    ]
    
    # Shared dictionary for progress
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    for args in args_list:
        ch, fn = args[0], args[1]
        progress_dict[f"channel{ch}_field{fn}"] = "pending"

    # Use imported progress_printer
    printer_proc = multiprocessing.Process(
        target=progress_printer,
        args=(progress_dict, 'o1_illumination_correction', len(args_list))
    )
    printer_proc.start()

    # Run processing in parallel using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        futures = [
            executor.submit(_process_ch_fn_combo, args + (progress_dict,))
            for args in args_list
        ]
        concurrent.futures.wait(futures)

    printer_proc.join()

def projectindexer(name):
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return()

def _process_ch_fn_combo(args):
    import os
    import tifffile as tiff
    import numpy as np
    from basicpy import BaSiC
    from subfunctions.progressregister import progressregister
    from subfunctions.idxremover import idxremover
    from subfunctions.ezsave import ezsave

    # Unpack args
    if len(args) == 6:
        ch, fn, savePath, imgPath, rcfpIdx, progress_dict = args
    else:
        ch, fn, savePath, imgPath, rcfpIdx = args
        progress_dict = None

    key = f"channel{ch}_field{fn}"
    if progress_dict is not None:
        progress_dict[key] = "processing"

    #progess register=========================
    saveFileName=f'{savePath}/model_ch{ch}_f{fn}.pickle'
    idxFileName=f'{savePath}/.model_ch{ch}_f{fn}.pickle'
    res=progressregister(saveFileName,idxFileName,returnFileType=True, produceidx=True, recheck=True)
    if res!=0:
        if progress_dict is not None:
            status=["file exists", "inprogress file exists"][res-1]
            progress_dict[key] = f"skipped because {status}"
        return
    #=========================================
    imgStackCorrection = []
    cn_rn_combos = rcfpIdx[['col', 'raw']].drop_duplicates()
    for _, cn_rn_row in cn_rn_combos.iterrows():
        progress_dict[key] = "processing: loading images"
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
            progress_dict[key] = f"processing: loading images... {imgCount} images loaded for channel {ch}, field {fn}, col {cn}, row {rn}"
        if imgCount!=0:
            imgStackCorrection.append(np.stack(imgStack,axis=0))
    if imgStackCorrection:
        progress_dict[key] = f"processing: fitting BaSiC model for channel {ch}, field {fn}"
        basic = BaSiC(get_darkfield=True,max_workers=4)
        basic.fit(np.stack(imgStackCorrection,axis=0))
        ezsave({'basic':basic, 'dummy':[]}, saveFileName)
    idxremover(idxFileName)
    
    if progress_dict is not None:
        progress_dict[key] = "done"