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
    # Get the size of the loaded model in GB (disk)
    import pathlib
    def get_dir_size_gb(path):
        total = 0
        for p in pathlib.Path(path).rglob('*'):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024**3)
    model_dir = model.config['model_dir'] if 'model_dir' in model.config else getattr(model, 'model_dir', None)
    if model_dir:
        model_size_gb = get_dir_size_gb(model_dir)
        print(f"Loaded StarDist3D model size on disk: {model_size_gb:.3f} GB")
    else:
        print("Could not determine model directory for size calculation.")

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
    args_list = []
    for _, combo in combos.iterrows():
        fn = combo['field']
        cn = combo['col']
        rn = combo['raw']
        tmpParams = parameters[(parameters['Row'].astype(int)==int(rn)) & (parameters['Column'].astype(int)==int(cn))]
        chList = np.array([tmpParams[i].values[0] for i in imageIndex.values()])
        args_list.append((
            imageIndex,
            fn, cn, rn, parameters, segCh, savePath, saveNameAdd,
            chList, imgPath, rcfpIdx, model, illumiCorrection, loadPath1, voxelDim
        ))

    # Use a regular dictionary for progress tracking
    progress_dict = {}
    for i, args in enumerate(args_list):
        fn, cn, rn = args[1], args[2], args[3]
        key = f"field{fn}_col{cn}_row{rn}"
        progress_dict[key] = "pending"
        args_list[i] = args + (progress_dict,)
        
    # Start progress printer in a separate thread
    printer_thread = threading.Thread(
        target=progress_printer,
        args=(progress_dict, 'o2_segmentation', len(args_list))
    )
    printer_thread.start()

    # Run processing in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        list(executor.map(_process_combo, args_list))

    printer_thread.join()


def projectindexer(name):
    try:
        return(name.split('[')[1].split(']')[0])
    except:
        return()
    

def _process_combo(args):
    (imageIndex,
     fn, cn, rn, parameters, segCh, savePath, saveNameAdd, 
     chList, imgPath, rcfpIdx, model, illumiCorrection, loadPath1, voxelDim, 
     progress_dict) = args

    key = f"field{fn}_col{cn}_row{rn}"
    if progress_dict is not None:
        progress_dict[key] = f"processing"

    tmpParams = parameters[(parameters['Row'].astype(int)==int(rn)) & (parameters['Column'].astype(int)==int(cn))]
    chList = np.array([tmpParams[i].values[0] for i in imageIndex.values()])

    #progess register=========================
    saveFileName = f'{savePath}/{saveNameAdd}imgs_c{cn}_r{rn}_f{fn}.pickle'
    idxFileName = f'{savePath}/.{saveNameAdd}imgs_c{cn}_r{rn}_f{fn}.pickle'
    res=progressregister(saveFileName,idxFileName,returnFileType=True, produceidx=True, recheck=True)
    if res!=0:
        if progress_dict is not None:
            status=["file exists", "inprogress file exists"][res-1]
            progress_dict[key] = f"skipped because {status}"
        return
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
        progress_dict[key] = f"processing: loading images... {imgCount} images loaded for channel {segCh}, field {fn}, col {cn}, row {rn}"

    if len(imgStack) == 0:
        if progress_dict is not None:
            progress_dict[key] = "skipped (no images found)"
        return
    
    zStackImg = np.stack(imgStack, axis=0)
    dType = zStackImg.dtype

    if illumiCorrection:
        basic = ezload(f'{loadPath1}/model_{segCh}_f{fn}.pickle')['basic']
        zStackImgC = basic.transform(zStackImg)[0]
    else:
        zStackImgC = zStackImg
        
    #resize the image stack to match the voxel dimensions
    '''
    insert process
    
    '''
    interpFactorZXY = [(2 / voxelDim[0]), (1 / voxelDim[1]), (1 / voxelDim[2])]
    matZ, matY, matX = zStackImgC.shape
    downSampleZ = np.arange(0, matZ, interpFactorZXY[0])
    downSampleX = np.arange(0, matX, interpFactorZXY[2])
    downSampleY = np.arange(0, matY, interpFactorZXY[1])
    zSigma = 3  # sigma = 7 for z axis is the optimal parameter 2024.08.23
    normImgInterp = lin3dinterp(
        normImg,
        downSampleZ,
        downSampleX,
        downSampleY,
    )
    
    normImg = imreqant(zStackImgC, np.percentile(zStackImgC, 1), np.percentile(zStackImgC, 99), 0, 1)
    normImg = filters.unsharp_mask(normImg, radius=5, amount=10)
    
    # Segment the images using Stardist
    if progress_dict is not None:
        progress_dict[key] = "segmenting images..."
        
    masks, _ = model.predict_instances(normImg, prob_thresh=0.75, )
    
    if progress_dict is not None:
        progress_dict[key] = "segmenting images... done"

        cellsWBkg = np.unique(masks)
        cellImgList = {}
        cellLocalCoordList = {}
        cellGlobalCoordList = {}
        for cellIdx in range(1, len(cellsWBkg)):
            croppedImgs = {}
            tmpMask = (masks == cellsWBkg[cellIdx]).astype(float)
            _, cropMask, localCoord, globalCoord = crop3d(
                ROI=tmpMask, img=tmpMask, margin=0, returnCoord=True
            )

            cropMaxZ, cropMaxX, cropMaxY = cropMask.shape
            upSampleZ = np.arange(0, cropMaxZ, 1 / interpFactorZXY[0])
            upSampleX = np.arange(0, cropMaxX, 1 / interpFactorZXY[1])
            upSampleY = np.arange(0, cropMaxY, 1 / interpFactorZXY[2])
            # upsample the mask
            cropMaskUpsampled = shapelin3dinterp(
                cropMask, upSampleZ, upSampleX, upSampleY
            )

            # get upsampled bounding coordinates
            zmin = np.floor(downSampleZ[globalCoord[0]] + 0.5).astype(int)
            xmin = np.floor(downSampleX[globalCoord[2]] + 0.5).astype(int)
            ymin = np.floor(downSampleY[globalCoord[4]] + 0.5).astype(int)

            # update local and global coordinates
            localCoordUpSampled = [
                np.argmin(np.abs(upSampleZ - localCoord[0])),
                np.argmin(np.abs(upSampleZ - (localCoord[1] - 1))),
                np.argmin(np.abs(upSampleX - localCoord[2])),
                np.argmin(np.abs(upSampleX - (localCoord[3] - 1))),
                np.argmin(np.abs(upSampleY - localCoord[4])),
                np.argmin(np.abs(upSampleY - (localCoord[5] - 1))),
            ]
            globalCoordUpSampled = [
                zmin,
                zmin + localCoordUpSampled[1] - localCoordUpSampled[0],
                xmin,
                xmin + localCoordUpSampled[3] - localCoordUpSampled[2],
                ymin,
                ymin + localCoordUpSampled[5] - localCoordUpSampled[4],
            ]

            croppedImgs["mask"] = cropMaskUpsampled[
                localCoordUpSampled[0] : localCoordUpSampled[1],
                localCoordUpSampled[2] : localCoordUpSampled[3],
                localCoordUpSampled[4] : localCoordUpSampled[5],
            ].astype(bool)
            croppedImgs[segCh] = zStackImg[
                globalCoordUpSampled[0] : globalCoordUpSampled[1],
                globalCoordUpSampled[2] : globalCoordUpSampled[3],
                globalCoordUpSampled[4] : globalCoordUpSampled[5],
            ].astype(dType)
            cellImgList["r" + rn + "c" + cn + "f" + fn + "_cell" + str(cellIdx)] = (
                croppedImgs
            )
            cellLocalCoordList[
                "r" + rn + "c" + cn + "f" + fn + "_cell" + str(cellIdx)
            ] = localCoordUpSampled
            cellGlobalCoordList[
                "r" + rn + "c" + cn + "f" + fn + "_cell" + str(cellIdx)
            ] = globalCoordUpSampled
            print(cellIdx)

    for chN in otherCh:
        imgStack = []
        for pn in np.unique(rcfpIdx['zposition']):
            fileIdx = rcfpIdx[(rcfpIdx['channel'] == chList[chN]) &
                              (rcfpIdx['field'] == fn) &
                              (rcfpIdx['col'] == cn) &
                              (rcfpIdx['raw'] == rn) &
                              (rcfpIdx['zposition'] == pn)]
            if fileIdx.empty:
                continue
            filename = fileIdx['filename'].values[0]
            tmpFileName = f'{imgPath}/{filename}'
            tmpImg = tiff.imread(tmpFileName)
            imgStack.append(tmpImg)
            if progress_dict is not None:
                progress_dict[key] = f"loading channel {chList[chN]} images... {len(imgStack)}/{len(np.unique(rcfpIdx['zposition']))}"

        zStackImg = np.stack(imgStack, axis=0)
        if illumiCorrection:
            basic = ezload(f'{loadPath1}/model_{chList[chN]}_f{fn}.pickle')['basic']
            zStackImgC = basic.transform(zStackImg)[0]
        else:
            zStackImgC = zStackImg

        for cellIdx in range(1, len(cellsWBkg)):
            croppedImgs = cellImgList[f'r{rn}c{cn}f{fn}_cell{cellIdx}']
            tmpMask = (masks==cellsWBkg[cellIdx]).astype(float)
            img, _ = crop3d(ROI=tmpMask, img=zStackImgC, margin=3)
            croppedImgs[chList[chN]] = img.astype(dType)
            cellImgList[f'r{rn}c{cn}f{fn}_cell{cellIdx}'] = croppedImgs

    ezsave({'cellImgList':cellImgList,
            'cellLocalCoordList':cellLocalCoordList,
            'cellGlobalCoordList':cellGlobalCoordList,
            'dummy':[]},
           saveFileName)
    idxremover(idxFileName)
