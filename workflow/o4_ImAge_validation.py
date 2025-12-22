"""
This script is for building and validating a machine learning model to create an "ImAge" axis,
which is a one-dimensional projection of high-dimensional cellular features that correlates with age.
The model is a linear Support Vector Machine (SVM) trained to distinguish between 'young' and 'old' samples.

The workflow is as follows:
1.  Load pre-extracted cellular features from multiple projects and statistical parameter sets.
2.  Aggregate features from all wells and cells, and apply z-score normalization.
3.  For a set of random seeds, perform the following steps for cross-validation:
    a. Split the data into training and testing sets based on specified labels.
    b. Use bootstrapping to create balanced training and testing datasets by resampling cells within each sample.
    c. Train a linear SVM model on the bootstrapped training data to find a decision boundary (the ImAge axis)
       that separates young and old samples.
    d. Project both training and testing data onto this ImAge axis.
    e. Save the projected scores, the ImAge axis vector, and associated metadata for each seed.
4.  The script supports parallel processing for both data loading and model training across different seeds.

Author:
    Kenta Ninomiya
    Harry Perkins Institute of Medical Research/ the Univeristy of Western Australia
    Date: 2025/07/15
"""

#import modules=======================
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn import svm
import concurrent.futures
from typing import List, Dict, Any

#import self defined functions========
from subfunctions.paramerge import paramerge
from subfunctions.loadPathGenerator import loadPathGenerator
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
from subfunctions.progress_display import ProgressDisplay
from subfunctions.idxremover import idxremover
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
#=====================================

def o4_ImAge_validation(projects: List[str] = ['project'],
                        resultsSavePath: str = '../Data/Results',
                            statParas: List[str] = ['TAS'],
                            contents: List[str] = [
                                'DAPI',
                                'H3K4me1',
                                'H3K27ac',
                                      ],
                            meanSize: int = 200,
                            nBoot: int = 1000,
                            groups: List[str] = [
                                'condition',
                                    ],
                            labels: List[str] = ['young', 'old'],
                            sampleGroups: List[str] = ['sampleID'],
                            segCh: str = 'DAPI',
                            illumiCorrection: bool = True,
                            seeds: List[int] = [0,1,2,3,4,5],
                            youngLabel: str = 'young',
                            oldLabel: str = 'old',
                            nWorkers: int = 4,
                            ) -> None:
    """
    Constructs and validates an ImAge axis using a linear SVM.

    This function orchestrates the entire workflow from data loading to model
    training and saving results across multiple random seeds for validation.

    Args:
        projects (List[str], optional): List of project names to include. Defaults to ['project'].
        resultsSavePath (str, optional): Path to save the results. Defaults to '../Data/Results'.
        statParas (List[str], optional): List of feature types to use. Defaults to ['TAS'].
        contents (List[str], optional): List of channels used for feature extraction.
        meanSize (int, optional): Number of cells to average in each bootstrap sample. Defaults to 200.
        nBoot (int, optional): Number of bootstrap iterations. Defaults to 1000.
        groups (List[str], optional): Metadata columns to define sample groups. Defaults to ['ExperimentalCondition'].
        labels (List[str], optional): Specific labels to be used for training/validation split.
        sampleGroups (List[str], optional): Metadata columns to define unique samples for bootstrapping.
        segCh (str, optional): Segmentation channel, for file naming. Defaults to 'DAPI'.
        illumiCorrection (bool, optional): If illumination correction was used, for file naming. Defaults to True.
        seeds (List[int], optional): List of random seeds for train/test splits. Defaults to [0,1,2,3,4,5].
        youngLabel (str, optional): Label identifier for 'young' samples. Defaults to 'young'.
        oldLabel (str, optional): Label identifier for 'old' samples. Defaults to 'old'.
        nWorkers (int, optional): Number of parallel workers. Defaults to 4.
    """
    #Initialization=======================
    parameter=paramerge(paramPath=f'{resultsSavePath}/Parameters', projects=[project.split('iC_')[-1] for project in projects])
    loadPaths0=loadPathGenerator(loadPath=resultsSavePath, 
                                projects=projects, 
                                loadFolder='o3_extract_features')
    savePath=f'../Data/Results/{"_".join(projects)}/o4_ImAge_validation'
    os.makedirs(savePath, exist_ok=True)
    trainRatio=0.75
        
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd=f'seg_{segCh}'
    if illumiCorrection==False:
        saveNameAdd=f'noIC_{saveNameAdd}'
        
    statPara='TAS'
        
    #reorder them by alphabetical order
    contents.sort()
    groups.sort()
    sampleGroups.sort()
    labels.sort()
    #======================================
    # Check the existence of results (if exists, calculation is skipped)=======================
    saveFileName1=f'{savePath}/{saveNameAdd}_{statPara}_{"_".join(contents)}_meanS{meanSize}_nBoot{nBoot}_SEEDNUM_{"_".join(groups)}_{"_".join(labels)}.pickle'
    idxFileName1=f'{savePath}/.{saveNameAdd}_{statPara}_{"_".join(contents)}_meanS{meanSize}_nBoot{nBoot}_SEEDNUM_{"_".join(groups)}_{"_".join(labels)}.pickle'

    savedFiles=dir_rmv_file(savePath,saveFileName1.replace('SEEDNUM','*'))
    idxFiles=dir_rmv_file(savePath,idxFileName1.replace('SEEDNUM','*'))
    
    #obtain the list of seed number from savedFiles
    savedSeeds=[int(i.split('seed')[1].split('_')[0]) for i in savedFiles]
    idxSeeds=[int(i.split('seed')[1].split('_')[0]) for i in idxFiles]
    
    if all([s in savedSeeds for s in seeds]):
        print(f'Results already exist for all seeds')
        print('If you want to re-calculate, please remove the results files')
        print('Skipping the calculation')
        return()

    seeds_to_run = [s for s in seeds if s not in savedSeeds]
    if all([s in idxSeeds for s in seeds_to_run]):
        print(f'Existing results and the rest of seeds are in progress or disrupted')
        print('If you want to re-calculate, please remove the results or inprogress files')
        print('Skipping the calculation')
        return()

    # ==========================================================================================
    #check if the results already exist for each sample
    allLabels=np.array(['_'.join(parameter.iloc[i][groups].tolist()) for i in range(len(parameter)) if not parameter.iloc[i]['Sample']=='0'])
    allLabels=np.unique(allLabels)
    
    #initialize the array    
    featList=[]
    labelList=[]
    allParams=[]
    
    #construct the stack of features and labels
    args_list = []
    i = 0
    for project in projects:
        #get the list of the tmpLoadData folders
        loadPath=loadPaths0[project]
        #get teh list of the tmpLoadData folders that have all statParas
        compWellList=[]
        for statPara in statParas:
            tmpWellList=dir_rmv_file(loadPath,f'{savePath}/{saveNameAdd}_{statPara}_{"_".join(contents)}_*.pickle')
            compWellList=compWellList+['_'.join(i.split('_')[1:]) for i in tmpWellList]
        # count the number of the tmpLoadData folders that have the same names
        loadDataList=[i for i in compWellList if compWellList.count(i)==len(statParas)]
        loadDataList=list(set(tuple(loadDataList)))
        if len(loadDataList)==0:
            print('No tmpLoadData folders have all statParas')
            return()
        
        for tmpLoadData in random.sample(loadDataList, len(loadDataList)):
            args_list.append((i, loadPath, statParas, tmpLoadData, parameter, groups))
            i += 1

    display = ProgressDisplay([f"well {a[3]}" for a in args_list], nWorkers)
    display.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        futures = [executor.submit(_process_well, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                featList.append(result['wellFeatStack'])
                labelList.append(result['label'])
                allParams.append(result['allParams'])
            display.update(result)
    display.finish()


    featStack=pd.concat(featList,axis=0,sort=False)
    #calculate the z-score of the features
    featStack=(featStack-featStack.mean())/featStack.std()
    featStack=featStack.fillna(0)
    
    labelStack=np.concatenate(labelList)
    numericIdx=np.array(list(range(len(labelStack))))
    #get the numericIdx for the samples with the labels
    labelsIn=np.array([youngLabel in l or oldLabel in l for l in labelStack])
    numericIdxTrainVal=numericIdx[labelsIn].tolist()
    numericIdxTestRest=numericIdx[~labelsIn].tolist()
    
    allParams=pd.concat(allParams,axis=0,sort=False)

    #train test split
    args_list = []
    for i, seed in enumerate(seeds):
        args_list.append((
            i, seed, saveFileName1, idxFileName1, numericIdxTrainVal, trainRatio, numericIdxTestRest, allParams, sampleGroups, numericIdx, nBoot, meanSize, featStack, labelStack, oldLabel
        ))

    display = ProgressDisplay([f"seed {s}" for s in seeds], nWorkers)
    display.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        futures = [executor.submit(_process_seed, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            display.update(future.result())
    display.finish()


def _process_well(args: tuple) -> Dict[str, Any]:
    """
    Worker function to load and process features from a single well.

    Args:
        args (tuple): A tuple containing:
            i (int): Job index.
            loadPath (str): Path to the feature data.
            statParas (List[str]): List of feature types to load.
            tmpLoadData (str): Filename identifier for the well.
            parameter (pd.DataFrame): Metadata dataframe.
            groups (List[str]): Metadata columns to define sample groups.

    Returns:
        Dict[str, Any]: A dictionary containing the aggregated features, labels, and metadata for the well.
    """
    i, loadPath, statParas, tmpLoadData, parameter, groups = args
    wellFeatStack=pd.DataFrame()
    for statPara in statParas:
        statPath=f'{loadPath}/{statPara}_{tmpLoadData}'
        tmpFeat=ezload(statPath)['cellFeats']
        print(f'{statPath} loaded')
        wellFeatStack=pd.concat([wellFeatStack,tmpFeat],axis=1,sort=False)
    #contain the number of cells in each tmpLoadData
    row=tmpLoadData.split('_r')[1].split('_f')[0]
    col=tmpLoadData.split('_c')[1].split('_r')[0]
    well_index = f"{('00' + row)[-3:]}{('00' + col)[-3:]}"
    tmpCond='_'.join(parameter.loc[parameter['WellIndex']==well_index][groups].values[0])
    tmpAllParam=parameter.loc[parameter['WellIndex']==well_index]

    label = np.repeat(tmpCond,len(wellFeatStack))
    allParams = pd.concat([tmpAllParam]*len(wellFeatStack), ignore_index=True)

    return {'index': i, 'wellFeatStack': wellFeatStack, 'label': label, 'allParams': allParams}


def _process_seed(args: tuple) -> Dict[str, Any]:
    """
    Worker function to run the training and validation for a single random seed.

    This includes train/test splitting, bootstrapping, SVM training, projecting data,
    and saving the results.

    Args:
        args (tuple): A tuple containing all necessary data and parameters for one seed run.

    Returns:
        Dict[str, Any]: A dictionary with the job index and an error message if applicable.
    """
    i, seed, saveFileName1, idxFileName1, numericIdxTrainVal, trainRatio, numericIdxTestRest, allParams, sampleGroups, numericIdx, nBoot, meanSize, featStack, labelStack, oldLabel = args
    saveFileNameSeed=saveFileName1.replace('SEEDNUM',f'seed{seed}')
    idxFileNameSeed=idxFileName1.replace('SEEDNUM',f'seed{seed}')
    res=progressregister(saveFileNameSeed,idxFileNameSeed)
    if res:
        return {'index': i, 'error': 'skipped'}
    random.Random(seed).shuffle(numericIdxTrainVal)
    trainIdx=random.Random(seed).sample(numericIdxTrainVal,int(len(numericIdxTrainVal)*trainRatio))
    testIdxVal=list(set(numericIdxTrainVal)-set(trainIdx))
    testIdx=testIdxVal+numericIdxTestRest

    #bootstrap for each sample for training
    trainLabelList=[]
    trainSampleList=[]
    trainFeatList=[]
    sampleList=np.array(['_'.join(i) for i in allParams[sampleGroups].to_numpy()])
    uniqueSamplesTrain=np.unique(sampleList[trainIdx])
    for sample in uniqueSamplesTrain:
        #find overlapping samples between the label and the uniqueSamples
        unique,count=np.unique(np.concatenate([numericIdx[sampleList==sample],trainIdx]),return_counts=True)
        #get the taining sample
        tmpSampleIdx=unique[count==2]
        #bootstrap the training sample
        np.random.seed(seed)
        bootstrap=np.random.choice(tmpSampleIdx.tolist(),nBoot*meanSize,replace=True)
        bootFeats=featStack.iloc[bootstrap]
        #calculate the mean value every 200 cells
        bootFeats=bootFeats.groupby(np.arange(len(bootFeats))//meanSize).mean()
        trainFeatList.append(bootFeats)
        trainSampleList.append(np.repeat(sample,len(bootFeats)))
        trainLabelList.append(np.repeat(labelStack[bootstrap[0]],len(bootFeats)))
    trainFeatList=pd.concat(trainFeatList,axis=0,sort=False)
    trainSampleList=np.concatenate(trainSampleList)
    trainLabelList=np.concatenate(trainLabelList)

    #bootstrap for each sample for test
    testLabelList=[]
    testSampleList=[]
    testFeatList=[]
    uniqueSamplesTest=np.unique(sampleList[testIdx])
    for sample in uniqueSamplesTest:
        #find overlapping samples between the label and the uniqueSamples
        unique,count=np.unique(np.concatenate([numericIdx[sampleList==sample],testIdx]),return_counts=True)
        #get the taining sample
        tmpSampleIdx=unique[count==2]
        #bootstrap the testing sample
        np.random.seed(seed)
        bootstrap=np.random.choice(tmpSampleIdx.tolist(),nBoot*meanSize,replace=True)
        bootFeats=featStack.iloc[bootstrap]
        #calculate the mean value every 200 cells
        bootFeats=bootFeats.groupby(np.arange(len(bootFeats))//meanSize).mean()
        testFeatList.append(bootFeats)
        testSampleList.append(np.repeat(sample,len(bootFeats)))
        testLabelList.append(np.repeat(labelStack[bootstrap[0]],len(bootFeats)))

    testFeatList=pd.concat(testFeatList,axis=0,sort=False)
    testSampleList=np.concatenate(testSampleList)
    testLabelList=np.concatenate(testLabelList)

    #obtain the ImAge axis using logistic regression
    model = svm.SVC(kernel='linear',
                    verbose=True,
                    class_weight='balanced',
                    probability=False,
                    ) #initialize the SVM model
    model.fit(trainFeatList,  [oldLabel in l for l in trainLabelList])

    #calculate the projection of the training and testing samples to the axis
    trainProj=model.decision_function(trainFeatList)
    testProj=model.decision_function(testFeatList)

    #get coefficient as the ImAge axis
    ImAgeAxis=model.coef_[0]

    #save the results
    ezsave({'trainProj':trainProj,
            'trainSampleList':trainSampleList,
            'trainLabelList':trainLabelList,
            'testProj':testProj,
            'testSampleList':testSampleList,
            'testLabelList':testLabelList,
            'ImAgeAxis':ImAgeAxis,
            'refLabel':oldLabel,
            },
            file=saveFileNameSeed)
    idxremover(idxFileNameSeed)
    return {'index': i}








