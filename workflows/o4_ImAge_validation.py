'''
Description o4_ImAge_validation
========================================
Construct one class SVM for individual samples
========================================
Kenta Ninomiya @ Sanford Burnham Prebys Medical Discovery Institute: 2023/07/21
'''

#import modules=======================
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn import svm
import concurrent.futures
import threading
#import self defined functions========
from subfunctions.paramerge import paramerge
from subfunctions.loadPathGenerator import loadPathGenerator
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
from subfunctions.progress_display import ProgressDisplay
from subfunctions.idxremover import idxremover
from subfunctions.imreqant import imreqant
from subfunctions.ezsave import ezsave
from subfunctions.ezload import ezload
#=====================================

def o4_ImAge_validation(projects=['project'],
                            statParas=['TAS'],
                            contents=[
                                'DAPI',
                                'H3K4me1',
                                'H3K27ac',
                                      ],
                            meanSize=200,
                            nBoot=1000,
                            groups=[
                                'ExperimentalCondition',
                                    ],
                            labels=['young_i4F_untreated','old_i4F_untreated'],
                            sampleGroups=['Passage','ExperimentalCondition'],
                            segCh='DAPI',
                            illumiCorrection=True,
                            seeds=[0,1,2,3,4,5],
                            youngLabel='young',
                            oldLabel='old',
                            nWorkers=4,
                            ):
    #Initialization=======================
    parameter=paramerge(paramPath='../Data/Results/Parameters', projects=[project.split('iC_')[-1] for project in projects])
    loadPaths0=loadPathGenerator(loadPath='../Data/Results', 
                                projects=projects, 
                                loadFolder='o3_extract_features')
    savePath='../Data/Results/'+'_'.join(projects)+'/o4_ImAge_validation'
    os.makedirs(savePath, exist_ok=True)
    trainRatio=0.75
        
    saveNameAdd=''
    if segCh!='DAPI':
        saveNameAdd='seg_'+segCh
    if illumiCorrection==False:
        saveNameAdd='noIC_'+saveNameAdd
        
    #reorder them by alphabetical order
    contents.sort()
    groups.sort()
    sampleGroups.sort()
    labels.sort()
    #======================================
    # Check the existence of results (if exists, calculation is skipped)=======================
    saveFileName1=f'{savePath}/{saveNameAdd}_{statPara}_{"_".join(contents)}_meanS{meanSize}_nBoot{nBoot}_SEEDNUM_'+'_'.join(groups)+'_'+'_'.join(labels)+'.pickle'
    idxFileName1=f'{savePath}/.{saveNameAdd}_{statPara}_{"_".join(contents)}_meanS{meanSize}_nBoot{nBoot}_SEEDNUM_'+'_'.join(groups)+'_'+'_'.join(labels)+'.pickle'

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


def _process_well(args):
    i, loadPath, statParas, tmpLoadData, parameter, groups = args
    wellFeatStack=pd.DataFrame()
    for statPara in statParas:
        statPath=loadPath+'/'+statPara+'_'+tmpLoadData
        tmpFeat=ezload(statPath)['cellFeats']
        print(statPath+' loaded')
        wellFeatStack=pd.concat([wellFeatStack,tmpFeat],axis=1,sort=False)
    #contain the number of cells in each tmpLoadData
    row=tmpLoadData.split('_r')[1].split('_f')[0]
    col=tmpLoadData.split('_c')[1].split('_r')[0]
    tmpCond='_'.join(parameter.loc[parameter['WellIndex']==('00'+row)[-3:]+('00'+col)[-3:]][groups].values[0])
    tmpAllParam=parameter.loc[parameter['WellIndex']==('00'+row)[-3:]+('00'+col)[-3:]]

    label = np.repeat(tmpCond,len(wellFeatStack))
    allParams = pd.concat([tmpAllParam]*len(wellFeatStack), ignore_index=True)

    return {'index': i, 'wellFeatStack': wellFeatStack, 'label': label, 'allParams': allParams}


def _process_seed(args):
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








