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

#import self defined functions========
from subfunctions.paramerge import paramerge
from subfunctions.loadPathGenerator import loadPathGenerator
from subfunctions.dir_rmv_file import dir_rmv_file
from subfunctions.progressregister import progressregister
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
    res=progressregister(saveFileName1,idxFileName1)
    if res:
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
                            
            featList.append(wellFeatStack)
            labelList.append(np.repeat(tmpCond,len(wellFeatStack)))
            allParams.append(pd.concat([tmpAllParam]*len(wellFeatStack), ignore_index=True))

    featStack=pd.concat(featList,axis=0,sort=False)
    #calculate the z-score of the features
    featStack=(featStack-featStack.mean())/featStack.std()
    featStack=featStack.fillna(0)
    
    labelStack=np.concatenate(labelList)
    numericIdx=np.array(list(range(len(labelStack))))
    #get the numericIdx for the samples with the labels
    labelsIn=[]
    for label in labels:
        labelsIn.append(labelStack==label)
    labelsIn=np.array(labelsIn).any(axis=0)
    numericIdxTrain=numericIdx[labelsIn].tolist()
    
    allParams=pd.concat(allParams,axis=0,sort=False)
    
    #train test split 
    for seed in seeds:
        random.Random(seed).shuffle(numericIdxTrain)
        trainIdx=random.Random(seed).sample(numericIdxTrain,int(len(numericIdxTrain)*trainRatio))
        
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
        testIdx=list(set(numericIdx)-set(trainIdx))
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
        
        #obtain the EpiAge axis using logistic regression
        model = svm.SVC(kernel='linear', 
                        verbose=True,
                        class_weight='balanced',
                        probability=False,
                        ) #initialize the SVM model
        model.fit(trainFeatList, trainLabelList==refLabel)
        
        #calculate the projection of the training and testing samples to the axis
        trainProj=model.decision_function(trainFeatList)
        testProj=model.decision_function(testFeatList)
        
        #get coefficient as the EpiAge axis
        EpiAgeAxis=model.coef_[0]
            
        #save the results
        ezsave({'trainProj':trainProj,
                'trainSampleList':trainSampleList,
                'trainLabelList':trainLabelList,
                'testProj':testProj,
                'testSampleList':testSampleList,
                'testLabelList':testLabelList,
                'EpiAgeAxis':EpiAgeAxis,
                'refLabel':refLabel,
                },
                file=saveFileName1)
        idxremover(idxFileName1)

        
        
        
        

    

